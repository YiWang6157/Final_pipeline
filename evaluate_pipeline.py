"""
Evaluate final_pipeline extraction outputs against GT.

Uses the same exact-match metrics as 0701_tnlr / Georgia:
  - conditions paired by key (EE: field; LR/notice/cert: condition_name)
  - scalar: operator + required_value + unit
  - nested checks: recursive by field
  - word groups / word lists: coverage in extracted text
  - primary score: all_comparisons_match (exact)

Writes (under <run_output>/evaluation/):
  evaluation_summary.json
  evaluation_table.json
  evaluation_table.csv
  evaluation_by_part.json

Usage:
  python final_pipeline/evaluate_pipeline.py
  python final_pipeline/evaluate_pipeline.py --run-dir final_pipeline/output
  python final_pipeline/evaluate_pipeline.py --run-dir <path> --jurisdictions federal TN
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent
FUNCTION_DIR = PIPELINE_DIR / "function"
DATA_DIR = PIPELINE_DIR / "data"

if str(FUNCTION_DIR) not in sys.path:
    sys.path.insert(0, str(FUNCTION_DIR))

from common import load_json, reason_slug, save_json, utc_now_iso  # noqa: E402

KNOWN_LEAVE_REASONS = [
    "adoption",
    "birth or pregnancy",
    "health condition",
    "military exigency",
    "military caregiver",
]

JURISDICTIONS = ["federal", "CA", "GA", "TN"]


# ---------------------------------------------------------------------------
# Shared exact-match comparison (tnlr / Georgia)
# ---------------------------------------------------------------------------

def is_filled(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, list):
        return len(value) > 0
    return True


def condition_is_filled(cond: Dict[str, Any]) -> bool:
    return is_filled(cond.get("operator")) and is_filled(cond.get("required_value"))


def collect_natural_language_text(node: Any) -> str:
    parts: List[str] = []
    if isinstance(node, dict):
        for key in ("field", "notes", "source_text", "citation", "unit"):
            val = node.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
        rv = node.get("required_value")
        if isinstance(rv, str) and rv.strip():
            parts.append(rv)
        elif isinstance(rv, list):
            for item in rv:
                if isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(collect_natural_language_text(item))
        elif isinstance(rv, dict):
            parts.append(collect_natural_language_text(rv))
        elif rv is not None and not isinstance(rv, (dict, list)):
            parts.append(str(rv))
    elif isinstance(node, list):
        for item in node:
            parts.append(collect_natural_language_text(item))
    elif isinstance(node, str):
        parts.append(node)
    return " ".join(p for p in parts if p).lower()


def collect_string_phrases(node: Any) -> List[str]:
    phrases: List[str] = []
    if isinstance(node, dict):
        rv = node.get("required_value")
        if isinstance(rv, str) and rv.strip():
            phrases.append(rv)
        elif isinstance(rv, list):
            for item in rv:
                if isinstance(item, str):
                    phrases.append(item)
                else:
                    phrases.extend(collect_string_phrases(item))
    elif isinstance(node, list):
        for item in node:
            phrases.extend(collect_string_phrases(item))
    return phrases


def normalize_word(w: str) -> str:
    return re.sub(r"\s+", " ", str(w).strip().lower())


def words_covered_by_extracted(words: List[str], ext_scope: Any) -> bool:
    text = collect_natural_language_text(ext_scope)
    phrases = [normalize_word(p) for p in collect_string_phrases(ext_scope)]
    for w in words:
        token = normalize_word(w)
        if not token:
            continue
        if token in text:
            continue
        if any(token in phrase for phrase in phrases):
            continue
        return False
    return True


def units_match_for_eval(gt_unit: Any, ext_unit: Any) -> bool:
    if gt_unit == ext_unit:
        return True
    if gt_unit in (None, "") and ext_unit in (None, ""):
        return True
    if gt_unit in (None, "") or ext_unit in (None, ""):
        return False
    g = normalize_word(str(gt_unit))
    e = normalize_word(str(ext_unit))
    if g == e:
        return True
    if g.startswith(e + " ") or g.startswith(e) or e in g:
        return True
    return g.split()[0] == e.split()[0] if g and e else False


def is_word_group(item: Any) -> bool:
    return isinstance(item, list) and all(isinstance(w, str) for w in item)


def classify_required_value(rv: Any) -> str:
    if rv is None:
        return "empty"
    if isinstance(rv, bool) or isinstance(rv, (int, float)):
        return "scalar"
    if isinstance(rv, str):
        return "scalar"
    if not isinstance(rv, list) or not rv:
        return "other"
    if all(isinstance(x, dict) for x in rv):
        return "nested_checks"
    if all(is_word_group(x) for x in rv):
        return "word_groups"
    if all(isinstance(x, str) for x in rv):
        return "word_list"
    return "other"


def find_nested_by_field(nodes: Any, field_name: str) -> Optional[Dict[str, Any]]:
    if not isinstance(nodes, list):
        return None
    target = field_name.strip().lower()
    for node in nodes:
        if isinstance(node, dict) and str(node.get("field", "")).strip().lower() == target:
            return node
    return None


def compare_word_groups(
    gt_rv: List[Any], ext_node: Dict[str, Any], path: str, ext_scope: Any
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    kind = classify_required_value(gt_rv)
    groups = [gt_rv] if kind == "word_list" else (gt_rv if kind == "word_groups" else [])
    for i, group in enumerate(groups):
        words = [normalize_word(w) for w in group if normalize_word(w)]
        results.append(
            {
                "path": f"{path}.required_value[{i}]",
                "type": "word_group",
                "words": words,
                "match": words_covered_by_extracted(words, ext_scope),
                "natural_language_excerpt": collect_natural_language_text(ext_scope)[:300],
            }
        )
    return results


def compare_nodes(
    gt_node: Dict[str, Any],
    ext_node: Optional[Dict[str, Any]],
    path: str,
    ext_scope: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Exact-match comparisons against GT (same logic as 0701_tnlr)."""
    results: List[Dict[str, Any]] = []
    if ext_node is None:
        results.append({"path": path, "match": False, "reason": "missing in extracted output"})
        return results

    scope = ext_scope if ext_scope is not None else ext_node
    filled = (
        condition_is_filled(ext_node)
        if "operator" in ext_node
        else is_filled(ext_node.get("required_value"))
    )
    results.append({"path": path, "type": "filled", "filled": filled, "match": filled})

    gt_rv = gt_node.get("required_value")
    ext_rv = ext_node.get("required_value")
    gt_kind = classify_required_value(gt_rv)

    if gt_kind in ("word_list", "word_groups"):
        results.extend(compare_word_groups(gt_rv, ext_node, path, scope))
    elif gt_kind == "scalar":
        op_match = ext_node.get("operator") == gt_node.get("operator")
        val_match = ext_rv == gt_rv
        unit_match = units_match_for_eval(gt_node.get("unit"), ext_node.get("unit"))
        results.append(
            {
                "path": f"{path}.required_value",
                "type": "scalar",
                "match": op_match and val_match and unit_match,
                "operator_match": op_match,
                "value_match": val_match,
                "unit_match": unit_match,
                "expected": {
                    "operator": gt_node.get("operator"),
                    "required_value": gt_rv,
                    "unit": gt_node.get("unit"),
                },
                "actual": {
                    "operator": ext_node.get("operator"),
                    "required_value": ext_rv,
                    "unit": ext_node.get("unit"),
                },
            }
        )
    elif gt_kind == "nested_checks":
        for gt_child in gt_rv:
            field = gt_child.get("field", "?")
            ext_child = (
                find_nested_by_field(ext_rv, field) if isinstance(ext_rv, list) else None
            )
            results.extend(compare_nodes(gt_child, ext_child, f"{path}/{field}", scope))
    return results


def _failed_word_groups(comparisons: List[Dict[str, Any]]) -> List[str]:
    return [
        " ".join(cmp.get("words", []))
        for cmp in comparisons
        if cmp.get("type") == "word_group" and cmp.get("match") is False
    ]


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

def normalize_ee_payload(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and isinstance(obj.get("employee_eligibility_conditions"), list):
        return obj
    if isinstance(obj, list):
        return {"employee_eligibility_conditions": obj}
    raise ValueError("EE payload missing employee_eligibility_conditions list")


def normalize_conditions_payload(obj: Any) -> Dict[str, Any]:
    """Accept {conditions: [...]}, bare condition object, or bare list."""
    if isinstance(obj, dict):
        if isinstance(obj.get("conditions"), list):
            return obj
        if obj.get("condition_name"):
            return {"conditions": [obj]}
    if isinstance(obj, list):
        return {"conditions": obj}
    raise ValueError("Payload missing conditions list")


def index_by_key(
    conditions: List[Any], key_field: str
) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        key = cond.get(key_field)
        if key:
            idx[str(key)] = cond
    return idx


def evaluate_by_key(
    *,
    part: str,
    jurisdiction: str,
    extracted: Dict[str, Any],
    gt: Dict[str, Any],
    key_field: str,
    conditions_key: str,
    extracted_path: Path,
    gt_path: Path,
) -> Dict[str, Any]:
    """Pair GT/extracted conditions by key_field and run exact-match comparisons."""
    ext_idx = index_by_key(extracted.get(conditions_key, []), key_field)
    gt_idx = index_by_key(gt.get(conditions_key, []), key_field)

    gt_names = set(gt_idx)
    ext_names = set(ext_idx)
    new_names = sorted(ext_names - gt_names)
    missing_names = sorted(gt_names - ext_names)

    condition_results: List[Dict[str, Any]] = []
    for name in sorted(gt_names):
        gt_cond = gt_idx[name]
        ext_cond = ext_idx.get(name)
        comparisons = (
            compare_nodes(gt_cond, ext_cond, name, ext_cond)
            if ext_cond
            else [{"path": name, "match": False, "reason": "missing in extracted output"}]
        )
        comp_matches = [c.get("match") for c in comparisons if "match" in c]
        condition_results.append(
            {
                "condition_name": name,
                "in_gt": True,
                "in_extracted": ext_cond is not None,
                "filled_in_extracted": condition_is_filled(ext_cond) if ext_cond else False,
                "is_new_extracted": False,
                "comparisons": comparisons,
                "all_comparisons_match": all(comp_matches) if comp_matches else False,
            }
        )
    for name in new_names:
        condition_results.append(
            {
                "condition_name": name,
                "in_gt": False,
                "in_extracted": True,
                "filled_in_extracted": condition_is_filled(ext_idx[name]),
                "is_new_extracted": True,
                "comparisons": [],
                "all_comparisons_match": None,
            }
        )

    gt_items = [c for c in condition_results if c["in_gt"]]
    matched = sum(1 for c in gt_items if c.get("all_comparisons_match"))
    filled = sum(1 for c in gt_items if c["filled_in_extracted"])
    gt_count = len(gt_names)

    return {
        "jurisdiction": jurisdiction,
        "part": part,
        "status": "ok",
        "extracted_path": str(extracted_path),
        "gt_path": str(gt_path),
        "summary": {
            "gt_condition_count": gt_count,
            "extracted_condition_count": len(ext_names),
            "gt_conditions_filled_in_extracted": filled,
            "gt_conditions_matched": matched,
            "exact_match_rate": round(matched / gt_count, 4) if gt_count else None,
            "new_extracted_conditions": new_names,
            "missing_from_extracted": missing_names,
        },
        "conditions": condition_results,
    }


# ---------------------------------------------------------------------------
# GT / output discovery
# ---------------------------------------------------------------------------

def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def find_ee_gt(gt_dir: Path) -> Optional[Path]:
    return first_existing(
        [
            gt_dir / "ee_gt.json",
            gt_dir / "ca_ee_gt.json",
            gt_dir / "tn_ee_gt.json",
            gt_dir / "ga_ee_gt.json",
        ]
    )


def find_lr_gt(gt_dir: Path, reason: str) -> Optional[Path]:
    slug = reason_slug(reason)
    return first_existing(
        [
            gt_dir / f"{slug}_gt.json",
            gt_dir / f"tn_{slug}_gt.json",
            gt_dir / f"ca_{slug}_gt.json",
            gt_dir / f"ga_{slug}_gt.json",
        ]
    )


def find_notice_gt(gt_dir: Path) -> Optional[Path]:
    return first_existing(
        [
            gt_dir / "notice_gt.json",
            gt_dir / "ca_notice_gt.json",
            gt_dir / "tn_notice_gt.json",
        ]
    )


def find_cert_gt(gt_dir: Path) -> Optional[Path]:
    return first_existing(
        [
            gt_dir / "cert_gt.json",
            gt_dir / "ca_cert_gt.json",
            gt_dir / "tn_cert_gt.json",
        ]
    )


def discover_lr_extracted(jurisdiction_out: Path) -> List[Tuple[str, Path]]:
    """Return [(leave_reason, path)] for *_lr_extracted.json files."""
    found: List[Tuple[str, Path]] = []
    for path in sorted(jurisdiction_out.glob("*_lr_extracted.json")):
        stem = path.name[: -len("_lr_extracted.json")]
        reason = None
        for known in KNOWN_LEAVE_REASONS:
            if reason_slug(known) == stem:
                reason = known
                break
        if reason is None:
            reason = stem.replace("_", " ")
        found.append((reason, path))
    return found


def skipped_report(
    jurisdiction: str, part: str, reason: str
) -> Dict[str, Any]:
    return {
        "jurisdiction": jurisdiction,
        "part": part,
        "status": "skipped",
        "reason": reason,
        "summary": {},
        "conditions": [],
    }


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_evaluation_table_rows(all_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in all_reports:
        jurisdiction = report.get("jurisdiction", "")
        part = report.get("part", "unknown")
        if report.get("status") == "skipped":
            rows.append(
                {
                    "jurisdiction": jurisdiction,
                    "part": part,
                    "row_type": "skipped",
                    "condition_name": None,
                    "in_gt": None,
                    "in_extracted": None,
                    "filled_in_extracted": None,
                    "all_comparisons_match": None,
                    "is_new_extracted": None,
                    "gt_condition_count": None,
                    "extracted_condition_count": None,
                    "exact_match_rate": None,
                    "new_extracted_conditions": None,
                    "missing_from_extracted": None,
                    "failed_word_groups": None,
                    "skip_reason": report.get("reason"),
                }
            )
            continue

        s = report["summary"]
        rows.append(
            {
                "jurisdiction": jurisdiction,
                "part": part,
                "row_type": "part_summary",
                "condition_name": None,
                "in_gt": None,
                "in_extracted": None,
                "filled_in_extracted": s.get("gt_conditions_filled_in_extracted"),
                "all_comparisons_match": None,
                "is_new_extracted": None,
                "gt_condition_count": s.get("gt_condition_count"),
                "extracted_condition_count": s.get("extracted_condition_count"),
                "exact_match_rate": s.get("exact_match_rate"),
                "new_extracted_conditions": (
                    ", ".join(s.get("new_extracted_conditions") or []) or None
                ),
                "missing_from_extracted": (
                    ", ".join(s.get("missing_from_extracted") or []) or None
                ),
                "failed_word_groups": None,
                "skip_reason": None,
            }
        )
        for cond in report.get("conditions", []):
            failed = _failed_word_groups(cond.get("comparisons", []))
            rows.append(
                {
                    "jurisdiction": jurisdiction,
                    "part": part,
                    "row_type": "condition",
                    "condition_name": cond.get("condition_name"),
                    "in_gt": cond.get("in_gt"),
                    "in_extracted": cond.get("in_extracted"),
                    "filled_in_extracted": cond.get("filled_in_extracted"),
                    "all_comparisons_match": cond.get("all_comparisons_match"),
                    "is_new_extracted": cond.get("is_new_extracted"),
                    "gt_condition_count": None,
                    "extracted_condition_count": None,
                    "exact_match_rate": None,
                    "new_extracted_conditions": None,
                    "missing_from_extracted": None,
                    "failed_word_groups": "; ".join(failed) if failed else None,
                    "skip_reason": None,
                }
            )
    return rows


def build_part_rollups(all_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rollups: List[Dict[str, Any]] = []
    for report in all_reports:
        jurisdiction = report.get("jurisdiction", "")
        part = report.get("part", "unknown")
        if report.get("status") == "skipped":
            rollups.append(
                {
                    "jurisdiction": jurisdiction,
                    "part": part,
                    "status": "skipped",
                    "reason": report.get("reason"),
                }
            )
            continue
        s = report["summary"]
        rollups.append(
            {
                "jurisdiction": jurisdiction,
                "part": part,
                "status": "ok",
                "gt_condition_count": s["gt_condition_count"],
                "extracted_condition_count": s["extracted_condition_count"],
                "gt_conditions_filled": s["gt_conditions_filled_in_extracted"],
                "gt_conditions_matched": s["gt_conditions_matched"],
                "exact_match_rate": s["exact_match_rate"],
                "new_extracted_conditions": s["new_extracted_conditions"],
                "missing_from_extracted": s["missing_from_extracted"],
            }
        )
    return rollups


def save_evaluation_artifacts(
    all_reports: List[Dict[str, Any]],
    run_dir: Path,
    eval_dir: Path,
) -> Dict[str, Path]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    table_rows = build_evaluation_table_rows(all_reports)
    rollups = build_part_rollups(all_reports)

    paths = {
        "summary": eval_dir / "evaluation_summary.json",
        "table_json": eval_dir / "evaluation_table.json",
        "table_csv": eval_dir / "evaluation_table.csv",
        "rollup_json": eval_dir / "evaluation_by_part.json",
    }

    save_json(
        {
            "evaluated_at": utc_now_iso(),
            "run_output_dir": str(run_dir),
            "eval_output_dir": str(eval_dir),
            "part_rollups": rollups,
            "table_rows": table_rows,
            "results": all_reports,
        },
        paths["summary"],
    )
    save_json(table_rows, paths["table_json"])
    save_json(rollups, paths["rollup_json"])

    if table_rows:
        fieldnames = list(table_rows[0].keys())
        with open(paths["table_csv"], "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(table_rows)

    return paths


def print_eval_report(report: Dict[str, Any]) -> None:
    label = f"{report.get('jurisdiction')}/{report.get('part')}"
    if report.get("status") == "skipped":
        print(f"  [SKIP] {label}: {report.get('reason')}")
        return
    s = report["summary"]
    rate = s.get("exact_match_rate")
    rate_s = f"{rate:.0%}" if isinstance(rate, float) else "n/a"
    print(
        f"  [OK]   {label}: matched {s['gt_conditions_matched']}/"
        f"{s['gt_condition_count']} ({rate_s} exact), "
        f"filled {s['gt_conditions_filled_in_extracted']}/{s['gt_condition_count']}"
    )


# ---------------------------------------------------------------------------
# Per-jurisdiction evaluation
# ---------------------------------------------------------------------------

def evaluate_jurisdiction(
    jurisdiction: str,
    jurisdiction_out: Path,
    gt_dir: Path,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []

    # EE
    ee_out = jurisdiction_out / "ee_extracted.json"
    ee_gt = find_ee_gt(gt_dir)
    if not ee_out.exists():
        reports.append(skipped_report(jurisdiction, "ee", f"missing output: {ee_out.name}"))
    elif ee_gt is None:
        reports.append(skipped_report(jurisdiction, "ee", f"missing GT under {gt_dir}"))
    else:
        try:
            reports.append(
                evaluate_by_key(
                    part="ee",
                    jurisdiction=jurisdiction,
                    extracted=normalize_ee_payload(load_json(ee_out)),
                    gt=normalize_ee_payload(load_json(ee_gt)),
                    key_field="field",
                    conditions_key="employee_eligibility_conditions",
                    extracted_path=ee_out,
                    gt_path=ee_gt,
                )
            )
        except Exception as exc:
            reports.append(skipped_report(jurisdiction, "ee", f"eval error: {exc}"))

    # LR
    lr_files = discover_lr_extracted(jurisdiction_out)
    if not lr_files:
        reports.append(
            skipped_report(jurisdiction, "lr", "no *_lr_extracted.json files found")
        )
    else:
        for reason, lr_out in lr_files:
            part = f"lr:{reason}"
            lr_gt = find_lr_gt(gt_dir, reason)
            if lr_gt is None:
                reports.append(
                    skipped_report(
                        jurisdiction, part, f"missing GT for leave reason {reason!r}"
                    )
                )
                continue
            try:
                reports.append(
                    evaluate_by_key(
                        part=part,
                        jurisdiction=jurisdiction,
                        extracted=normalize_conditions_payload(load_json(lr_out)),
                        gt=normalize_conditions_payload(load_json(lr_gt)),
                        key_field="condition_name",
                        conditions_key="conditions",
                        extracted_path=lr_out,
                        gt_path=lr_gt,
                    )
                )
            except Exception as exc:
                reports.append(skipped_report(jurisdiction, part, f"eval error: {exc}"))

    # Notice / cert
    for category, finder, filename in (
        ("notice", find_notice_gt, "notice.json"),
        ("certification", find_cert_gt, "certification.json"),
    ):
        out_path = jurisdiction_out / filename
        gt_path = finder(gt_dir)
        if not out_path.exists():
            reports.append(
                skipped_report(jurisdiction, category, f"missing output: {filename}")
            )
        elif gt_path is None:
            reports.append(
                skipped_report(jurisdiction, category, f"missing GT under {gt_dir}")
            )
        else:
            try:
                reports.append(
                    evaluate_by_key(
                        part=category,
                        jurisdiction=jurisdiction,
                        extracted=normalize_conditions_payload(load_json(out_path)),
                        gt=normalize_conditions_payload(load_json(gt_path)),
                        key_field="condition_name",
                        conditions_key="conditions",
                        extracted_path=out_path,
                        gt_path=gt_path,
                    )
                )
            except Exception as exc:
                reports.append(
                    skipped_report(jurisdiction, category, f"eval error: {exc}")
                )

    return reports


def evaluate_run(
    run_dir: Path,
    jurisdictions: Optional[Sequence[str]] = None,
    eval_dir: Optional[Path] = None,
) -> Path:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run output directory not found: {run_dir}")

    selected = list(jurisdictions) if jurisdictions else [
        j for j in JURISDICTIONS if (run_dir / j).is_dir()
    ]
    if not selected:
        # Fall back: any subdir that looks like a jurisdiction folder with outputs
        selected = sorted(
            p.name for p in run_dir.iterdir() if p.is_dir() and p.name != "evaluation"
        )

    eval_dir = Path(eval_dir).resolve() if eval_dir else run_dir / "evaluation"
    print(f"RUN DIR   : {run_dir}")
    print(f"EVAL DIR  : {eval_dir}")
    print(f"Jurisdictions: {selected}")

    all_reports: List[Dict[str, Any]] = []
    for jurisdiction in selected:
        jurisdiction_out = run_dir / jurisdiction
        gt_dir = DATA_DIR / jurisdiction / "gt"
        print(f"\n{'=' * 60}")
        print(f"Evaluating {jurisdiction}")
        print(f"  output: {jurisdiction_out}")
        print(f"  gt    : {gt_dir}")

        if not jurisdiction_out.is_dir():
            report = skipped_report(
                jurisdiction, "*", f"missing jurisdiction output dir: {jurisdiction_out}"
            )
            all_reports.append(report)
            print_eval_report(report)
            continue
        if not gt_dir.is_dir():
            report = skipped_report(
                jurisdiction, "*", f"missing GT dir: {gt_dir}"
            )
            all_reports.append(report)
            print_eval_report(report)
            continue

        reports = evaluate_jurisdiction(jurisdiction, jurisdiction_out, gt_dir)
        for report in reports:
            print_eval_report(report)
        all_reports.extend(reports)

    paths = save_evaluation_artifacts(all_reports, run_dir, eval_dir)
    print(f"\n{'=' * 60}")
    print("Wrote evaluation artifacts:")
    for name, path in paths.items():
        print(f"  {name:12} → {path}")
    return eval_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final_pipeline outputs against GT (exact-match metrics)."
    )
    parser.add_argument(
        "--run-dir",
        default=str(PIPELINE_DIR / "output"),
        help="Path to extraction output folder (default: final_pipeline/output).",
    )
    parser.add_argument(
        "--jurisdictions",
        nargs="+",
        default=None,
        help="Optional subset of jurisdictions to evaluate.",
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Override evaluation output directory (default: <run-dir>/evaluation).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_run(
        run_dir=Path(args.run_dir),
        jurisdictions=args.jurisdictions,
        eval_dir=Path(args.eval_dir) if args.eval_dir else None,
    )
