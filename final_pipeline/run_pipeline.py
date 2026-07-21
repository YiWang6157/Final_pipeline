"""
Run all final_pipeline extractions across jurisdictions.

Loops federal / CA / GA / TN under data/, runs:
  1. EE extraction
  2. LR extraction for each leave reason that has leave-eligibility chunks
     (and a matching schema file)
  3. Notice + certification extraction (when notice_requirements chunks exist)

Outputs:
  final_pipeline/output/
    {jurisdiction}/
      ee_extracted.json
      {reason}_lr_extracted.json
      notice.json
      certification.json
    run_summary.json

Usage (from repo root or final_pipeline/):
  python final_pipeline/run_pipeline.py
  python final_pipeline/run_pipeline.py --jurisdictions federal GA
  python final_pipeline/run_pipeline.py --skip-critique
  python final_pipeline/run_pipeline.py --skip-ee --skip-notice-cert
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PIPELINE_DIR = Path(__file__).resolve().parent
FUNCTION_DIR = PIPELINE_DIR / "function"
DATA_DIR = PIPELINE_DIR / "data"
DEFAULT_OUTPUT_ROOT = PIPELINE_DIR / "output"

if str(FUNCTION_DIR) not in sys.path:
    sys.path.insert(0, str(FUNCTION_DIR))

from common import (  # noqa: E402
    chunk_matches_function,
    chunk_matches_reason_and_function,
    load_chunks,
    normalize_to_list,
    reason_slug,
    save_json,
    utc_now_iso,
)
from ee_extractor import extract_ee_conditions  # noqa: E402
from lr_extractor import extract_lr_conditions  # noqa: E402
from notice_cert_extractor import extract_notice_cert  # noqa: E402

# ---------------------------------------------------------------------------
# Jurisdiction data map (edit chunk filenames here if data layout changes)
# ---------------------------------------------------------------------------

JURISDICTIONS: Dict[str, Dict[str, Any]] = {
    "federal": {
        "chunk_files": ["fmla_chunk_schema_filled.json"],
        "schema_dir": "schema",
    },
    "CA": {
        "chunk_files": [
            "cfra_chunk_schema_filled.json",
            "pdl_chunk_schema_filled.json",
        ],
        "schema_dir": "schema",
    },
    "GA": {
        "chunk_files": ["chunk_schema_filled.json"],
        "schema_dir": "schema",
    },
    "TN": {
        "chunk_files": ["tn_extracted_sections_chunk_schema_filled.json"],
        "schema_dir": "schema",
    },
}

# Canonical leave-reason labels used in chunk "reason" fields.
KNOWN_LEAVE_REASONS: List[str] = [
    "adoption",
    "birth or pregnancy",
    "health condition",
    "military exigency",
    "military caregiver",
]

TARGET_LR_FUNCTION = "leave eligibility"
TARGET_NOTICE_FUNCTION = "notice_requirements"


def schema_stem_to_leave_reason(stem: str) -> Optional[str]:
    """Map schema filename stem (e. sn. tn_health_condition) → leave reason label."""
    cleaned = stem.strip().lower()
    for prefix in ("tn_", "ca_", "ga_", "federal_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    slug = reason_slug(cleaned.replace("-", "_"))
    for reason in KNOWN_LEAVE_REASONS:
        if reason_slug(reason) == slug:
            return reason
    return None


def discover_schema_map(schema_dir: Path) -> Dict[str, Path]:
    """
    Return {leave_reason: schema_path} from schema/*.json files.
    Only includes known leave reasons.
    """
    mapping: Dict[str, Path] = {}
    if not schema_dir.is_dir():
        return mapping
    for path in sorted(schema_dir.glob("*.json")):
        reason = schema_stem_to_leave_reason(path.stem)
        if reason is None:
            print(f"  [warn] skipping unrecognized schema file: {path.name}")
            continue
        # Prefer unprefixed names if both tn_*.json and *.json somehow exist.
        if reason in mapping and path.stem.startswith(("tn_", "ca_", "ga_")):
            continue
        mapping[reason] = path
    return mapping


def leave_reasons_with_chunks(
    chunks: List[Dict[str, Any]],
    candidate_reasons: Sequence[str],
) -> List[str]:
    """Keep leave reasons that have at least one leave-eligibility chunk."""
    found: List[str] = []
    for reason in candidate_reasons:
        if any(
            chunk_matches_reason_and_function(c, reason, TARGET_LR_FUNCTION)
            for c in chunks
        ):
            found.append(reason)
    return found


def resolve_chunk_paths(jurisdiction: str, cfg: Dict[str, Any]) -> List[Path]:
    base = DATA_DIR / jurisdiction
    paths = [base / name for name in cfg["chunk_files"]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{jurisdiction}: missing chunk file(s): "
            + ", ".join(str(p) for p in missing)
        )
    return paths


def run_jurisdiction(
    jurisdiction: str,
    cfg: Dict[str, Any],
    jurisdiction_out: Path,
    *,
    run_ee: bool = True,
    run_lr: bool = True,
    run_notice_cert: bool = True,
    run_critique: bool = True,
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    """Run all enabled extraction tasks for one jurisdiction."""
    summary: Dict[str, Any] = {
        "jurisdiction": jurisdiction,
        "started_at": utc_now_iso(),
        "tasks": [],
    }
    jurisdiction_out.mkdir(parents=True, exist_ok=True)

    chunk_paths = resolve_chunk_paths(jurisdiction, cfg)
    chunks = load_chunks(chunk_paths)
    print(f"\n{'=' * 60}")
    print(f"Jurisdiction: {jurisdiction}")
    print(f"  chunks     : {len(chunks)} from {[p.name for p in chunk_paths]}")
    print(f"  output     : {jurisdiction_out}")

    def record(task: str, status: str, **extra: Any) -> None:
        entry = {"task": task, "status": status, **extra}
        summary["tasks"].append(entry)
        print(f"  [{status.upper():7}] {task}" + (f" — {extra.get('detail', '')}" if extra.get("detail") else ""))

    # ── EE ────────────────────────────────────────────────────────────────
    if run_ee:
        ee_count = sum(1 for c in chunks if chunk_matches_function(c, "employee eligibility"))
        if ee_count == 0:
            record("ee", "skipped", detail="no employee eligibility chunks")
        else:
            try:
                extract_ee_conditions(
                    chunk_file_paths=chunk_paths,
                    output_dir=jurisdiction_out,
                    output_filename="ee_extracted.json",
                )
                record("ee", "ok", detail=f"{ee_count} EE-tagged chunk(s)")
            except Exception as exc:
                record("ee", "error", detail=str(exc), traceback=traceback.format_exc())
                if not continue_on_error:
                    raise

    # ── LR (per leave reason with chunks + schema) ────────────────────────
    if run_lr:
        schema_dir = DATA_DIR / jurisdiction / cfg["schema_dir"]
        schema_map = discover_schema_map(schema_dir)
        reasons = leave_reasons_with_chunks(chunks, list(schema_map.keys()))
        # Also surface chunk-only reasons with no schema (skipped explicitly).
        chunk_only = [
            r
            for r in leave_reasons_with_chunks(chunks, KNOWN_LEAVE_REASONS)
            if r not in schema_map
        ]
        print(f"  schemas    : {sorted(schema_map)}")
        print(f"  LR reasons : {reasons}" + (f" (no schema: {chunk_only})" if chunk_only else ""))

        for reason in chunk_only:
            record(
                f"lr:{reason_slug(reason)}",
                "skipped",
                detail="leave-eligibility chunks found but no schema file",
            )

        if not reasons:
            record("lr", "skipped", detail="no leave reasons with both schema and chunks")
        else:
            for reason in reasons:
                schema_path = schema_map[reason]
                task_key = f"lr:{reason_slug(reason)}"
                try:
                    extract_lr_conditions(
                        chunk_file_paths=chunk_paths,
                        schema_file_path=schema_path,
                        leave_reason=reason,
                        output_dir=jurisdiction_out,
                        output_filename=f"{reason_slug(reason)}_lr_extracted.json",
                        run_critique=run_critique,
                    )
                    record(task_key, "ok", schema=schema_path.name)
                except Exception as exc:
                    record(
                        task_key,
                        "error",
                        detail=str(exc),
                        schema=schema_path.name,
                        traceback=traceback.format_exc(),
                    )
                    if not continue_on_error:
                        raise

    # ── Notice + certification ────────────────────────────────────────────
    if run_notice_cert:
        notice_count = sum(
            1 for c in chunks if chunk_matches_function(c, TARGET_NOTICE_FUNCTION)
        )
        if notice_count == 0:
            record("notice", "skipped", detail="no notice_requirements chunks")
            record("certification", "skipped", detail="no notice_requirements chunks")
        else:
            for category in ("notice", "certification"):
                try:
                    extract_notice_cert(
                        chunk_file_paths=chunk_paths,
                        category=category,
                        output_dir=jurisdiction_out,
                        output_filename=f"{category}.json",
                    )
                    record(category, "ok", detail=f"{notice_count} notice_requirements chunk(s)")
                except Exception as exc:
                    record(
                        category,
                        "error",
                        detail=str(exc),
                        traceback=traceback.format_exc(),
                    )
                    if not continue_on_error:
                        raise

    summary["finished_at"] = utc_now_iso()
    ok = sum(1 for t in summary["tasks"] if t["status"] == "ok")
    err = sum(1 for t in summary["tasks"] if t["status"] == "error")
    skip = sum(1 for t in summary["tasks"] if t["status"] == "skipped")
    summary["counts"] = {"ok": ok, "error": err, "skipped": skip}
    return summary


def run_pipeline(
    jurisdictions: Optional[Sequence[str]] = None,
    *,
    run_ee: bool = True,
    run_lr: bool = True,
    run_notice_cert: bool = True,
    run_critique: bool = True,
    continue_on_error: bool = True,
    output_root: Optional[Path] = None,
) -> Path:
    selected = list(jurisdictions) if jurisdictions else list(JURISDICTIONS.keys())
    unknown = [j for j in selected if j not in JURISDICTIONS]
    if unknown:
        raise ValueError(f"Unknown jurisdiction(s): {unknown}. Known: {list(JURISDICTIONS)}")

    run_out = Path(output_root).resolve() if output_root else DEFAULT_OUTPUT_ROOT
    run_out.mkdir(parents=True, exist_ok=True)

    print(f"OUTPUT : {run_out}")
    print(f"Jurisdictions: {selected}")
    print(
        f"Tasks: EE={run_ee} LR={run_lr} notice/cert={run_notice_cert} "
        f"critique={run_critique}"
    )

    all_summaries: List[Dict[str, Any]] = []
    for jurisdiction in selected:
        cfg = JURISDICTIONS[jurisdiction]
        try:
            summary = run_jurisdiction(
                jurisdiction,
                cfg,
                run_out / jurisdiction,
                run_ee=run_ee,
                run_lr=run_lr,
                run_notice_cert=run_notice_cert,
                run_critique=run_critique,
                continue_on_error=continue_on_error,
            )
        except Exception as exc:
            summary = {
                "jurisdiction": jurisdiction,
                "status": "fatal_error",
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "tasks": [],
            }
            print(f"\n  [FATAL] {jurisdiction}: {exc}")
            if not continue_on_error:
                raise
        all_summaries.append(summary)

    run_summary = {
        "extracted_at": utc_now_iso(),
        "output_dir": str(run_out),
        "jurisdictions": selected,
        "options": {
            "run_ee": run_ee,
            "run_lr": run_lr,
            "run_notice_cert": run_notice_cert,
            "run_critique": run_critique,
            "continue_on_error": continue_on_error,
        },
        "results": all_summaries,
    }
    summary_path = save_json(run_summary, run_out / "run_summary.json")
    print(f"\n{'=' * 60}")
    print(f"Done. Summary → {summary_path}")
    return run_out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EE / LR / notice-cert extraction for all jurisdictions."
    )
    parser.add_argument(
        "--jurisdictions",
        nargs="+",
        choices=list(JURISDICTIONS.keys()),
        default=None,
        help="Subset of jurisdictions (default: all).",
    )
    parser.add_argument("--skip-ee", action="store_true")
    parser.add_argument("--skip-lr", action="store_true")
    parser.add_argument("--skip-notice-cert", action="store_true")
    parser.add_argument(
        "--skip-critique",
        action="store_true",
        help="Skip LR API2 critique (critique runs by default).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error (default: continue).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: final_pipeline/output/).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        jurisdictions=args.jurisdictions,
        run_ee=not args.skip_ee,
        run_lr=not args.skip_lr,
        run_notice_cert=not args.skip_notice_cert,
        run_critique=not args.skip_critique,
        continue_on_error=not args.fail_fast,
        output_root=Path(args.output_dir) if args.output_dir else None,
    )
