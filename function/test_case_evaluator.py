"""
Shared test-case evaluator for final_pipeline GT and extracted condition packs.

Aggregation rules (per jurisdiction track):
  - EE: ALL top-level conditions must pass.
  - leave_reason / notice / cert: ANY one top-level (main) condition passing
    is enough for that part.
  - Inside a main condition: respect the node operator. ALL_OF (default when
    required_value is a list of child objects) means every sub-condition must
    pass; ANY_OF means one sub-condition is enough.

Tracks:
  - Always evaluate a federal track (EE + notice + cert + leave_reason).
  - CA / TN: also evaluate a state track in parallel; APPROVED if either passes.
  - GA: state track only when facts.is_government_worker is true; otherwise federal only.
"""

from __future__ import annotations

import operator as op
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .common import load_json, save_json
except ImportError:  # pragma: no cover - script-style import
    from common import load_json, save_json

PIPELINE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PIPELINE_DIR / "data"
TEST_CASES_DIR = DATA_DIR / "test_cases"
OUTPUT_DIR = PIPELINE_DIR / "output"

DECISION_APPROVED = "APPROVED"
DECISION_NEED_HUMAN_REVIEW = "NEED HUMAN REVIEW"

OPERATORS = {
    ">=": op.ge,
    "<=": op.le,
    ">": op.gt,
    "<": op.lt,
    "==": op.eq,
    "!=": op.ne,
}

# Claim-fact aliases → GT / extracted field labels (normalized lookup uses both).
FIELD_ALIASES: Dict[str, List[str]] = {
    "duration of incapacity": ["incapacity_duration_days", "incapacity_duration"],
    "visit count": ["treatment_count", "visit_count"],
    "visit window": ["treatment_window_days", "visit_window_days"],
    "first in-person visit timing": ["first_visit_window_days", "first_visit_timing_days"],
    "unable to perform job functions": ["unable_to_perform_job_functions"],
    "overnight stay location": ["overnight_stay_location"],
    "advance notice": ["advance_notice", "advance_notice_days", "foreseeable_leave_notice"],
    "foreseeable leave": ["foreseeable_leave", "foreseeable_need"],
    "foreseeable need": ["foreseeable_need", "foreseeable_leave"],
    "notice timing": ["notice_timing", "unforeseeable_leave_notice"],
    "30-day notice practicable": ["notice_30_days_practicable", "thirty_day_notice_practicable"],
    "30-day notice possible": ["notice_30_days_practicable", "thirty_day_notice_practicable"],
    "foreseeable leave notice": ["foreseeable_leave_notice", "advance_notice", "advance_notice_days"],
    "unforeseeable leave notice": ["unforeseeable_leave_notice", "notice_timing"],
    "medical certification provided": ["medical_certification_provided"],
    "qualifying exigency certification provided": ["qualifying_exigency_certification_provided"],
    "military caregiver certification provided": ["military_caregiver_certification_provided"],
    "time since birth": ["months_since_birth", "time_since_birth_months"],
    "time since placement": ["months_since_placement", "time_since_placement_months"],
    "employment duration months": ["employment_duration_months"],
    "hours worked last 12 months": ["hours_worked_last_12_months"],
    "hours worked preceding 6 months": ["hours_worked_preceding_6_months"],
    "employer size within 75 miles": ["employer_size_within_75_miles"],
    "scheduled hours per week": ["scheduled_hours_per_week"],
    "employee location": ["employee_location"],
    "employment type": ["employment_type"],
    "is temporary employee": ["is_temporary_employee"],
    "is hourly employee": ["is_hourly_employee"],
    "leave request approval required": ["leave_request_approval_required"],
}

MAY_REQUIRE_MARKERS = (
    "employer may require",
    "coordinator may require",
    "agency may require",
    "may require",
)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _norm_key(value: Any) -> str:
    return _norm(value).replace(" ", "_")


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _word_groups(required: Any) -> List[List[str]]:
    if required is None:
        return []
    if isinstance(required, list):
        if not required:
            return []
        if all(isinstance(x, list) for x in required):
            return [[_norm(t) for t in group] for group in required]
        if all(isinstance(x, str) for x in required):
            # Flat list of alternatives
            return [[_norm(x)] for x in required]
    return [[_norm(required)]]


# Synonyms for relationship IN checks (claim value → acceptable GT tokens).
RELATIONSHIP_ALIASES: Dict[str, List[str]] = {
    "child": ["child", "son or daughter", "son", "daughter"],
    "son or daughter": ["son or daughter", "child", "son", "daughter"],
    "son": ["son", "child", "son or daughter"],
    "daughter": ["daughter", "child", "son or daughter"],
    "parent": ["parent", "parent in law", "parent-in-law"],
    "spouse": ["spouse", "domestic partner"],
    "domestic partner": ["domestic partner", "spouse"],
    "self": ["self"],
}


def _matches_word_groups(actual: Any, required: Any) -> bool:
    actual_n = _norm(actual)
    if not actual_n:
        return False
    candidates = [actual_n]
    for alias in RELATIONSHIP_ALIASES.get(actual_n, []):
        an = _norm(alias)
        if an and an not in candidates:
            candidates.append(an)

    for cand in candidates:
        for group in _word_groups(required):
            tokens = [t for t in group if t]
            if not tokens:
                continue
            joined = " ".join(tokens)
            if cand == joined or cand.replace(" ", "") == joined.replace(" ", ""):
                return True
            if all(tok in cand for tok in tokens):
                return True
            # GT group "son or daughter" vs claim "child"
            if joined in RELATIONSHIP_ALIASES and cand in {
                _norm(a) for a in RELATIONSHIP_ALIASES[joined]
            }:
                return True
    return False


def _lookup_fact(facts: Dict[str, Any], field_name: Optional[str]) -> Tuple[Any, Optional[str]]:
    if not field_name:
        return None, None
    keys_to_try = [field_name, _norm_key(field_name), _norm(field_name)]
    aliases = FIELD_ALIASES.get(_norm(field_name), [])
    keys_to_try.extend(aliases)

    for key in keys_to_try:
        if key in facts:
            return facts[key], key
        nk = _norm_key(key)
        for fk, fv in facts.items():
            if _norm_key(fk) == nk:
                return fv, fk
    return None, None


@dataclass
class EvalNote:
    level: str  # pass | fail | skip | info
    message: str


@dataclass
class PartResult:
    name: str
    passed: Optional[bool]
    notes: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackResult:
    jurisdiction: str
    passed: bool
    parts: Dict[str, PartResult]
    notes: List[str] = field(default_factory=list)


def _has_child_nodes(required: Any) -> bool:
    return (
        isinstance(required, list)
        and bool(required)
        and all(isinstance(x, dict) for x in required)
    )


def _condition_label(node: Dict[str, Any]) -> str:
    """Human-readable label for a condition node: prefer the plain-language
    'field' phrase (leave-reason packs write these as prose); fall back to a
    humanized condition_name. Plain field slugs (e.g. EE conditions like
    'employment_duration_months') are de-slugged into readable text."""
    field_val = node.get("field")
    if isinstance(field_val, str) and field_val.strip():
        text = field_val.strip()
        if "_" in text and " " not in text:
            text = text.replace("_", " ")
        return text
    name = node.get("condition_name") or "condition"
    return str(name).replace("_", " ")


def _record(checklist: Optional[List[Dict[str, Any]]], **kwargs: Any) -> None:
    """Append a citation-annotated checklist entry, if the caller is collecting one."""
    if checklist is None:
        return
    checklist.append(kwargs)


def evaluate_node(
    node: Dict[str, Any],
    facts: Dict[str, Any],
    *,
    path: str = "",
    checklist: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[bool], List[str]]:
    """Recursively evaluate a condition node. None = pending/skip.

    Nested rule: a parent with child nodes passes only when its operator is
    satisfied — ALL_OF (default) requires every child; ANY_OF requires one.

    When a `checklist` list is passed, every leaf and group node encountered
    is recorded with its plain-language label, citation, and pass/fail result
    (for citation-referenced UI display), without altering the pass/fail
    logic or the existing (ok, notes) return contract.
    """
    if not isinstance(node, dict):
        return False, [f"{path}: invalid node"]

    name = node.get("condition_name") or node.get("field") or "node"
    here = f"{path}/{name}" if path else str(name)
    operator = (node.get("operator") or "").strip().upper()
    if operator == "APPLIES":
        operator = "=="
    required = node.get("required_value")
    field_name = node.get("field")
    label = _condition_label(node)
    citation = node.get("citation")
    unit = node.get("unit")
    source_text = node.get("source_text")
    plain_notes = node.get("notes") if isinstance(node.get("notes"), str) else None

    # Child-object lists without an operator default to ALL_OF (all subs must pass).
    if _has_child_nodes(required) and operator not in {"ALL_OF", "ANY_OF"}:
        operator = "ALL_OF"

    # Leaf shortcut only: do not bypass ALL_OF/ANY_OF child evaluation.
    if node.get("condition_name") and not _has_child_nodes(required):
        shortcut, _ = _lookup_fact(facts, node["condition_name"])
        if isinstance(shortcut, bool) and operator in {"", "==", "IN"}:
            note = f"{here}: condition_name fact={shortcut}"
            _record(
                checklist, path=here, name=name, label=label, citation=citation, unit=unit,
                source_text=source_text, plain_notes=plain_notes, operator=operator or "==",
                required_value=required, actual_value=shortcut, passed=shortcut, is_group=False,
            )
            return shortcut, [note]

    if operator in {"ALL_OF", "ANY_OF"}:
        children = required if isinstance(required, list) else []
        child_results: List[Optional[bool]] = []
        notes: List[str] = []
        for i, child in enumerate(children):
            if not isinstance(child, dict):
                child_results.append(False)
                notes.append(f"{here}[{i}]: non-object child")
                continue
            ok, child_notes = evaluate_node(child, facts, path=here, checklist=checklist)
            child_results.append(ok)
            notes.extend(child_notes)
        known = [r for r in child_results if r is not None]
        if not known:
            group_ok = None
        elif operator == "ALL_OF":
            if any(r is False for r in child_results):
                group_ok = False
            elif any(r is None for r in child_results):
                group_ok = None
            else:
                group_ok = True
        else:
            # ANY_OF — one successful sub-condition is enough
            if any(r is True for r in child_results):
                group_ok = True
            elif all(r is False for r in child_results):
                group_ok = False
            else:
                group_ok = None
        _record(
            checklist, path=here, name=name, label=label, citation=citation, unit=unit,
            source_text=source_text, plain_notes=plain_notes, operator=operator,
            required_value=None, actual_value=None, passed=group_ok, is_group=True,
        )
        if not known:
            return None, notes + [f"{here}: no evaluable children"]
        return group_ok, notes

    actual, used_key = _lookup_fact(facts, field_name)
    if actual is None and name:
        actual, used_key = _lookup_fact(facts, name)

    def _finish(ok_val: Optional[bool], extra_note: str) -> Tuple[Optional[bool], List[str]]:
        _record(
            checklist, path=here, name=name, label=label, citation=citation, unit=unit,
            source_text=source_text, plain_notes=plain_notes, operator=op_sym,
            required_value=required, actual_value=actual, passed=ok_val, is_group=False,
        )
        return ok_val, [extra_note]

    # Employer-discretion cert/notice phrases are not employee denial rules.
    if isinstance(required, str) and any(m in _norm(required) for m in MAY_REQUIRE_MARKERS):
        _record(
            checklist, path=here, name=name, label=label, citation=citation, unit=unit,
            source_text=source_text, plain_notes=plain_notes, operator=operator or "==",
            required_value=required, actual_value=actual, passed=True, is_group=False,
            skipped_reason="employer discretion — not blocking",
        )
        return True, [f"{here}: employer-discretion rule — not blocking (treated as pass)"]

    if actual is None:
        _record(
            checklist, path=here, name=name, label=label, citation=citation, unit=unit,
            source_text=source_text, plain_notes=plain_notes, operator=operator or "==",
            required_value=required, actual_value=None, passed=None, is_group=False,
            skipped_reason=f"missing fact '{field_name}'",
        )
        return None, [f"{here}: missing fact for field '{field_name}'"]

    op_sym = (node.get("operator") or "==").strip()
    if op_sym.upper() in {"ALL_OF", "ANY_OF", "IN"}:
        op_sym = op_sym.upper()
    else:
        op_sym = op_sym if op_sym in OPERATORS else "=="

    if op_sym == "IN":
        ok = _matches_word_groups(actual, required)
        # Also allow exact membership for simple lists / enums
        if not ok and isinstance(required, list) and not any(isinstance(x, list) for x in required):
            actual_n = _norm(actual)
            allowed = {_norm(x) for x in required}
            for alias in RELATIONSHIP_ALIASES.get(actual_n, [actual_n]):
                if _norm(alias) in allowed:
                    ok = True
                    break
            if not ok:
                ok = actual_n in allowed
        return _finish(ok, f"{here}: {_norm(used_key)}={actual!r} IN {required!r} → {ok}")

    if isinstance(required, list) and required and isinstance(required[0], list):
        ok = _matches_word_groups(actual, required)
        return _finish(ok, f"{here}: {_norm(used_key)}={actual!r} matches groups → {ok}")

    if isinstance(required, list) and required and all(isinstance(x, str) for x in required):
        # Word-group disguised as flat token list for ==
        if op_sym == "==":
            ok = _matches_word_groups(actual, [required]) or _norm(actual) in {_norm(x) for x in required}
            return _finish(ok, f"{here}: {_norm(used_key)}={actual!r} == tokens {required!r} → {ok}")

    left = _as_number(actual)
    right = _as_number(required)
    if left is not None and right is not None and op_sym in OPERATORS:
        ok = bool(OPERATORS[op_sym](left, right))
        return _finish(ok, f"{here}: {used_key}={actual} {op_sym} {required} → {ok}")

    if op_sym in OPERATORS:
        if op_sym == "==":
            ok = _norm(actual) == _norm(required)
            # leave_reason alias expansion (health_condition ↔ serious_health_condition)
            if not ok and _norm(field_name) == "leave reason":
                aliases = facts.get("_leave_reason_aliases") or []
                ok = _norm(required) in {_norm(a) for a in aliases} or _matches_word_groups(
                    required, [[a] for a in aliases]
                )
        elif op_sym == "!=":
            ok = _norm(actual) != _norm(required)
        else:
            _record(
                checklist, path=here, name=name, label=label, citation=citation, unit=unit,
                source_text=source_text, plain_notes=plain_notes, operator=op_sym,
                required_value=required, actual_value=actual, passed=None, is_group=False,
                skipped_reason="cannot compare non-numeric values",
            )
            return None, [f"{here}: cannot compare non-numeric {actual!r} {op_sym} {required!r}"]
        return _finish(ok, f"{here}: {used_key}={actual!r} {op_sym} {required!r} → {ok}")

    _record(
        checklist, path=here, name=name, label=label, citation=citation, unit=unit,
        source_text=source_text, plain_notes=plain_notes, operator=node.get("operator"),
        required_value=required, actual_value=actual, passed=None, is_group=False,
        skipped_reason="unsupported operator",
    )
    return None, [f"{here}: unsupported operator {node.get('operator')!r}"]


def _should_skip_ee_condition(cond: Dict[str, Any], facts: Dict[str, Any]) -> Optional[str]:
    field_name = cond.get("field") or ""
    notes = _norm(cond.get("notes"))
    fn = _norm(field_name)
    is_air = bool(facts.get("is_air_crew"))
    pay = _norm(facts.get("employee_pay_type"))

    if "airline" in fn or ("airline" in notes or "flight crew" in notes):
        return None if is_air else "skipped airline-only EE rule"
    if fn == "hours_worked_last_12_months" and is_air:
        return "skipped standard hours rule for air crew"
    if "userra" in notes or "rehire agreement" in notes or "break in service" in fn:
        if _lookup_fact(facts, field_name)[0] is None:
            return "skipped optional break-in-service / USERRA rule"
    if "salaried" in notes and pay == "hourly":
        return "skipped salaried-only EE rule"
    if "hourly" in notes and pay == "salaried":
        return "skipped hourly-only EE rule"
    if cond.get("operator") == "applies" and _lookup_fact(facts, field_name)[0] is None:
        return "skipped applies-operator rule without fact"
    return None


def evaluate_ee(conditions: Sequence[Dict[str, Any]], facts: Dict[str, Any]) -> PartResult:
    notes: List[str] = []
    detail: List[Dict[str, Any]] = []
    checklist: List[Dict[str, Any]] = []
    evaluated = 0
    failed = 0
    pending = 0

    for cond in conditions:
        skip = _should_skip_ee_condition(cond, facts)
        field_name = cond.get("field")
        if skip:
            notes.append(f"EE skip {field_name}: {skip}")
            checklist.append({
                "path": f"ee/{cond.get('condition_name') or field_name}",
                "name": cond.get("condition_name") or field_name,
                "label": _condition_label(cond),
                "citation": cond.get("citation"),
                "unit": cond.get("unit"),
                "source_text": cond.get("source_text"),
                "plain_notes": cond.get("notes") if isinstance(cond.get("notes"), str) else None,
                "operator": cond.get("operator"),
                "required_value": cond.get("required_value"),
                "actual_value": None,
                "passed": None,
                "is_group": False,
                "skipped_reason": skip,
            })
            continue
        ok, node_notes = evaluate_node(cond, facts, path="ee", checklist=checklist)
        notes.extend(node_notes)
        detail.append({"field": field_name, "passed": ok})
        if ok is None:
            pending += 1
        else:
            evaluated += 1
            if not ok:
                failed += 1

    if evaluated == 0 and pending == 0:
        return PartResult("ee", False, notes + ["EE: no conditions evaluated"], {"items": detail, "checklist": checklist})
    if failed:
        return PartResult("ee", False, notes, {"items": detail, "checklist": checklist})
    if pending:
        return PartResult("ee", None, notes + ["EE: pending missing facts"], {"items": detail, "checklist": checklist})
    return PartResult("ee", True, notes, {"items": detail, "checklist": checklist})


def evaluate_relationship(included: Optional[Dict[str, Any]], facts: Dict[str, Any]) -> PartResult:
    if not included:
        return PartResult("relationship", True, ["No included_relationships — treated as pass"])
    rel, _ = _lookup_fact(facts, "relationship")
    if rel is None:
        return PartResult("relationship", None, ["relationship fact missing"])
    checklist: List[Dict[str, Any]] = []
    ok, notes = evaluate_node(
        {
            "condition_name": "included_relationships",
            "field": "relationship",
            "operator": included.get("operator", "IN"),
            "required_value": included.get("required_value"),
            "citation": included.get("citation"),
            "notes": included.get("notes"),
        },
        facts,
        path="relationship",
        checklist=checklist,
    )
    return PartResult("relationship", ok, notes, {"checklist": checklist})


def _eval_top_level_any_of(
    conditions: Sequence[Dict[str, Any]],
    facts: Dict[str, Any],
    *,
    path: str,
    checklist: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[bool], List[str], List[Dict[str, Any]], List[str]]:
    """Top-level leave/notice/cert: pass if ANY main condition passes.

    A main condition with nested children passes only when all of its
    sub-conditions pass (ALL_OF default) or per its own ANY_OF operator.
    An explicit boolean fact keyed by condition_name may assert that main
    pathway when present (test-author shorthand that all subs are met).

    If `checklist` is provided, every main condition (and its nested
    sub-conditions) is recorded with citation info for citation-referenced
    UI display.
    """
    if checklist is None:
        checklist = []
    notes: List[str] = []
    detail: List[Dict[str, Any]] = []
    hits: List[str] = []

    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        name = cond.get("condition_name") or cond.get("field") or "condition"
        shortcut, _ = _lookup_fact(facts, name)
        if isinstance(shortcut, bool):
            # Author shorthand: True means this main pathway is satisfied.
            ok, node_notes = shortcut, [f"{path}/{name}: main condition_name fact={shortcut}"]
            checklist.append({
                "path": f"{path}/{name}",
                "name": name,
                "label": _condition_label(cond),
                "citation": cond.get("citation"),
                "unit": cond.get("unit"),
                "source_text": cond.get("source_text"),
                "plain_notes": cond.get("notes") if isinstance(cond.get("notes"), str) else None,
                "operator": cond.get("operator"),
                "required_value": None,
                "actual_value": shortcut,
                "passed": ok,
                "is_group": _has_child_nodes(cond.get("required_value")),
            })
        else:
            ok, node_notes = evaluate_node(cond, facts, path=path, checklist=checklist)
        notes.extend(node_notes)
        detail.append({"condition_name": name, "passed": ok})
        if ok is True:
            hits.append(str(name))

    if not detail:
        return False, notes + [f"{path}: no main conditions"], detail, hits
    if any(d["passed"] is True for d in detail):
        return True, notes, detail, hits
    if all(d["passed"] is False for d in detail):
        return False, notes, detail, hits
    return None, notes + [f"{path}: pending — no main condition fully resolved"], detail, hits


def evaluate_leave_reason(pack: Dict[str, Any], facts: Dict[str, Any]) -> Tuple[PartResult, PartResult]:
    rel_part = evaluate_relationship(pack.get("included_relationships"), facts)
    conditions = pack.get("conditions") or []
    checklist: List[Dict[str, Any]] = []
    leave_ok, notes, pathway_detail, pathway_hits = _eval_top_level_any_of(
        conditions, facts, path="leave", checklist=checklist
    )

    if rel_part.passed is False:
        leave_ok = False
        notes.append("leave_reason failed because relationship not covered")
    elif leave_ok is True and rel_part.passed is None:
        leave_ok = None

    leave_part = PartResult(
        "leave_reason",
        leave_ok if rel_part.passed is not False else False,
        notes,
        {
            "mode": "any_of_main_conditions",
            "pathways_passed": pathway_hits,
            "pathways": pathway_detail,
            "checklist": checklist,
        },
    )
    return rel_part, leave_part


def _normalize_condition_list(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("conditions"), list):
            return [x for x in data["conditions"] if isinstance(x, dict)]
        if "operator" in data or "condition_name" in data or "field" in data:
            return [data]
        if isinstance(data.get("employee_eligibility_conditions"), list):
            return [x for x in data["employee_eligibility_conditions"] if isinstance(x, dict)]
    return []


def evaluate_notice(pack: Any, facts: Dict[str, Any]) -> PartResult:
    conditions = _normalize_condition_list(pack)
    if not conditions:
        return PartResult("notice", True, ["notice: empty pack — treated as pass"])

    # Single wrapped root (e.g. federal notice_gt): evaluate that tree as-is.
    if len(conditions) == 1 and (conditions[0].get("operator") or "").upper() in {"ANY_OF", "ALL_OF"}:
        checklist: List[Dict[str, Any]] = []
        ok, notes = evaluate_node(conditions[0], facts, path="notice", checklist=checklist)
        return PartResult("notice", ok, notes, {"mode": "single_root", "checklist": checklist})

    checklist = []
    ok, notes, detail, hits = _eval_top_level_any_of(conditions, facts, path="notice", checklist=checklist)
    if ok is True:
        return PartResult(
            "notice", True, notes,
            {"mode": "any_of_main_conditions", "passed_mains": hits, "items": detail, "checklist": checklist},
        )
    if ok is False:
        return PartResult(
            "notice", False, notes,
            {"mode": "any_of_main_conditions", "items": detail, "checklist": checklist},
        )
    # Pending fallback: foreseeable advance notice facts
    adv, _ = _lookup_fact(facts, "advance_notice")
    fore, _ = _lookup_fact(facts, "foreseeable_leave")
    if fore and _as_number(adv) is not None and _as_number(adv) >= 30:
        return PartResult(
            "notice",
            True,
            notes + ["notice: satisfied via foreseeable advance_notice >= 30 fallback"],
            {"mode": "fallback_foreseeable", "items": detail, "checklist": checklist},
        )
    return PartResult(
        "notice", None, notes, {"mode": "any_of_main_conditions", "items": detail, "checklist": checklist}
    )


def evaluate_cert(pack: Any, facts: Dict[str, Any]) -> PartResult:
    """Cert passes if ANY top-level main condition passes (same as leave/notice)."""
    conditions = _normalize_condition_list(pack)
    if not conditions:
        return PartResult("cert", True, ["cert: empty pack — treated as pass"])

    leave_reason = _norm(facts.get("leave_reason")).replace(" ", "_")
    notes: List[str] = []
    applicable: List[Dict[str, Any]] = []

    for cond in conditions:
        name = _norm(cond.get("condition_name"))
        req = cond.get("required_value")
        op_u = (cond.get("operator") or "").upper()

        # Skip leave_reason-gated mains that do not apply to this claim.
        gated = False
        gate_match = True
        if op_u in {"ALL_OF", "ANY_OF"} and isinstance(req, list):
            for child in req:
                if not isinstance(child, dict):
                    continue
                if _norm(child.get("field")) == "leave reason":
                    gated = True
                    g_ok, _ = evaluate_node(child, facts, path="cert_gate")
                    gate_match = bool(g_ok)
                    break

        if not gated:
            if "exigency" in name and "military_exigency" not in leave_reason:
                continue
            if "caregiver" in name and "military_caregiver" not in leave_reason:
                continue
            if leave_reason in {"birth_or_pregnancy", "adoption"} and "health" in name:
                continue

        if gated and not gate_match:
            notes.append(f"cert/{cond.get('condition_name')}: leave_reason gate not matched — skipped")
            continue

        applicable.append(cond)

    if not applicable:
        return PartResult(
            "cert",
            True,
            notes + ["cert: no applicable main conditions for this leave_reason — pass"],
        )

    checklist: List[Dict[str, Any]] = []
    ok, more_notes, detail, hits = _eval_top_level_any_of(applicable, facts, path="cert", checklist=checklist)
    notes.extend(more_notes)
    return PartResult(
        "cert",
        ok,
        notes,
        {"mode": "any_of_main_conditions", "passed_mains": hits, "items": detail, "checklist": checklist},
    )


def evaluate_track(
    *,
    jurisdiction: str,
    packs: Dict[str, Any],
    facts: Dict[str, Any],
) -> TrackResult:
    ee = evaluate_ee(_normalize_condition_list(packs.get("ee")), facts)
    notice = evaluate_notice(packs.get("notice"), facts)
    cert = evaluate_cert(packs.get("cert"), facts)
    rel, leave = evaluate_leave_reason(packs.get("leave_reason") or {}, facts)

    parts = {
        "ee": ee,
        "notice": notice,
        "cert": cert,
        "relationship": rel,
        "leave_reason": leave,
    }
    notes: List[str] = []
    for part in parts.values():
        notes.extend(part.notes)

    statuses = [ee.passed, notice.passed, cert.passed, leave.passed]
    if any(s is False for s in statuses):
        passed = False
    elif any(s is None for s in statuses):
        passed = False
        notes.append(f"{jurisdiction} track: pending treated as not approved")
    else:
        passed = True

    failed_parts = [name for name, part in parts.items() if part.passed is False]
    if failed_parts:
        notes.append(f"{jurisdiction} track failed parts: {', '.join(failed_parts)}")
    elif passed:
        notes.append(f"{jurisdiction} track passed")

    return TrackResult(jurisdiction=jurisdiction, passed=passed, parts=parts, notes=notes)


# ---------------------------------------------------------------------------
# Condition pack loading
# ---------------------------------------------------------------------------

LEAVE_REASONS = {
    "health_condition",
    "birth_or_pregnancy",
    "adoption",
    "military_exigency",
    "military_caregiver",
}


def _gt_paths(jurisdiction: str, leave_reason: str) -> Dict[str, Path]:
    j = jurisdiction
    base = DATA_DIR / j / "gt"
    if j == "federal":
        return {
            "ee": base / "ee_gt.json",
            "notice": base / "notice_gt.json",
            "cert": base / "cert_gt.json",
            "leave_reason": base / f"{leave_reason}_gt.json",
        }
    if j == "CA":
        return {
            "ee": base / "ca_ee_gt.json",
            "notice": base / "ca_notice_gt.json",
            "cert": base / "ca_cert_gt.json",
            "leave_reason": base / f"ca_{leave_reason}_gt.json",
        }
    if j == "TN":
        return {
            "ee": base / "tn_ee_gt.json",
            "notice": base / "notice_gt.json",
            "cert": base / "cert_gt.json",
            "leave_reason": base / f"tn_{leave_reason}_gt.json",
        }
    if j == "GA":
        return {
            "ee": base / "ee_gt.json",
            "notice": base / "notice_gt.json",
            "cert": base / "cert_gt.json",
            "leave_reason": base / f"{leave_reason}_gt.json",
        }
    raise ValueError(f"Unknown jurisdiction: {jurisdiction}")


def _extracted_paths(jurisdiction: str, leave_reason: str) -> Dict[str, Path]:
    base = OUTPUT_DIR / jurisdiction
    return {
        "ee": base / "ee_extracted.json",
        "notice": base / "notice.json",
        "cert": base / "certification.json",
        "leave_reason": base / f"{leave_reason}_lr_extracted.json",
    }


def load_condition_packs(
    jurisdiction: str,
    leave_reason: str,
    source: str,
) -> Dict[str, Any]:
    paths = _gt_paths(jurisdiction, leave_reason) if source == "gt" else _extracted_paths(jurisdiction, leave_reason)
    packs: Dict[str, Any] = {}
    for key, path in paths.items():
        if not path.exists():
            packs[key] = {} if key == "leave_reason" else []
            packs[f"_{key}_missing"] = str(path)
            continue
        packs[key] = load_json(path)
    return packs


def tracks_for_case(case: Dict[str, Any]) -> List[str]:
    juris = case.get("jurisdiction")
    facts = case.get("facts") or {}
    tracks = ["federal"]
    if juris in {"CA", "TN"}:
        tracks.append(juris)
    elif juris == "GA":
        if bool(facts.get("is_government_worker")):
            tracks.append("GA")
    elif juris == "federal":
        pass
    else:
        tracks.append(str(juris))
    return tracks


def _normalize_claim_facts(facts: Dict[str, Any], leave_reason: Optional[str]) -> Dict[str, Any]:
    """Expand aliases so GT gates (e.g. serious_health_condition) match claim enums."""
    out = dict(facts)
    lr = leave_reason or out.get("leave_reason")
    if lr:
        out.setdefault("leave_reason", lr)
        aliases = {
            "health_condition": [
                "health_condition",
                "health condition",
                "serious_health_condition",
                "serious health condition",
            ],
            "birth_or_pregnancy": [
                "birth_or_pregnancy",
                "birth or pregnancy",
                "birth",
                "pregnancy",
            ],
            "adoption": ["adoption", "foster care", "foster_care"],
            "military_exigency": [
                "military_exigency",
                "military exigency",
                "qualifying_exigency",
                "qualifying exigency",
            ],
            "military_caregiver": [
                "military_caregiver",
                "military caregiver",
            ],
        }
        key = _norm_key(lr)
        # Store a list so IN / == word-group checks can match alternate labels.
        out["_leave_reason_aliases"] = aliases.get(key, [lr])
        # Prefer canonical string for == against spaced GT values
        spaced = key.replace("_", " ")
        if _norm(out.get("leave_reason")) in {_norm(x) for x in aliases.get(key, [])}:
            # Keep original; comparison helper below uses aliases
            pass
        out.setdefault("leave_reason_spaced", spaced)
    return out


def run_case(case: Dict[str, Any], source: str) -> Dict[str, Any]:
    leave_reason = case.get("leave_reason") or (case.get("facts") or {}).get("leave_reason")
    facts = _normalize_claim_facts(dict(case.get("facts") or {}), leave_reason)
    juris = case.get("jurisdiction")
    track_names = tracks_for_case(case)

    def _run_one(track_j: str) -> TrackResult:
        packs = load_condition_packs(track_j, leave_reason, source)
        return evaluate_track(jurisdiction=track_j, packs=packs, facts=facts)

    track_results: Dict[str, TrackResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(track_names))) as pool:
        futures = {pool.submit(_run_one, t): t for t in track_names}
        for fut, t in futures.items():
            track_results[t] = fut.result()

    any_approved = any(tr.passed for tr in track_results.values())
    decision = DECISION_APPROVED if any_approved else DECISION_NEED_HUMAN_REVIEW

    notes: List[str] = [
        f"source={source}",
        f"tracks={track_names}",
        f"is_government_worker={facts.get('is_government_worker')}",
    ]
    for t in track_names:
        tr = track_results[t]
        notes.append(f"[{t}] passed={tr.passed}")
        for part_name, part in tr.parts.items():
            notes.append(f"[{t}.{part_name}] passed={part.passed}")
            for n in part.notes:
                if "→ False" in n or "failed" in n.lower() or "missing" in n.lower():
                    notes.append(f"  · {n}")

    # Compact failure summary
    for t, tr in track_results.items():
        if tr.passed:
            notes.append(f"Approved via {t} track")
            break
    else:
        for t, tr in track_results.items():
            failed = [k for k, p in tr.parts.items() if p.passed is False]
            pending = [k for k, p in tr.parts.items() if p.passed is None]
            if failed:
                notes.append(f"{t} failed: {', '.join(failed)}")
            if pending:
                notes.append(f"{t} pending: {', '.join(pending)}")

    return {
        "id": case.get("id"),
        "jurisdiction": juris,
        "leave_reason": leave_reason,
        "case_type": case.get("case_type"),
        "label": case.get("label"),
        "source": source,
        "tracks_evaluated": track_names,
        "decision": decision,
        "expected_decision": (case.get("expected") or {}).get("decision"),
        "decision_matches_expected": decision == (case.get("expected") or {}).get("decision"),
        "tracks": {
            t: {
                "passed": tr.passed,
                "parts": {
                    name: {
                        "passed": part.passed,
                        "notes": part.notes,
                        "detail": part.detail,
                    }
                    for name, part in tr.parts.items()
                },
            }
            for t, tr in track_results.items()
        },
        "notes": notes,
    }


def load_all_cases() -> List[Dict[str, Any]]:
    index = load_json(TEST_CASES_DIR / "index.json")
    cases: List[Dict[str, Any]] = []
    for pair in index.get("pairs", []):
        path = TEST_CASES_DIR / pair["path"]
        data = load_json(path)
        if isinstance(data, list):
            cases.extend(data)
    return cases


def run_all_cases(source: str) -> Dict[str, Any]:
    cases = load_all_cases()
    results = [run_case(case, source) for case in cases]
    approved = sum(1 for r in results if r["decision"] == DECISION_APPROVED)
    matched = sum(1 for r in results if r.get("decision_matches_expected"))
    summary = {
        "source": source,
        "total_cases": len(results),
        "approved": approved,
        "need_human_review": len(results) - approved,
        "decision_matches_expected": matched,
        "results": results,
    }
    out_dir = OUTPUT_DIR / "test_results" / source
    out_path = save_json(summary, out_dir / "results.json")
    summary["output_path"] = str(out_path)
    return summary
