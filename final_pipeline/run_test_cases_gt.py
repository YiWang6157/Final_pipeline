#!/usr/bin/env python3
"""Run synthetic test cases against GT condition packs (final_pipeline/data/*/gt)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from function.test_case_evaluator import run_all_cases  # noqa: E402


def main() -> int:
    summary = run_all_cases("gt")
    print(
        json.dumps(
            {
                "source": summary["source"],
                "total_cases": summary["total_cases"],
                "approved": summary["approved"],
                "need_human_review": summary["need_human_review"],
                "decision_matches_expected": summary["decision_matches_expected"],
                "output_path": summary["output_path"],
            },
            indent=2,
        )
    )
    for r in summary["results"]:
        mark = "OK" if r.get("decision_matches_expected") else "DIFF"
        print(
            f"[{mark}] {r['id']}: {r['decision']} "
            f"(expected {r.get('expected_decision')}) tracks={r['tracks_evaluated']}"
        )
        fails = [n for n in r["notes"] if "failed" in n.lower() or "pending:" in n.lower()]
        for n in fails[:8]:
            print(f"    {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
