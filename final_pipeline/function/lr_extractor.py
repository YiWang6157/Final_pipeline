"""
Leave Reason (LR) eligibility extractor for final_pipeline (predefined schema).

Uses Final/lr_extraction/API1 (fill-in) and optionally Final/lr_extraction/API2
(critique). Requires regulation chunks + a predefined schema JSON.

Data layout examples:
  chunks:
    final_pipeline/data/federal/fmla_chunk_schema_filled.json
    final_pipeline/data/GA/chunk_schema_filled.json
    final_pipeline/data/TN/tn_extracted_sections_chunk_schema_filled.json
    final_pipeline/data/CA/{cfra,pdl}_chunk_schema_filled.json
  schemas:
    final_pipeline/data/federal/schema/adoption.json
    final_pipeline/data/GA/schema/health_condition.json
    final_pipeline/data/TN/schema/tn_adoption.json
    final_pipeline/data/CA/schema/birth_or_pregnancy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SRC_DIR,
    PIPELINE_DIR,
    ChunkPaths,
    as_source_list,
    chunk_matches_reason_and_function,
    invoke_llm,
    load_chunks,
    load_json,
    normalize_empty_to_null,
    normalize_to_list,
    parse_llm_json,
    reason_slug,
    resolve_path,
    save_json,
    setup_llm,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Fill-in fields (edit these, or override via function args / CLI)
# ---------------------------------------------------------------------------

CHUNK_FILE_PATHS: ChunkPaths = PIPELINE_DIR / "data" / "federal" / "fmla_chunk_schema_filled.json"
SCHEMA_FILE_PATH: Path = PIPELINE_DIR / "data" / "federal" / "schema" / "adoption.json"

# Must match chunk "reason" labels (spaces), not necessarily the schema filename.
LEAVE_REASON: str = "adoption"

OUTPUT_DIR: Path = DEFAULT_OUTPUT_DIR
OUTPUT_FILENAME: Optional[str] = None  # default: {leave_reason_slug}_lr_extracted.json

YML_CONFIG_PATH: Path = DEFAULT_CONFIG
SRC_DIR: Path = DEFAULT_SRC_DIR

FUNCTION_NAME: str = "Final"
API1_SUB_FUNCTION: str = "lr_extraction/API1"
API2_SUB_FUNCTION: str = "lr_extraction/API2"
MODEL_KEY: str = "ari_model"

TARGET_LR_FUNCTION: str = "leave eligibility"
RUN_CRITIQUE: bool = True  # always run API2 critique after API1


def prepare_schema_template(raw_schema: Dict[str, Any], leave_reason: str) -> Dict[str, Any]:
    schema = normalize_empty_to_null(raw_schema)
    schema["leave_reason"] = leave_reason
    rel = schema.setdefault("included_relationships", {})
    for key in ("operator", "required_value", "citation", "notes"):
        rel.setdefault(key, None)
    for cond in schema.get("conditions", []) or []:
        if isinstance(cond, dict):
            for key in ("notes", "source_text"):
                cond.setdefault(key, None)
    return schema


def build_section_records(filtered_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Jurisdiction-agnostic section builder.

    One record per distinct regulation number; falls back to chunk index when
    regulation number is missing. Merges text when the same number repeats.
    """
    section_map: Dict[str, Dict[str, Any]] = {}
    for i, chunk in enumerate(filtered_chunks):
        regs = chunk.get("regulation number")
        if regs is None or regs == [] or regs == "":
            reg_list = [f"chunk_{i}"]
        elif isinstance(regs, list):
            reg_list = [str(r) for r in regs] or [f"chunk_{i}"]
        else:
            reg_list = [str(regs)]

        text = str(chunk.get("raw text") or "").strip()
        for reg in reg_list:
            if reg not in section_map:
                section_map[reg] = {
                    "section_number": reg,
                    "section_title": reg,
                    "section_text": text,
                    "source_chunk_indices": [i],
                }
            else:
                existing = section_map[reg]["section_text"]
                if text and text not in existing:
                    section_map[reg]["section_text"] = (
                        existing.rstrip() + "\n\n" + text if existing else text
                    )
                section_map[reg]["source_chunk_indices"].append(i)

    return list(section_map.values())


def build_predefined_rendered_input(
    leave_reason: str, sections: List[Dict[str, Any]]
) -> str:
    # Drop internal indices from what the model sees.
    api_sections = [
        {
            "section_number": s["section_number"],
            "section_title": s["section_title"],
            "section_text": s["section_text"],
        }
        for s in sections
    ]
    sections_json_str = json.dumps(api_sections, indent=2, ensure_ascii=False)
    return "\n".join(
        [
            "[LEAVE REASON]",
            leave_reason,
            "",
            "[NO RELATIONSHIP PROVIDED]",
            "Infer included/covered relationships only from the regulation sections below.",
            "",
            "[RETRIEVED REGULATION SECTIONS]",
            "These sections are filtered for this leave reason and leave-eligibility task only.",
            "",
            sections_json_str,
            "",
            "[TASK]",
            "Fill in the predefined condition template provided below.",
            "Keep every predefined condition_name.",
            "Preserve nested field labels from the template.",
            "Use null for fields not supported by the section text.",
            "Keep required_value concise: numbers, booleans, or short keyword phrases only.",
            "You may append new top-level conditions if the sections support additional pathways.",
            "Return only valid JSON with keys: leave_reason, included_relationships, conditions.",
        ]
    )


def build_critique_rendered_input(
    leave_reason: str,
    sections: List[Dict[str, Any]],
    api1_output: Dict[str, Any],
) -> str:
    api_sections = [
        {
            "section_number": s["section_number"],
            "section_title": s["section_title"],
            "section_text": s["section_text"],
        }
        for s in sections
    ]
    payload = {
        "leave_reason": leave_reason,
        "API1_output": api1_output,
        "regulation_chunks": api_sections,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def extract_lr_conditions(
    chunk_file_paths: ChunkPaths = CHUNK_FILE_PATHS,
    schema_file_path: str | Path = SCHEMA_FILE_PATH,
    leave_reason: str = LEAVE_REASON,
    output_dir: Optional[str | Path] = OUTPUT_DIR,
    output_filename: Optional[str] = OUTPUT_FILENAME,
    yml_config_path: str | Path = YML_CONFIG_PATH,
    src_dir: str | Path = SRC_DIR,
    function_name: str = FUNCTION_NAME,
    api1_sub_function: str = API1_SUB_FUNCTION,
    api2_sub_function: str = API2_SUB_FUNCTION,
    model_key: str = MODEL_KEY,
    target_lr_function: str = TARGET_LR_FUNCTION,
    run_critique: bool = RUN_CRITIQUE,
) -> Dict[str, Any]:
    """
    Fill a predefined LR schema from leave-eligibility chunks and write JSON.

    Returns the final LR payload (from API2 revised_output when critique runs,
    otherwise API1 output).
    """
    all_chunks = load_chunks(chunk_file_paths)
    filtered = [
        c
        for c in all_chunks
        if chunk_matches_reason_and_function(c, leave_reason, target_lr_function)
    ]
    if not filtered:
        reasons_seen = sorted(
            {
                r
                for c in all_chunks
                for r in normalize_to_list(c.get("reason"))
            }
        )
        raise RuntimeError(
            f"No chunks matched reason={leave_reason!r} and "
            f"function={target_lr_function!r}. Reasons present: {reasons_seen}"
        )

    sections = build_section_records(filtered)
    if not sections:
        raise RuntimeError("Matched chunks but produced no section records.")

    schema_path = resolve_path(schema_file_path)
    predefined_schema = prepare_schema_template(load_json(schema_path), leave_reason)
    rendered_input = build_predefined_rendered_input(leave_reason, sections)
    predefined_schema_str = json.dumps(predefined_schema, indent=2, ensure_ascii=False)

    print(
        f"Leave reason={leave_reason!r}: {len(filtered)} chunk(s), "
        f"{len(sections)} section(s); schema={schema_path.name}"
    )

    LLM_Ops, provider_info_config, llm_provider, llm_deployment = setup_llm(
        yml_config_path=yml_config_path,
        src_dir=src_dir,
        model_key=model_key,
    )
    llm_kwargs = dict(
        LLM_Ops=LLM_Ops,
        provider_info_config=provider_info_config,
        llm_provider=llm_provider,
        llm_deployment=llm_deployment,
    )

    usage1, raw1 = invoke_llm(
        function_name=function_name,
        sub_function_name=api1_sub_function,
        input_data={
            "rendered_input": rendered_input,
            "predefined_schema": predefined_schema_str,
        },
        **llm_kwargs,
    )
    print(f"API1 token usage: {usage1}")
    api1_output = parse_llm_json(raw1)

    review = None
    usage2 = None
    raw2 = None
    final_output = api1_output

    if run_critique:
        critique_input = build_critique_rendered_input(
            leave_reason, sections, api1_output
        )
        usage2, raw2 = invoke_llm(
            function_name=function_name,
            sub_function_name=api2_sub_function,
            input_data={"rendered_input": critique_input},
            **llm_kwargs,
        )
        print(f"API2 token usage: {usage2}")
        critique = parse_llm_json(raw2)
        review = critique.get("review")
        revised = critique.get("revised_output")
        if isinstance(revised, dict):
            final_output = revised
        else:
            raise ValueError("API2 response missing revised_output object")

    out_dir = resolve_path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    fname = output_filename or f"{reason_slug(leave_reason)}_lr_extracted.json"
    out_path = out_dir / fname

    record: Dict[str, Any] = {
        "extracted_at": utc_now_iso(),
        "source": as_source_list(chunk_file_paths),
        "schema_source": str(schema_path),
        "function_name": function_name,
        "api1_sub_function": api1_sub_function,
        "api2_sub_function": api2_sub_function if run_critique else None,
        "model_key": model_key,
        "leave_reason": leave_reason,
        "matched_chunk_count": len(filtered),
        "section_count": len(sections),
        "included_relationships": final_output.get("included_relationships"),
        "conditions": final_output.get("conditions"),
        "api1_token_usage": usage1,
        "api1_raw_llm_response": (
            raw1 if isinstance(raw1, str) else json.dumps(raw1, ensure_ascii=False)
        ),
    }
    if run_critique:
        record["review"] = review
        record["api2_token_usage"] = usage2
        record["api2_raw_llm_response"] = (
            raw2 if isinstance(raw2, str) else json.dumps(raw2, ensure_ascii=False)
        )

    save_json(record, out_path)
    print(f"Saved → {out_path}")
    return {
        "leave_reason": leave_reason,
        "included_relationships": final_output.get("included_relationships"),
        "conditions": final_output.get("conditions"),
        **({"review": review} if review is not None else {}),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract leave-reason conditions (Final/lr_extraction)."
    )
    parser.add_argument("--chunks", nargs="+", default=None)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--leave-reason", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--config", default=str(YML_CONFIG_PATH))
    parser.add_argument("--src-dir", default=str(SRC_DIR))
    parser.add_argument("--function", default=FUNCTION_NAME)
    parser.add_argument("--api1-sub-function", default=API1_SUB_FUNCTION)
    parser.add_argument("--api2-sub-function", default=API2_SUB_FUNCTION)
    parser.add_argument("--model-key", default=MODEL_KEY)
    parser.add_argument(
        "--skip-critique",
        action="store_true",
        help="Skip API2 critique (critique runs by default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_lr_conditions(
        chunk_file_paths=args.chunks or CHUNK_FILE_PATHS,
        schema_file_path=args.schema or SCHEMA_FILE_PATH,
        leave_reason=args.leave_reason or LEAVE_REASON,
        output_dir=args.output_dir or OUTPUT_DIR,
        output_filename=args.output_filename or OUTPUT_FILENAME,
        yml_config_path=args.config,
        src_dir=args.src_dir,
        function_name=args.function,
        api1_sub_function=args.api1_sub_function,
        api2_sub_function=args.api2_sub_function,
        model_key=args.model_key,
        run_critique=(not args.skip_critique) and RUN_CRITIQUE,
    )
