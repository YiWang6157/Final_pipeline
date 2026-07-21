"""
Employee Eligibility (EE) extractor for final_pipeline.

Uses Final/ee_extraction. Fill in the paths below (or pass as args) and call
extract_ee_conditions().

Data layout examples:
  final_pipeline/data/federal/fmla_chunk_schema_filled.json
  final_pipeline/data/GA/chunk_schema_filled.json
  final_pipeline/data/TN/tn_extracted_sections_chunk_schema_filled.json
  final_pipeline/data/CA/cfra_chunk_schema_filled.json
  final_pipeline/data/CA/pdl_chunk_schema_filled.json   # pass both for CA
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
    chunk_matches_function,
    invoke_llm,
    load_chunks,
    parse_llm_json,
    resolve_path,
    save_json,
    setup_llm,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Fill-in fields (edit these, or override via function args / CLI)
# ---------------------------------------------------------------------------

# One path, or a list (e.g. CA CFRA + PDL).
CHUNK_FILE_PATHS: ChunkPaths = PIPELINE_DIR / "data" / "federal" / "fmla_chunk_schema_filled.json"

OUTPUT_DIR: Path = DEFAULT_OUTPUT_DIR
OUTPUT_FILENAME: str = "ee_extracted.json"

YML_CONFIG_PATH: Path = DEFAULT_CONFIG
SRC_DIR: Path = DEFAULT_SRC_DIR

FUNCTION_NAME: str = "Final"
SUB_FUNCTION_NAME: str = "ee_extraction"
MODEL_KEY: str = "ari_model"

# If True, only EE-tagged chunks are sent. If False, all chunks are sent
# (matches Georgia / federal_pipeline behavior).
FILTER_TO_EE_CHUNKS: bool = False


def extract_ee_conditions(
    chunk_file_paths: ChunkPaths = CHUNK_FILE_PATHS,
    output_dir: Optional[str | Path] = OUTPUT_DIR,
    output_filename: str = OUTPUT_FILENAME,
    yml_config_path: str | Path = YML_CONFIG_PATH,
    src_dir: str | Path = SRC_DIR,
    function_name: str = FUNCTION_NAME,
    sub_function_name: str = SUB_FUNCTION_NAME,
    model_key: str = MODEL_KEY,
    filter_to_ee_chunks: bool = FILTER_TO_EE_CHUNKS,
) -> Dict[str, Any]:
    """
    Call the EE LLM prompt and write extracted JSON.

    Returns the parsed extraction payload (employee_eligibility_conditions).
    """
    all_chunks = load_chunks(chunk_file_paths)
    ee_chunks = [
        c for c in all_chunks if chunk_matches_function(c, "employee eligibility")
    ]
    chunks_for_llm: List[Dict[str, Any]] = (
        ee_chunks if filter_to_ee_chunks else all_chunks
    )

    print(f"Loaded {len(all_chunks)} chunk(s); {len(ee_chunks)} tagged 'employee eligibility'.")
    print(f"Sending {len(chunks_for_llm)} chunk(s) to LLM ({sub_function_name}).")

    LLM_Ops, provider_info_config, llm_provider, llm_deployment = setup_llm(
        yml_config_path=yml_config_path,
        src_dir=src_dir,
        model_key=model_key,
    )
    usage, raw = invoke_llm(
        LLM_Ops=LLM_Ops,
        provider_info_config=provider_info_config,
        llm_provider=llm_provider,
        llm_deployment=llm_deployment,
        function_name=function_name,
        sub_function_name=sub_function_name,
        input_data={"chunks_json": json.dumps(chunks_for_llm, ensure_ascii=False)},
    )
    print(f"Token usage: {usage}")

    parsed = parse_llm_json(raw)
    conditions = parsed.get("employee_eligibility_conditions")
    if not isinstance(conditions, list):
        raise ValueError("Expected employee_eligibility_conditions to be a list")

    out_dir = resolve_path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_path = out_dir / output_filename
    record = {
        "extracted_at": utc_now_iso(),
        "source": as_source_list(chunk_file_paths),
        "function_name": function_name,
        "sub_function_name": sub_function_name,
        "model_key": model_key,
        "employee_eligibility_chunk_count": len(ee_chunks),
        "chunks_sent_count": len(chunks_for_llm),
        "employee_eligibility_conditions": conditions,
        "token_usage": usage,
        "raw_llm_response": raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False),
    }
    save_json(record, out_path)
    print(f"Saved → {out_path}")
    return {"employee_eligibility_conditions": conditions}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract employee eligibility conditions (Final/ee_extraction)."
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        default=None,
        help="Chunk JSON path(s). Default: fill-in CHUNK_FILE_PATHS.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-filename", default=OUTPUT_FILENAME)
    parser.add_argument("--config", default=str(YML_CONFIG_PATH))
    parser.add_argument("--src-dir", default=str(SRC_DIR))
    parser.add_argument("--function", default=FUNCTION_NAME)
    parser.add_argument("--sub-function", default=SUB_FUNCTION_NAME)
    parser.add_argument("--model-key", default=MODEL_KEY)
    parser.add_argument(
        "--filter-ee",
        action="store_true",
        help="Send only employee-eligibility-tagged chunks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_ee_conditions(
        chunk_file_paths=args.chunks or CHUNK_FILE_PATHS,
        output_dir=args.output_dir or OUTPUT_DIR,
        output_filename=args.output_filename,
        yml_config_path=args.config,
        src_dir=args.src_dir,
        function_name=args.function,
        sub_function_name=args.sub_function,
        model_key=args.model_key,
        filter_to_ee_chunks=args.filter_ee or FILTER_TO_EE_CHUNKS,
    )
