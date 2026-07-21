"""
Notice and certification requirements extractor for final_pipeline.

Uses Final/notice_cert (open schema). No predefined schema — only regulation
chunks + extraction task (notice | certification).

Data layout examples:
  chunks:
    final_pipeline/data/federal/fmla_chunk_schema_filled.json
    final_pipeline/data/GA/chunk_schema_filled.json
    final_pipeline/data/TN/tn_extracted_sections_chunk_schema_filled.json
    final_pipeline/data/CA/{cfra,pdl}_chunk_schema_filled.json
  GT (for later eval, not required by this extractor):
    final_pipeline/data/*/gt/*notice_gt.json
    final_pipeline/data/*/gt/*cert_gt.json
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
    slim_chunk,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Fill-in fields (edit these, or override via function args / CLI)
# ---------------------------------------------------------------------------

CHUNK_FILE_PATHS: ChunkPaths = PIPELINE_DIR / "data" / "federal" / "fmla_chunk_schema_filled.json"

# "notice" or "certification"
CATEGORY: str = "notice"

OUTPUT_DIR: Path = DEFAULT_OUTPUT_DIR
OUTPUT_FILENAME: Optional[str] = None  # default: {category}.json

YML_CONFIG_PATH: Path = DEFAULT_CONFIG
SRC_DIR: Path = DEFAULT_SRC_DIR

FUNCTION_NAME: str = "Final"
SUB_FUNCTION_NAME: str = "notice_cert"
MODEL_KEY: str = "ari_model"

TARGET_NOTICE_FUNCTION: str = "notice_requirements"

CATEGORY_TASK_LABELS = {
    "notice": "notice requirements",
    "certification": "certification requirements",
}

CONCISENESS_INSTRUCTION = (
    "Keep every field, required_value, unit, and condition_name as short and "
    "concise as possible. Use numbers, booleans, or 1-6 word keyword phrases "
    "only. Do not use full regulatory sentences in field, required_value, or unit."
)


def build_notice_rendered_input(
    task_label: str, category: str, chunks: List[Dict[str, Any]]
) -> str:
    payload = {
        "extraction_task": task_label,
        "category": category,
        "regulation_chunks": [slim_chunk(c) for c in chunks],
        "conciseness_rule": CONCISENESS_INSTRUCTION,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def extract_notice_cert(
    chunk_file_paths: ChunkPaths = CHUNK_FILE_PATHS,
    category: str = CATEGORY,
    output_dir: Optional[str | Path] = OUTPUT_DIR,
    output_filename: Optional[str] = OUTPUT_FILENAME,
    yml_config_path: str | Path = YML_CONFIG_PATH,
    src_dir: str | Path = SRC_DIR,
    function_name: str = FUNCTION_NAME,
    sub_function_name: str = SUB_FUNCTION_NAME,
    model_key: str = MODEL_KEY,
    target_notice_function: str = TARGET_NOTICE_FUNCTION,
) -> Dict[str, Any]:
    """
    Extract notice or certification requirements and write JSON.

    Returns the parsed payload with keys category + conditions.
    """
    category_norm = category.strip().lower()
    if category_norm not in CATEGORY_TASK_LABELS:
        raise ValueError(
            f"category must be one of {sorted(CATEGORY_TASK_LABELS)}, got {category!r}"
        )
    task_label = CATEGORY_TASK_LABELS[category_norm]

    all_chunks = load_chunks(chunk_file_paths)
    notice_chunks = [
        c for c in all_chunks if chunk_matches_function(c, target_notice_function)
    ]
    if not notice_chunks:
        raise RuntimeError(
            f"No chunks tagged with function={target_notice_function!r}."
        )

    rendered_input = build_notice_rendered_input(
        task_label, category_norm, notice_chunks
    )
    print(
        f"Category={category_norm!r}: {len(notice_chunks)} notice_requirements "
        f"chunk(s) → {sub_function_name}"
    )

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
        input_data={"rendered_input": rendered_input},
    )
    print(f"Token usage: {usage}")

    parsed = parse_llm_json(raw)
    conditions = parsed.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("Expected conditions to be a list")

    out_dir = resolve_path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    fname = output_filename or f"{category_norm}.json"
    out_path = out_dir / fname

    record = {
        "extracted_at": utc_now_iso(),
        "source": as_source_list(chunk_file_paths),
        "function_name": function_name,
        "sub_function_name": sub_function_name,
        "model_key": model_key,
        "matched_chunk_count": len(notice_chunks),
        "category": parsed.get("category", category_norm),
        "conditions": conditions,
        "token_usage": usage,
        "raw_llm_response": raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False),
    }
    save_json(record, out_path)
    print(f"Saved → {out_path}")
    return {
        "category": record["category"],
        "conditions": conditions,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract notice or certification requirements (Final/notice_cert)."
    )
    parser.add_argument("--chunks", nargs="+", default=None)
    parser.add_argument(
        "--category",
        choices=sorted(CATEGORY_TASK_LABELS),
        default=None,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--config", default=str(YML_CONFIG_PATH))
    parser.add_argument("--src-dir", default=str(SRC_DIR))
    parser.add_argument("--function", default=FUNCTION_NAME)
    parser.add_argument("--sub-function", default=SUB_FUNCTION_NAME)
    parser.add_argument("--model-key", default=MODEL_KEY)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_notice_cert(
        chunk_file_paths=args.chunks or CHUNK_FILE_PATHS,
        category=args.category or CATEGORY,
        output_dir=args.output_dir or OUTPUT_DIR,
        output_filename=args.output_filename or OUTPUT_FILENAME,
        yml_config_path=args.config,
        src_dir=args.src_dir,
        function_name=args.function,
        sub_function_name=args.sub_function,
        model_key=args.model_key,
    )
