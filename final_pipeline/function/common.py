"""
Shared helpers for final_pipeline extractors (EE, LR, notice/cert).
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import yaml

FUNCTION_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = FUNCTION_DIR.parent  # final_pipeline/
PROJECT_ROOT = PIPELINE_DIR.parent
DEFAULT_SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG = DEFAULT_SRC_DIR / "configs" / "main_config.yaml"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "output"

PathLike = Union[str, Path]
ChunkPaths = Union[PathLike, Sequence[PathLike]]


def resolve_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def load_json(path: PathLike) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: PathLike) -> Path:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return out


def load_chunks(chunk_file_paths: ChunkPaths) -> List[Dict[str, Any]]:
    """Load one or more chunk JSON files and concatenate."""
    if isinstance(chunk_file_paths, (str, Path)):
        paths = [chunk_file_paths]
    else:
        paths = list(chunk_file_paths)
    if not paths:
        raise ValueError("At least one chunk file path is required.")

    chunks: List[Dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Chunk file must be a JSON array: {path}")
        chunks.extend(data)
    return chunks


def normalize_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip().lower() for x in value]
    return [str(value).strip().lower()]


def chunk_matches_function(chunk: Dict[str, Any], function: str) -> bool:
    return function.strip().lower() in normalize_to_list(chunk.get("function"))


def chunk_matches_reason_and_function(
    chunk: Dict[str, Any], reason: str, function: str
) -> bool:
    return (
        reason.strip().lower() in normalize_to_list(chunk.get("reason"))
        and function.strip().lower() in normalize_to_list(chunk.get("function"))
    )


def slim_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "regulation number": chunk.get("regulation number"),
        "reason": chunk.get("reason"),
        "function": chunk.get("function"),
        "raw text": chunk.get("raw text"),
    }


def reason_slug(reason: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", reason.strip().lower()).strip("_")


def normalize_empty_to_null(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_empty_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_empty_to_null(v) for v in obj]
    return None if obj == "" else obj


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_llm_json(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    text = str(response).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            from untruncate_json import complete

            return json.loads(complete(text))
        except Exception:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise ValueError(f"No JSON object found in LLM response:\n{text[:800]}")
            return json.loads(text[start : end + 1])


def setup_llm(
    yml_config_path: PathLike = DEFAULT_CONFIG,
    src_dir: PathLike = DEFAULT_SRC_DIR,
    model_key: str = "ari_model",
) -> Tuple[Any, Dict[str, Any], str, str]:
    """
    Chdir into src/, import LLM_Ops, and resolve provider/deployment for model_key.

    model_key examples from main_config.yaml: "lt", "ari_model".
    """
    src = resolve_path(src_dir)
    cfg_path = resolve_path(yml_config_path)

    os.chdir(src)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from ari.language_models.llm_ops import LLM_Ops  # noqa: E402

    with open(cfg_path, "r", encoding="utf-8") as f:
        main_config = yaml.load(f, Loader=yaml.FullLoader)

    provider_info_config = main_config["providers"]
    model_cfg = main_config["model_llm_configs"][model_key]
    return (
        LLM_Ops,
        provider_info_config,
        model_cfg["llm_provider"],
        model_cfg["llm_deployment"],
    )


def invoke_llm(
    *,
    LLM_Ops: Any,
    provider_info_config: Dict[str, Any],
    llm_provider: str,
    llm_deployment: str,
    function_name: str,
    sub_function_name: str,
    input_data: Dict[str, Any],
) -> Tuple[Any, Any]:
    llm = LLM_Ops(
        provider_info_config=provider_info_config,
        provider=llm_provider,
        model_deployment=llm_deployment,
        function_name=function_name,
        sub_function_name=sub_function_name,
        input_data=input_data,
    )
    response_obj = llm.invoke_model()
    return llm._return_usage_data(response_obj)


def default_run_output_dir(prefix: str = "run") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"{stamp}_{prefix}"


def as_source_list(chunk_file_paths: ChunkPaths) -> List[str]:
    if isinstance(chunk_file_paths, (str, Path)):
        return [str(resolve_path(chunk_file_paths))]
    return [str(resolve_path(p)) for p in chunk_file_paths]
