#!/usr/bin/env python3
"""
Leave Eligibility Engine — local web UI.

Serves a single-page app for the final_pipeline leave-eligibility rule engine:
  - browse the saved eligibility rules (EE / notice / certification / leave-reason
    condition packs) for each jurisdiction
  - submit a claim (paste JSON or upload file(s)) and evaluate it against those
    rules, seeing the decision, the reasoning, and every condition that was
    checked with its pass/fail result

Usage (from anywhere — paths are auto-detected relative to this file):
    python3 ui/server.py [--port 8791] [--host 0.0.0.0] [--no-browser]

Then open http://localhost:8791 in your browser.

On SageMaker Studio / Code Editor, bind stays 0.0.0.0 (default) and open:
    https://<domain>.studio.<region>.sagemaker.aws/<app>/default/proxy/8791/
(Use jupyterlab or jupyter depending on your space; Code Editor proxy support varies.)

No third-party dependencies — stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

UI_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = UI_DIR.parent

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from function.common import load_json  # noqa: E402
from function.test_case_evaluator import (  # noqa: E402
    LEAVE_REASONS,
    _gt_paths,
    load_all_cases,
    run_case,
)

INDEX_HTML = UI_DIR / "index.html"
JURISDICTIONS = ["federal", "CA", "GA", "TN"]

# Evaluation source: rules are always evaluated against the saved condition
# packs (internally keyed "gt" in the evaluator module — not surfaced in the UI).
SOURCE = "gt"

# SageMaker / Jupyter proxies forward URLs like /proxy/8791/api/rules (or
# /codeeditor/default/proxy/8791/...). Strip that prefix so route matching works.
_PROXY_PREFIX_RE = re.compile(r"^(?:/[^/]+)*/proxy/\d+")


def _route_path(raw: str) -> str:
    """Normalize request path: drop query string and any /proxy/<port> prefix."""
    path = urlparse(raw).path or "/"
    stripped = _PROXY_PREFIX_RE.sub("", path)
    if not stripped or stripped == "/":
        return "/"
    return stripped if stripped.startswith("/") else f"/{stripped}"


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(PIPELINE_DIR))
    except ValueError:
        return str(p)


def _load_rules() -> Dict[str, Any]:
    """Build a browsable structure of every saved condition pack per jurisdiction."""
    rules: Dict[str, Any] = {}
    for j in JURISDICTIONS:
        packs: Dict[str, Any] = {"leave_reasons": {}}
        base_paths = _gt_paths(j, "health_condition")  # ee/notice/cert stable across reasons
        for key in ("ee", "notice", "cert"):
            p = base_paths[key]
            if p.exists():
                packs[key] = {"path": _rel(p), "data": load_json(p)}
        for reason in sorted(LEAVE_REASONS):
            paths = _gt_paths(j, reason)
            p = paths["leave_reason"]
            if p.exists():
                packs["leave_reasons"][reason] = {"path": _rel(p), "data": load_json(p)}
        rules[j] = packs
    return rules


def _examples() -> Dict[str, Any]:
    """One representative sample claim per jurisdiction + leave-reason combo."""
    cases = load_all_cases()
    out: Dict[str, Any] = {}
    for c in cases:
        key = f"{c.get('jurisdiction')}::{c.get('leave_reason')}"
        prefer = c.get("case_type") == "pass"
        if key not in out or (prefer and out[key].get("_case_type") != "pass"):
            out[key] = {
                "jurisdiction": c.get("jurisdiction"),
                "leave_reason": c.get("leave_reason"),
                "narrative": c.get("narrative", ""),
                "facts": c.get("facts", {}),
                "_case_type": c.get("case_type"),
            }
    for v in out.values():
        v.pop("_case_type", None)
    return out


def _normalize_claim(raw: Any, idx: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Claim #{idx + 1}: expected a JSON object")

    if isinstance(raw.get("facts"), dict):
        case = dict(raw)
        facts = dict(raw["facts"])
    else:
        facts = dict(raw)
        case = {}

    jurisdiction = case.get("jurisdiction") or facts.get("jurisdiction")
    leave_reason = case.get("leave_reason") or facts.get("leave_reason")

    if not jurisdiction:
        raise ValueError(f"Claim #{idx + 1}: missing 'jurisdiction'")
    if not leave_reason:
        raise ValueError(f"Claim #{idx + 1}: missing 'leave_reason'")

    case["id"] = case.get("id") or raw.get("id") or f"claim-{idx + 1}"
    case["jurisdiction"] = jurisdiction
    case["leave_reason"] = leave_reason
    case["facts"] = facts
    case.setdefault("label", raw.get("label", ""))
    case.setdefault("narrative", raw.get("narrative", ""))
    return case


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path):
        if not path.exists():
            self.send_error(404, "Not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = _route_path(self.path)
        if path == "/" or path == "/index.html":
            self._send_html(INDEX_HTML)
        elif path == "/api/rules":
            try:
                self._send_json(_load_rules())
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=500)
        elif path == "/api/examples":
            try:
                self._send_json(_examples())
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, f"Not found: {path}")

    def do_POST(self):
        path = _route_path(self.path)
        if path == "/api/evaluate":
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw or b"{}")
                claims = payload.get("claims") or payload.get("cases") or []
                if not isinstance(claims, list) or not claims:
                    raise ValueError("No claims submitted")
                normalized = [_normalize_claim(c, i) for i, c in enumerate(claims)]
                results: List[Dict[str, Any]] = [run_case(c, SOURCE) for c in normalized]
                self._send_json({"results": results})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=400)
        else:
            self.send_error(404, f"Not found: {path}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8791)
    # 0.0.0.0 so SageMaker / remote proxies can reach the process (localhost-only
    # binds are unreachable from the Studio proxy sidecar).
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    local_url = f"http://127.0.0.1:{args.port}"
    print(f"Leave Eligibility Engine listening on {args.host}:{args.port}  (Ctrl+C to stop)")
    print(f"  Local:     {local_url}")
    print(f"  SageMaker: open …/proxy/{args.port}/ on your Studio / Code Editor URL")
    print(f"             (e.g. https://<domain>.studio.<region>.sagemaker.aws/jupyterlab/default/proxy/{args.port}/)")
    if not args.no_browser:
        try:
            webbrowser.open(local_url)
        except Exception:  # noqa: BLE001
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
