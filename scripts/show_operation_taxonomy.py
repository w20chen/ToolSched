from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolsched.features.command import EXEC_TOOL_OPERATION, TOOL_FAMILY_BY_OPERATION, normalize_operation
from toolsched.features.exec_classifier import classify_exec_tool_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Show ToolSched operation taxonomy or classify one tool call.")
    parser.add_argument("--tool", default="exec", help="Tool name to classify. Defaults to raw exec.")
    parser.add_argument("--command", help="Shell command for exec-like tools.")
    parser.add_argument("--payload-json", help="Full payload JSON. --command overrides its command key.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    if args.command is not None or args.payload_json is not None:
        payload = parse_payload(args.payload_json)
        if args.command is not None:
            payload["command"] = args.command
        result = classify_call(args.tool, payload)
        emit(result, args.format)
        return

    emit({"operations": operation_catalog()}, args.format)


def parse_payload(payload_json: str | None) -> dict[str, Any]:
    if not payload_json:
        return {}
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --payload-json: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--payload-json must decode to an object")
    return payload


def classify_call(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    classified_tool = classify_exec_tool_name(tool, payload)
    operation, family = normalize_operation(classified_tool, payload)
    return {
        "input_tool": tool,
        "classified_tool": classified_tool,
        "operation": operation,
        "tool_family": family,
        "command": payload.get("command"),
    }


def operation_catalog() -> list[dict[str, Any]]:
    exec_fallbacks: dict[str, list[str]] = defaultdict(list)
    for category, operation in EXEC_TOOL_OPERATION.items():
        exec_fallbacks[operation].append(f"exec-{category}")

    rows = []
    for operation in sorted(TOOL_FAMILY_BY_OPERATION):
        rows.append({
            "operation": operation,
            "tool_family": TOOL_FAMILY_BY_OPERATION[operation],
            "exec_tool_fallbacks": sorted(exec_fallbacks.get(operation, [])),
        })
    return rows


def emit(payload: dict[str, Any], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if "operations" in payload:
        print_catalog(payload["operations"])
    else:
        print_classification(payload)


def print_catalog(rows: list[dict[str, Any]]) -> None:
    print("operation | tool_family | exec_tool_fallbacks")
    print("--- | --- | ---")
    for row in rows:
        fallbacks = ", ".join(row["exec_tool_fallbacks"])
        print(f"{row['operation']} | {row['tool_family']} | {fallbacks}")


def print_classification(row: dict[str, Any]) -> None:
    print(f"input_tool: {row['input_tool']}")
    print(f"classified_tool: {row['classified_tool']}")
    print(f"operation: {row['operation']}")
    print(f"tool_family: {row['tool_family']}")
    if row.get("command") is not None:
        print(f"command: {row['command']}")


if __name__ == "__main__":
    main()
