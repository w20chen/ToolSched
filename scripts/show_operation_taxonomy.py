from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolsched.features.command import (
    DIRECT_TOOL_OPERATION,
    EXEC_TOOL_OPERATION,
    TOOL_FAMILY_BY_OPERATION,
    normalize_operation,
)
from toolsched.features.exec_classifier import classify_exec_tool_name


COMMAND_SENSITIVE_TOOLS_BY_OPERATION = {
    "package_install": [
        "exec-apt",
        "exec-conda",
        "exec-npm",
        "exec-pip",
        "exec-poetry",
        "exec-python",
        "exec-uv",
    ],
    "project_build": [
        "exec-cargo",
        "exec-docker",
        "exec-gcc",
        "exec-go",
        "exec-gradle",
        "exec-make",
        "exec-mvn",
        "exec-npm",
        "exec-python",
    ],
    "shell_script": ["exec"],
    "test_run": [
        "exec-cargo",
        "exec-go",
        "exec-make",
        "exec-mvn",
        "exec-npm",
        "exec-pytest",
        "exec-python",
    ],
    "text_search_recursive": ["exec-grep"],
    "text_search_simple": ["exec-grep"],
    "version_control_diff": ["exec-git"],
    "version_control_history": ["exec-git"],
    "version_control_status": ["exec-git"],
    "version_control_update": ["exec-git"],
}


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
    tools_by_operation: dict[str, set[str]] = defaultdict(set)

    for tool, operation in DIRECT_TOOL_OPERATION.items():
        tools_by_operation[operation].add(tool)
    for category, operation in EXEC_TOOL_OPERATION.items():
        tools_by_operation[operation].add(f"exec-{category}")
    for operation, tools in COMMAND_SENSITIVE_TOOLS_BY_OPERATION.items():
        tools_by_operation[operation].update(tools)

    rows = []
    for operation in sorted(TOOL_FAMILY_BY_OPERATION):
        rows.append({
            "operation": operation,
            "tool_family": TOOL_FAMILY_BY_OPERATION[operation],
            "tools": sorted(tools_by_operation.get(operation, [])),
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
    print("operation | tool_family | tools")
    print("--- | --- | ---")
    for row in rows:
        tools = ", ".join(row["tools"])
        print(f"{row['operation']} | {row['tool_family']} | {tools}")
    print()
    print("Note: exec tools can appear under multiple operations when command text refines the load class.")


def print_classification(row: dict[str, Any]) -> None:
    print(f"input_tool: {row['input_tool']}")
    print(f"classified_tool: {row['classified_tool']}")
    print(f"operation: {row['operation']}")
    print(f"tool_family: {row['tool_family']}")
    if row.get("command") is not None:
        print(f"command: {row['command']}")


if __name__ == "__main__":
    main()
