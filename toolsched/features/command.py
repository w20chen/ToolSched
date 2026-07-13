from __future__ import annotations

import re
import shlex
from typing import Any


SEARCH_WORDS = ("grep", "rg", "find", "ripgrep")
TEST_WORDS = ("pytest", "tox", "unittest", "nosetests")
BUILD_WORDS = ("make", "cmake", "ninja", "pip", "npm", "cargo", "go test", "mvn")


def normalize_operation(tool: str, payload: dict[str, Any]) -> tuple[str, str]:
    text = command_text(tool, payload).lower()
    if tool in {"read_file", "write_file", "edit_file", "list_dir", "web_search", "web_fetch"}:
        op = tool
    elif any(word in text for word in TEST_WORDS):
        op = "pytest" if "pytest" in text else "test"
    elif any(word in text for word in SEARCH_WORDS):
        op = "grep" if ("grep" in text or "rg" in text) else "find"
    elif "git diff" in text:
        op = "git_diff"
    elif "git status" in text:
        op = "git_status"
    elif any(word in text for word in BUILD_WORDS):
        op = "build"
    elif "python" in text:
        op = "python"
    elif "curl" in text or "wget" in text:
        op = "download"
    else:
        op = tool.replace("exec-", "")
    return op, tool_family(tool, op)


def tool_family(tool: str, operation: str) -> str:
    if tool in {"read_file", "write_file", "edit_file", "list_dir"}:
        return "file"
    if tool in {"web_search", "web_fetch"} or operation in {"download"}:
        return "network"
    if operation in {"grep", "find"}:
        return "search"
    if operation in {"pytest", "test"}:
        return "test"
    if operation in {"build", "python"}:
        return "terminal"
    if tool.startswith("exec"):
        return "terminal"
    if tool in {"ls", "cd", "pwd"}:
        return "control"
    return "tool"


def command_text(tool: str, payload: dict[str, Any]) -> str:
    if "command" in payload:
        return str(payload.get("command") or "")
    if "path" in payload:
        return f"{tool} {payload.get('path')}"
    return " ".join(f"{k}={v}" for k, v in sorted(payload.items()))


def extract_command_features(tool: str, payload: dict[str, Any], preview: str) -> dict[str, Any]:
    text = command_text(tool, payload)
    try:
        argv = shlex.split(text, posix=False)
    except ValueError:
        argv = text.split()
    flags = [a for a in argv if a.startswith("-")]
    path_like = re.findall(r"[/\\][\w./\\-]+", text)
    include_count = text.count("--include")
    return {
        "command_len": len(text),
        "argv_count": len(argv),
        "flag_count": len(flags),
        "has_pipe": "|" in text,
        "has_recursive_hint": any(x in text for x in [" -r", " -R", " --recursive", "grep -r", "rg "]),
        "include_count": include_count,
        "path_token_count": len(path_like),
        "input_key_count": len(payload),
        "preview_len": len(preview or ""),
        "preview_line_count": len((preview or "").splitlines()),
        "exit_code_nonzero": "Exit code: 1" in (preview or "") or '"error"' in (preview or ""),
    }

