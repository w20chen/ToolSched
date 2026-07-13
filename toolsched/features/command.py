from __future__ import annotations

import re
import shlex
from typing import Any


TOOL_FAMILY_BY_OPERATION = {
    "data_script": "data_analysis_scripting",
    "shell_script": "data_analysis_scripting",
    "test_run": "test_execution",
    "project_build": "package_environment_mgmt",
    "package_install": "package_environment_mgmt",
    "environment_manage": "package_environment_mgmt",
    "text_search_simple": "search_text_processing",
    "text_search_recursive": "search_text_processing",
    "text_transform": "search_text_processing",
    "directory_list": "file_navigation",
    "file_discovery": "file_navigation",
    "working_directory": "file_navigation",
    "version_control_status": "version_control",
    "version_control_diff": "version_control",
    "version_control_history": "version_control",
    "version_control_update": "version_control",
    "file_read": "file_io",
    "file_write": "file_io",
    "file_edit": "file_io",
    "memory_read": "file_io",
    "memory_write": "file_io",
    "web_search": "web_network",
    "web_fetch": "web_network",
    "download": "web_network",
    "unknown_command": "data_analysis_scripting",
}


TEST_WORDS = (
    "pytest", "tox", "unittest", "nosetests", "go test", "cargo test",
    "npm test", "pnpm test", "yarn test", "mvn test",
)
BUILD_WORDS = (
    "make", "cmake", "ninja", "cargo build", "go build", "mvn package",
    "mvn install", "npm run build", "pnpm build", "yarn build",
)
PACKAGE_INSTALL_WORDS = (
    "pip install", "pip3 install", "npm install", "pnpm install",
    "yarn install", "poetry install", "conda install", "apt install",
    "apt-get install", "brew install",
)
ENV_WORDS = ("venv", "virtualenv", "conda create", "conda env", "poetry env")
SCRIPT_WORDS = ("python", "python3", "ipython", "rscript", "node", "julia")
TEXT_SEARCH_WORDS = ("grep", "rg", "ripgrep")
TEXT_TRANSFORM_WORDS = ("sed", "awk", "sort", "uniq", "wc", "xargs")
FILE_READ_WORDS = ("cat", "head", "tail", "less")
FILE_DISCOVERY_WORDS = ("find", "fd")


def normalize_operation(tool: str, payload: dict[str, Any]) -> tuple[str, str]:
    """Return (operation, tool_family).

    The hierarchy is:

        tool -> operation -> tool_family

    ``tool`` is the concrete invoked tool. ``operation`` is the load-oriented
    abstraction, so it deliberately splits cases such as simple text search and
    recursive text search. ``tool_family`` is the functional grouping shown in
    the project taxonomy; tools in the same family need not have similar load.
    """

    tool_l = tool.lower()
    text = command_text(tool, payload).lower()

    if tool_l == "read_file":
        return _pair("file_read")
    if tool_l == "write_file":
        return _pair("file_write")
    if tool_l == "edit_file":
        return _pair("file_edit")
    if tool_l == "list_dir":
        return _pair("directory_list")
    if tool_l == "web_search":
        return _pair("web_search")
    if tool_l == "web_fetch":
        return _pair("web_fetch")
    if _is_memory_read(tool_l):
        return _pair("memory_read")
    if _is_memory_write(tool_l):
        return _pair("memory_write")

    if _has_any(text, TEST_WORDS):
        return _pair("test_run")
    if _has_any(text, PACKAGE_INSTALL_WORDS):
        return _pair("package_install")
    if _has_any(text, ENV_WORDS):
        return _pair("environment_manage")
    if _has_any(text, BUILD_WORDS):
        return _pair("project_build")

    if _has_command(text, "git"):
        if "git diff" in text or "git show" in text:
            return _pair("version_control_diff")
        if "git status" in text:
            return _pair("version_control_status")
        if "git log" in text or "git blame" in text:
            return _pair("version_control_history")
        return _pair("version_control_update")

    if _has_command_any(text, TEXT_SEARCH_WORDS) or tool_l in {"exec-grep", "exec-rg"}:
        if _is_recursive_search(text, payload):
            return _pair("text_search_recursive")
        return _pair("text_search_simple")
    if _has_command_any(text, FILE_DISCOVERY_WORDS):
        return _pair("file_discovery")
    if _has_command_any(text, TEXT_TRANSFORM_WORDS):
        return _pair("text_transform")
    if _has_command_any(text, FILE_READ_WORDS):
        return _pair("file_read")
    if _has_command(text, "ls"):
        return _pair("directory_list")
    if _has_command(text, "pwd") or _has_command(text, "cd"):
        return _pair("working_directory")
    if "curl" in text or "wget" in text:
        return _pair("download")
    if _has_any(text, SCRIPT_WORDS):
        return _pair("data_script")
    if tool_l.startswith("exec"):
        return _pair("shell_script")
    return _pair("unknown_command")


def tool_family(tool: str, operation: str) -> str:
    return TOOL_FAMILY_BY_OPERATION.get(operation, "data_analysis_scripting")


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
        "has_recursive_hint": _is_recursive_search(text.lower(), payload),
        "include_count": include_count,
        "path_token_count": len(path_like),
        "input_key_count": len(payload),
        "preview_len": len(preview or ""),
        "preview_line_count": len((preview or "").splitlines()),
        "exit_code_nonzero": "Exit code: 1" in (preview or "") or '"error"' in (preview or ""),
    }


def _pair(operation: str) -> tuple[str, str]:
    return operation, tool_family("", operation)


def _is_recursive_search(text: str, payload: dict[str, Any]) -> bool:
    recursive_flags = (" -r", " -R", " --recursive", "grep -r", "grep -R")
    if any(flag in text for flag in recursive_flags):
        return True
    try:
        argv = shlex.split(text, posix=False)
    except ValueError:
        argv = text.split()
    if argv and argv[0].lower() in {"rg", "ripgrep"}:
        positionals = [arg for arg in argv[1:] if not arg.startswith("-")]
        targets = positionals[1:]
        return not targets or any(not _looks_like_file_target(target) for target in targets)
    if "--include" in text or "**" in text:
        return True
    path = str(payload.get("path") or payload.get("directory") or "")
    return bool(path and not re.search(r"\.[A-Za-z0-9]{1,8}$", path))


def _is_memory_read(tool: str) -> bool:
    return "memory" in tool and any(token in tool for token in ("retrieve", "list", "search"))


def _is_memory_write(tool: str) -> bool:
    return "memory" in tool and any(
        token in tool for token in ("add", "clear", "remove", "replace", "update", "set")
    )


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _has_command(text: str, command: str) -> bool:
    return re.search(rf"(^|[;&|()\s]){re.escape(command)}($|[;&|()\s])", text) is not None


def _has_command_any(text: str, commands: tuple[str, ...]) -> bool:
    return any(_has_command(text, command) for command in commands)


def _looks_like_file_target(value: str) -> bool:
    cleaned = value.strip("\"'")
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}$", cleaned))
