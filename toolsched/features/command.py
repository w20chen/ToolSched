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
    "file_mutation": "file_io",
    "archive_operation": "file_io",
    "memory_read": "file_io",
    "memory_write": "file_io",
    "web_search": "web_network",
    "web_fetch": "web_network",
    "download": "web_network",
    "container_operation": "package_environment_mgmt",
    "database_query": "data_analysis_scripting",
    "system_operation": "data_analysis_scripting",
    "shell_control": "data_analysis_scripting",
    "unknown_command": "data_analysis_scripting",
}

DIRECT_TOOL_OPERATION = {
    "read_file": "file_read",
    "write_file": "file_write",
    "edit_file": "file_edit",
    "list_dir": "directory_list",
    "web_search": "web_search",
    "web_fetch": "web_fetch",
}

EXEC_TOOL_OPERATION = {
    "pytest": "test_run",
    "python": "data_script",
    "r": "data_script",
    "node": "data_script",
    "jupyter": "data_script",
    "scala": "data_script",
    "spark": "data_script",
    "pip": "environment_manage",
    "npm": "environment_manage",
    "apt": "package_install",
    "conda": "environment_manage",
    "make": "project_build",
    "cargo": "project_build",
    "go": "project_build",
    "mvn": "project_build",
    "gradle": "project_build",
    "gcc": "project_build",
    "uv": "environment_manage",
    "poetry": "environment_manage",
    "java": "data_script",
    "git": "version_control_update",
    "grep": "text_search_simple",
    "find": "file_discovery",
    "which": "file_discovery",
    "cat": "file_read",
    "head": "file_read",
    "tail": "file_read",
    "less": "file_read",
    "ls": "directory_list",
    "cd": "working_directory",
    "pwd": "working_directory",
    "sed": "text_transform",
    "awk": "text_transform",
    "sort": "text_transform",
    "uniq": "text_transform",
    "wc": "text_transform",
    "tr": "text_transform",
    "cut": "text_transform",
    "tee": "text_transform",
    "diff": "text_transform",
    "xargs": "text_transform",
    "base64": "text_transform",
    "curl": "download",
    "tar": "archive_operation",
    "docker": "container_operation",
    "sqlite3": "database_query",
    "duckdb": "database_query",
    "psql": "database_query",
    "mysql": "database_query",
    "mariadb": "database_query",
    "mongosh": "database_query",
    "redis-cli": "database_query",
    "mkdir": "file_mutation",
    "cp": "file_mutation",
    "mv": "file_mutation",
    "rm": "file_mutation",
    "chmod": "file_mutation",
    "touch": "file_mutation",
    "ln": "file_mutation",
    "systemctl": "system_operation",
    "ps": "system_operation",
    "kill": "system_operation",
    "top": "system_operation",
    "df": "system_operation",
    "free": "system_operation",
    "mount": "system_operation",
    "echo": "shell_control",
    "source": "shell_control",
    "export": "shell_control",
    "env": "shell_control",
    "true": "shell_control",
    "test": "shell_control",
    "sleep": "shell_control",
    "date": "shell_control",
    "time": "shell_control",
    "watch": "shell_control",
    "man": "shell_control",
    "su": "shell_control",
    "bash": "shell_control",
}


TEST_WORDS = (
    "pytest", "tox", "unittest", "nosetests", "go test", "cargo test",
    "npm test", "pnpm test", "yarn test", "mvn test", "make test",
    "python setup.py test", "python3 setup.py test", "setup.py test",
    "python -m unittest", "python3 -m unittest", "manage.py test",
    "django test",
)
BUILD_WORDS = (
    "make", "cmake", "ninja", "cargo build", "go build", "mvn package",
    "mvn install", "npm run build", "pnpm build", "yarn build",
    "python setup.py build", "python3 setup.py build", "setup.py build",
    "docker build", "podman build", "gradle build",
)
PACKAGE_INSTALL_WORDS = (
    "pip install", "pip3 install", "npm install", "pnpm install",
    "yarn install", "poetry install", "conda install", "apt install",
    "apt-get install", "brew install", "python setup.py install",
    "python3 setup.py install", "setup.py install", "uv pip install",
    "uv sync", "npm ci", "conda env create", "poetry add",
)
ENV_WORDS = ("venv", "virtualenv", "conda create", "conda env", "poetry env")
SCRIPT_WORDS = ("python", "python3", "ipython", "rscript", "node", "julia")
TEXT_SEARCH_WORDS = ("grep", "rg", "ripgrep")
TEXT_TRANSFORM_WORDS = ("sed", "awk", "sort", "uniq", "wc", "xargs")
FILE_READ_WORDS = ("cat", "head", "tail", "less")
FILE_DISCOVERY_WORDS = ("find", "fd")
ARCHIVE_WORDS = ("tar", "gzip", "gunzip", "zip", "unzip")
CONTAINER_WORDS = ("docker", "podman")
DATABASE_WORDS = ("sqlite3", "duckdb", "psql", "mysql", "mariadb", "mongosh", "redis-cli")
FILE_MUTATION_WORDS = ("mkdir", "cp", "mv", "rm", "rmdir", "chmod", "chown", "touch", "ln")
SYSTEM_WORDS = (
    "systemctl", "service", "ps", "kill", "killall", "top", "htop",
    "df", "du", "free", "mount", "umount",
)
SHELL_CONTROL_WORDS = (
    "echo", "printf", "source", "export", "env", "unset", "set",
    "true", "false", "sleep", "date", "time", "watch", "man", "info",
    "su", "sudo", "bash", "sh", "zsh",
)


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

    direct_operation = DIRECT_TOOL_OPERATION.get(tool_l)
    if direct_operation is not None:
        return _pair(direct_operation)
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
    if _is_working_directory_only(text):
        return _pair("working_directory")
    if _has_command_any(text, FILE_MUTATION_WORDS):
        return _pair("file_mutation")
    if "curl" in text or "wget" in text:
        return _pair("download")
    if _has_command_any(text, ARCHIVE_WORDS):
        return _pair("archive_operation")
    if _has_command_any(text, CONTAINER_WORDS):
        return _pair("container_operation")
    if _has_command_any(text, DATABASE_WORDS):
        return _pair("database_query")
    if _has_command_any(text, SYSTEM_WORDS):
        return _pair("system_operation")
    if _has_command_any(text, SHELL_CONTROL_WORDS):
        return _pair("shell_control")
    if _has_any(text, SCRIPT_WORDS):
        return _pair("data_script")
    exec_operation = _operation_from_exec_tool(tool_l)
    if exec_operation is not None:
        return _pair(exec_operation)
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


def _operation_from_exec_tool(tool: str) -> str | None:
    if not tool.startswith("exec-"):
        return None
    category = tool.removeprefix("exec-")
    return EXEC_TOOL_OPERATION.get(category)


def _is_recursive_search(text: str, payload: dict[str, Any]) -> bool:
    recursive_flags = (" -r", " -R", " --recursive", "grep -r", "grep -R")
    if any(flag in text for flag in recursive_flags):
        return True
    if _has_command(text, "find") and _has_command_any(text, TEXT_SEARCH_WORDS):
        return True
    if _has_command(text, "xargs") and _has_command_any(text, TEXT_SEARCH_WORDS):
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


def _is_working_directory_only(text: str) -> bool:
    """Return true only when every shell segment is a cd/pwd command.

    A leading ``cd /work &&`` is common setup for expensive commands and must
    not turn the whole invocation into a working-directory operation.
    """

    segments = [segment.strip() for segment in re.split(r"&&|\|\||[;|]", text) if segment.strip()]
    if not segments:
        return False
    return all(segment.split(maxsplit=1)[0].lower() in {"cd", "pwd"} for segment in segments)


def _looks_like_file_target(value: str) -> bool:
    cleaned = value.strip("\"'")
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}$", cleaned))
