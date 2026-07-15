"""Classify raw ``exec`` tool calls by the command they run.

This mirrors the core behavior of the trace collection classifier used by
agent-test-bench: raw ``exec`` calls are split into names such as
``exec-grep`` and ``exec-python`` while non-exec tools are left unchanged.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any


_COMMAND_CATEGORY_MAP: dict[str, str] = {
    "grep": "grep",
    "egrep": "grep",
    "fgrep": "grep",
    "rg": "grep",
    "ripgrep": "grep",
    "find": "find",
    "fd": "find",
    "locate": "find",
    "which": "which",
    "whereis": "which",
    "type": "which",
    "cat": "cat",
    "head": "head",
    "tail": "tail",
    "less": "less",
    "more": "less",
    "ls": "ls",
    "dir": "ls",
    "cd": "cd",
    "pushd": "cd",
    "popd": "cd",
    "pwd": "pwd",
    "mkdir": "mkdir",
    "cp": "cp",
    "mv": "mv",
    "rm": "rm",
    "rmdir": "rm",
    "chmod": "chmod",
    "chown": "chmod",
    "touch": "touch",
    "ln": "ln",
    "sed": "sed",
    "awk": "awk",
    "sort": "sort",
    "uniq": "uniq",
    "wc": "wc",
    "tr": "tr",
    "cut": "cut",
    "tee": "tee",
    "diff": "diff",
    "patch": "diff",
    "xargs": "xargs",
    "base64": "base64",
    "echo": "echo",
    "printf": "echo",
    "source": "source",
    "export": "export",
    "env": "env",
    "unset": "env",
    "set": "env",
    "python": "python",
    "python3": "python",
    "python3.12": "python",
    "python3.11": "python",
    "python3.10": "python",
    "python3.9": "python",
    "pip": "pip",
    "pip3": "pip",
    "pytest": "pytest",
    "django": "pytest",
    "R": "r",
    "Rscript": "r",
    "scala": "scala",
    "scalac": "scala",
    "sbt": "sbt",
    "spark-submit": "spark",
    "spark-shell": "spark",
    "pyspark": "spark",
    "node": "node",
    "npm": "npm",
    "npx": "npm",
    "yarn": "npm",
    "pnpm": "npm",
    "git": "git",
    "curl": "curl",
    "wget": "curl",
    "jupyter": "jupyter",
    "apt": "apt",
    "apt-get": "apt",
    "apt-cache": "apt",
    "yum": "apt",
    "dnf": "apt",
    "apk": "apt",
    "brew": "apt",
    "conda": "conda",
    "mamba": "conda",
    "docker": "docker",
    "podman": "docker",
    "systemctl": "systemctl",
    "service": "systemctl",
    "ps": "ps",
    "kill": "kill",
    "killall": "kill",
    "top": "top",
    "htop": "top",
    "df": "df",
    "du": "df",
    "free": "free",
    "mount": "mount",
    "umount": "mount",
    "make": "make",
    "cmake": "make",
    "ninja": "make",
    "gcc": "gcc",
    "g++": "gcc",
    "clang": "gcc",
    "clang++": "gcc",
    "sqlite3": "sqlite3",
    "duckdb": "duckdb",
    "psql": "psql",
    "mysql": "mysql",
    "mariadb": "mariadb",
    "mongosh": "mongosh",
    "redis-cli": "redis-cli",
    "tar": "tar",
    "gzip": "tar",
    "gunzip": "tar",
    "zip": "tar",
    "unzip": "tar",
    "xxd": "xxd",
    "md5sum": "checksum",
    "sha1sum": "checksum",
    "sha256sum": "checksum",
    "sha512sum": "checksum",
    "file": "file",
    "true": "true",
    "false": "true",
    "test": "test",
    "sleep": "sleep",
    "date": "date",
    "time": "time",
    "watch": "watch",
    "man": "man",
    "info": "man",
    "su": "su",
    "sudo": "su",
    "bash": "bash",
    "sh": "bash",
    "zsh": "bash",
}

_COMMAND_PRIORITY: dict[str, int] = {
    "pip": 4,
    "pip3": 4,
    "pytest": 4,
    "django": 4,
    "spark-submit": 4,
    "spark-shell": 4,
    "pyspark": 4,
    "python": 3,
    "python3": 3,
    "python3.12": 3,
    "python3.11": 3,
    "python3.10": 3,
    "python3.9": 3,
    "git": 3,
    "docker": 3,
    "podman": 3,
    "make": 3,
    "cmake": 3,
    "ninja": 3,
    "gcc": 3,
    "g++": 3,
    "clang": 3,
    "clang++": 3,
    "apt": 3,
    "apt-get": 3,
    "yum": 3,
    "dnf": 3,
    "apk": 3,
    "brew": 3,
    "conda": 3,
    "mamba": 3,
    "npm": 3,
    "npx": 3,
    "yarn": 3,
    "pnpm": 3,
    "node": 3,
    "systemctl": 3,
    "service": 3,
    "curl": 3,
    "wget": 3,
    "su": 3,
    "sudo": 3,
    "R": 3,
    "Rscript": 3,
    "scala": 3,
    "scalac": 3,
    "sbt": 3,
    "jupyter": 3,
    "sqlite3": 3,
    "duckdb": 3,
    "psql": 3,
    "mysql": 3,
    "mariadb": 3,
    "mongosh": 3,
    "redis-cli": 3,
    "grep": 2,
    "egrep": 2,
    "fgrep": 2,
    "rg": 2,
    "ripgrep": 2,
    "find": 2,
    "fd": 2,
    "sed": 2,
    "awk": 2,
    "diff": 2,
    "patch": 2,
    "cat": 2,
    "tar": 2,
    "gzip": 2,
    "gunzip": 2,
    "zip": 2,
    "unzip": 2,
    "chmod": 2,
    "chown": 2,
    "cp": 2,
    "mv": 2,
    "rm": 2,
    "rmdir": 2,
    "mkdir": 2,
    "touch": 2,
    "ln": 2,
    "kill": 2,
    "killall": 2,
    "mount": 2,
    "umount": 2,
    "ps": 2,
    "top": 2,
    "htop": 2,
    "df": 2,
    "du": 2,
    "free": 2,
    "which": 2,
    "whereis": 2,
    "man": 2,
    "watch": 2,
    "xxd": 2,
    "md5sum": 2,
    "sha1sum": 2,
    "sha256sum": 2,
    "sha512sum": 2,
    "file": 2,
    "base64": 2,
}

_SAFE_EXECUTABLE_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
_ENV_ASSIGN_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)
_MAX_EXECUTABLE_SLUG_LENGTH = 64
_SHELL_RESERVED_WORDS = frozenset({
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do",
    "done", "case", "esac", "in", "function", "select", "coproc",
})
_PYTHON_INTERPS = frozenset({
    "python", "python3", "python3.9", "python3.10", "python3.11", "python3.12",
})
_NAVIGATION_TOKENS = frozenset({"cd", "pushd", "popd", "pwd", "ls", "dir"})
_WRAPPERS = frozenset({"sudo", "nice", "nohup", "timeout", "env"})


def classify_exec_tool_name(tool_name: str, tool_args: str | dict[str, Any] | None) -> str:
    """Return a classified exec tool name, leaving non-exec tools unchanged."""
    if tool_name != "exec":
        return tool_name

    command = _command_from_args(tool_args)
    if not command:
        return tool_name

    base = _extract_base_command(command)
    if base == "exec":
        return tool_name
    category = _COMMAND_CATEGORY_MAP.get(base) or _safe_unknown_category(base)
    if category is None:
        return tool_name
    return f"exec-{category}"


def _command_from_args(tool_args: str | dict[str, Any] | None) -> str:
    if isinstance(tool_args, str):
        try:
            parsed = json.loads(tool_args)
        except (TypeError, json.JSONDecodeError):
            return ""
    elif isinstance(tool_args, dict):
        parsed = tool_args
    else:
        return ""

    command = parsed.get("command", "")
    if not command and isinstance(parsed.get("exec"), dict):
        command = parsed["exec"].get("command", "")
    return str(command or "")


def _extract_base_command(command: str) -> str:
    segments = _split_shell_segments(command)
    best_token = "exec"
    best_priority = -1
    for segment in segments:
        token = _tokenize_segment(segment)
        if not token:
            continue
        priority = _COMMAND_PRIORITY.get(token, 1)
        if priority >= best_priority:
            best_priority = priority
            best_token = token
    return best_token


def _split_shell_segments(command: str) -> list[str]:
    segments: list[str] = []
    current = ""
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single and (i == 0 or command[i - 1] != "\\"):
            in_double = not in_double

        if not in_single and not in_double:
            if ch == "|" and (i == 0 or command[i - 1] != "\\"):
                segments.append(current)
                current = ""
                i += 1
                continue
            if ch == "&" and i + 1 < len(command) and command[i + 1] == "&":
                segments.append(current)
                current = ""
                i += 2
                continue
            if ch == ";":
                segments.append(current)
                current = ""
                i += 1
                continue
        current += ch
        i += 1
    segments.append(current)
    return segments


def _tokenize_segment(segment: str) -> str:
    try:
        parts = shlex.split(segment.strip(), posix=True)
    except ValueError:
        return ""
    if not parts:
        return ""

    token_idx = _skip_assignments(parts, 0)
    while token_idx < len(parts) and _basename(parts[token_idx]) in _WRAPPERS:
        wrapper = _basename(parts[token_idx])
        token_idx += 1
        if wrapper == "env":
            token_idx = _skip_assignments(parts, token_idx)
        while token_idx < len(parts) and parts[token_idx].startswith("-"):
            token_idx += 2 if _option_likely_has_value(parts[token_idx]) else 1
        token_idx = _skip_assignments(parts, token_idx)

    if token_idx >= len(parts):
        return ""
    token = _basename(parts[token_idx])

    if token == "command":
        token_idx += 1
        while token_idx < len(parts) and parts[token_idx].startswith("-"):
            if "v" in parts[token_idx][1:] or "V" in parts[token_idx][1:]:
                return "exec"
            token_idx += 1
        if token_idx >= len(parts):
            return "command"
        token = _basename(parts[token_idx])

    if token in _NAVIGATION_TOKENS:
        best_action_token = ""
        best_action_priority = -1
        for raw_token in parts[token_idx + 1:]:
            candidate = _basename(raw_token)
            priority = _COMMAND_PRIORITY.get(candidate, -1)
            if priority > best_action_priority:
                best_action_token = candidate
                best_action_priority = priority
        if best_action_priority >= 3:
            token = best_action_token

    if token == "xargs":
        xargs_idx = _xargs_command_index(parts, token_idx + 1)
        if xargs_idx is not None:
            token = _basename(parts[xargs_idx])

    if token in _PYTHON_INTERPS and len(parts) > token_idx + 2 and parts[token_idx + 1] == "-m":
        module_token = parts[token_idx + 2]
        if module_token in _COMMAND_CATEGORY_MAP:
            token = module_token

    return token


def _skip_assignments(parts: list[str], start: int) -> int:
    idx = start
    while idx < len(parts) and _ENV_ASSIGN_TOKEN_RE.fullmatch(parts[idx]):
        idx += 1
    return idx


def _xargs_command_index(parts: list[str], start: int) -> int | None:
    idx = start
    while idx < len(parts):
        token = parts[idx]
        if token == "--":
            return idx + 1 if idx + 1 < len(parts) else None
        if not token.startswith("-"):
            return idx
        idx += 2 if _option_likely_has_value(token) else 1
    return None


def _option_likely_has_value(option: str) -> bool:
    return option in {
        "-a", "-E", "-I", "-L", "-n", "-P", "-s", "-d", "-k", "-u", "-g",
        "-h", "-p", "-r", "-t", "-c", "-o", "-e",
    }


def _basename(token: str) -> str:
    return token.rsplit("/", 1)[-1]


def _safe_unknown_category(token: str) -> str | None:
    basename = _basename(token).lower()
    if basename in _SHELL_RESERVED_WORDS:
        return None
    if not basename or len(basename) > _MAX_EXECUTABLE_SLUG_LENGTH:
        return None
    if _SAFE_EXECUTABLE_RE.fullmatch(basename) is None:
        return None
    return basename
