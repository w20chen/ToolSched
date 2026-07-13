from __future__ import annotations


def infer_resource_class(tool_family: str, operation: str, features: dict) -> str:
    if operation in {"pytest", "test"}:
        return "cpu_memory_mixed"
    if operation in {"build", "python"}:
        return "cpu"
    if operation in {"grep", "find"}:
        if features.get("has_recursive_hint") or features.get("include_count", 0) > 0:
            return "io_search"
        return "search"
    if operation in {"read_file", "write_file", "edit_file", "list_dir", "git_diff", "git_status"}:
        return "file_io"
    if tool_family == "network":
        return "network"
    if tool_family == "control":
        return "light_control"
    return "unknown"

