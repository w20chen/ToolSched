from __future__ import annotations


RESOURCE_CLASS_BY_OPERATION = {
    "data_script": "cpu",
    "shell_script": "unknown",
    "test_run": "cpu_memory_mixed",
    "project_build": "cpu_memory_mixed",
    "package_install": "network_disk_io",
    "environment_manage": "file_io",
    "text_search_simple": "search",
    "text_search_recursive": "io_search",
    "text_transform": "cpu_io_mixed",
    "directory_list": "metadata_io",
    "file_discovery": "metadata_io",
    "working_directory": "light_control",
    "version_control_status": "metadata_io",
    "version_control_diff": "file_io",
    "version_control_history": "file_io",
    "version_control_update": "network_disk_io",
    "file_read": "file_io",
    "file_write": "file_io",
    "file_edit": "file_io",
    "file_mutation": "file_io",
    "archive_operation": "cpu_io_mixed",
    "memory_read": "light_control",
    "memory_write": "light_control",
    "web_search": "network",
    "web_fetch": "network",
    "download": "network_disk_io",
    "container_operation": "cpu_memory_mixed",
    "database_query": "cpu_io_mixed",
    "system_operation": "metadata_io",
    "shell_control": "light_control",
    "unknown_command": "unknown",
}


def infer_resource_class(tool_family: str, operation: str, features: dict) -> str:
    """Return the fixed load class for an operation.

    The relation is many-to-one: every operation has exactly one resource
    class, while many operations may share the same resource class. The
    tool_family argument is kept for backward-compatible call sites but is not
    part of the decision.
    """

    return RESOURCE_CLASS_BY_OPERATION.get(operation, "unknown")
