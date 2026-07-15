import unittest

from toolsched.features.command import DIRECT_TOOL_OPERATION, normalize_operation
from toolsched.features.exec_classifier import classify_exec_tool_name


class ToolTaxonomyTests(unittest.TestCase):
    def test_text_search_splits_load_under_same_functional_family(self) -> None:
        simple_op, simple_family = normalize_operation(
            "exec-grep", {"command": "grep TODO README.md"}
        )
        recursive_op, recursive_family = normalize_operation(
            "exec-grep", {"command": "grep -R TODO src"}
        )

        self.assertEqual(simple_family, "search_text_processing")
        self.assertEqual(recursive_family, "search_text_processing")
        self.assertEqual(simple_op, "text_search_simple")
        self.assertEqual(recursive_op, "text_search_recursive")

    def test_package_family_can_contain_different_load_classes(self) -> None:
        install_op, install_family = normalize_operation(
            "exec-sh", {"command": "pip install -r requirements.txt"}
        )
        build_op, build_family = normalize_operation(
            "exec-sh", {"command": "npm run build"}
        )

        self.assertEqual(install_family, "package_environment_mgmt")
        self.assertEqual(build_family, "package_environment_mgmt")

    def test_file_navigation_is_separate_from_file_io(self) -> None:
        list_op, list_family = normalize_operation("list_dir", {"path": "src"})
        read_op, read_family = normalize_operation("read_file", {"path": "src/app.py"})

        self.assertEqual((list_op, list_family), ("directory_list", "file_navigation"))
        self.assertEqual((read_op, read_family), ("file_read", "file_io"))

    def test_direct_tool_operation_catalog_matches_normalization(self) -> None:
        for tool, expected_operation in DIRECT_TOOL_OPERATION.items():
            with self.subTest(tool=tool):
                operation, _ = normalize_operation(tool, {})
                self.assertEqual(operation, expected_operation)

    def test_leading_cd_does_not_hide_the_executed_operation(self) -> None:
        script_op, _ = normalize_operation(
            "exec",
            {"command": "cd /testbed && python -c \"print('ok')\""},
        )
        directory_op, _ = normalize_operation("exec", {"command": "cd /testbed && pwd"})

        self.assertEqual(script_op, "data_script")
        self.assertEqual(directory_op, "working_directory")

    def test_raw_exec_tool_name_is_classified_from_command(self) -> None:
        self.assertEqual(
            classify_exec_tool_name("exec", {"command": "cd /testbed && python -m pytest tests"}),
            "exec-pytest",
        )
        self.assertEqual(
            classify_exec_tool_name("exec", {"command": "find src -type f | xargs grep TODO"}),
            "exec-grep",
        )

    def test_non_exec_tool_names_are_not_classified(self) -> None:
        self.assertEqual(
            classify_exec_tool_name("read_file", {"path": "src/app.py"}),
            "read_file",
        )

    def test_exec_tool_name_fallbacks_when_command_is_missing(self) -> None:
        cases = {
            "exec-pytest": ("test_run", "test_execution"),
            "exec-find": ("file_discovery", "file_navigation"),
            "exec-cat": ("file_read", "file_io"),
            "exec-ls": ("directory_list", "file_navigation"),
            "exec-git": ("version_control_update", "version_control"),
            "exec-cargo": ("project_build", "package_environment_mgmt"),
            "exec-tar": ("archive_operation", "file_io"),
            "exec-docker": ("container_operation", "package_environment_mgmt"),
            "exec-sqlite3": ("database_query", "data_analysis_scripting"),
        }

        for tool, expected in cases.items():
            with self.subTest(tool=tool):
                self.assertEqual(normalize_operation(tool, {}), expected)

    def test_command_text_overrides_broad_exec_tool_category_for_load(self) -> None:
        cases = [
            ("exec-python", {"command": "python setup.py test"}, "test_run"),
            ("exec-python", {"command": "python setup.py install"}, "package_install"),
            ("exec-make", {"command": "make test"}, "test_run"),
            ("exec-docker", {"command": "docker build ."}, "project_build"),
            ("exec-docker", {"command": "docker run image"}, "container_operation"),
            ("exec-tar", {"command": "tar -xf archive.tar.gz"}, "archive_operation"),
            ("exec-sqlite3", {"command": "sqlite3 db.sqlite .tables"}, "database_query"),
            ("exec-cp", {"command": "cp src/a dst/a"}, "file_mutation"),
        ]

        for tool, payload, expected_operation in cases:
            with self.subTest(tool=tool, command=payload["command"]):
                operation, family = normalize_operation(tool, payload)
                self.assertEqual(operation, expected_operation)
                self.assertNotEqual(family, "unknown")

    def test_find_piped_to_grep_is_recursive_search_load(self) -> None:
        operation, family = normalize_operation(
            "exec-grep",
            {"command": "find src -type f | xargs grep TODO"},
        )

        self.assertEqual(operation, "text_search_recursive")
        self.assertEqual(family, "search_text_processing")


if __name__ == "__main__":
    unittest.main()
