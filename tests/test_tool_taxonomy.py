import unittest

from toolsched.features.command import normalize_operation
from toolsched.features.resource_class import infer_resource_class


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
        self.assertEqual(infer_resource_class(simple_family, simple_op, {}), "search")
        self.assertEqual(infer_resource_class(recursive_family, recursive_op, {}), "io_search")

    def test_package_family_can_contain_different_load_classes(self) -> None:
        install_op, install_family = normalize_operation(
            "exec-sh", {"command": "pip install -r requirements.txt"}
        )
        build_op, build_family = normalize_operation(
            "exec-sh", {"command": "npm run build"}
        )

        self.assertEqual(install_family, "package_environment_mgmt")
        self.assertEqual(build_family, "package_environment_mgmt")
        self.assertEqual(infer_resource_class(install_family, install_op, {}), "network_disk_io")
        self.assertEqual(infer_resource_class(build_family, build_op, {}), "cpu_memory_mixed")

    def test_file_navigation_is_separate_from_file_io(self) -> None:
        list_op, list_family = normalize_operation("list_dir", {"path": "src"})
        read_op, read_family = normalize_operation("read_file", {"path": "src/app.py"})

        self.assertEqual((list_op, list_family), ("directory_list", "file_navigation"))
        self.assertEqual((read_op, read_family), ("file_read", "file_io"))
        self.assertEqual(infer_resource_class(list_family, list_op, {}), "metadata_io")
        self.assertEqual(infer_resource_class(read_family, read_op, {}), "file_io")


if __name__ == "__main__":
    unittest.main()
