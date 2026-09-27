import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("lilith_vm_executor", HERE / "lilith_vm_executor.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_stable_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "b").write_text("two")
            (root / "a").write_text("one")
            first = root / "first.tar"
            first_result = MODULE._snapshot(root, first)
            first.unlink()
            second = root / "second.tar"
            second_result = MODULE._snapshot(root, second)
            self.assertEqual(first_result, second_result)

    def test_snapshot_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "file").write_text("x")
            (root / "link").symlink_to("file")
            with self.assertRaisesRegex(ValueError, "symlink"):
                MODULE._snapshot(root, root.parent / "snapshot.tar")

    def test_target_must_remain_relative(self):
        for target in ("", "/tmp/x", "../x", "a/../../x", "."):
            with self.subTest(target=target), self.assertRaises(ValueError):
                MODULE._safe_target(target)
        self.assertEqual(str(MODULE._safe_target("src/main.rs")), "src/main.rs")


if __name__ == "__main__":
    unittest.main()
