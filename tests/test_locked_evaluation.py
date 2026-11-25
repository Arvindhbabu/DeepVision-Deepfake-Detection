"""Smoke tests for locked evaluation import and metrics."""
import unittest
from pathlib import Path
from scripts.evaluate_locked_test import sha256_file


class TestLockedEvaluation(unittest.TestCase):
    def test_hash_changes_with_file_bytes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample'
            path.write_bytes(b'first')
            first = sha256_file(path)
            path.write_bytes(b'second')
            self.assertNotEqual(first, sha256_file(path))


if __name__ == '__main__':
    unittest.main()
