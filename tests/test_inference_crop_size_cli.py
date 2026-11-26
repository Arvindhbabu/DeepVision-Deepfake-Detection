"""Stage 6 CLI regression: crop-size is selectable, previous default preserved."""
import subprocess
import sys
import unittest
from pathlib import Path


class TestInferenceCropSizeCLI(unittest.TestCase):
    def test_help_exposes_explicit_face_size(self):
        script = Path(__file__).resolve().parents[1] / "scripts/predict_cached_bilstm.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--face-size", result.stdout)
        self.assertIn("260", result.stdout)


if __name__ == "__main__":
    unittest.main()
