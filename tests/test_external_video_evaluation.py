"""Independent manifest validation and metrics tests, with no model downloads."""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_external_videos import load_manifest, summarize_predictions


class TestExternalEvaluation(unittest.TestCase):
    def test_metrics_expose_real_false_positives(self):
        rows = [
            {"status": "ok", "label": 0, "prediction": 1},
            {"status": "ok", "label": 0, "prediction": 0},
            {"status": "ok", "label": 1, "prediction": 1},
            {"status": "error", "label": 1},
        ]
        m = summarize_predictions(rows)
        self.assertEqual(m["evaluated"], 3)
        self.assertEqual(m["error_count"], 1)
        self.assertEqual(m["confusion_matrix_real_fake"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(m["real_false_positive_rate"], 0.5)

    def test_empty_predictions(self):
        result = summarize_predictions([{"status": "error"}])
        self.assertEqual(result["evaluated"], 0)

    def test_manifest_missing_group_id(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "clip.mp4").write_bytes(b"test")
            manifest = folder / "manifest.csv"
            manifest.write_text("video_path,label,group_id\nclip.mp4,Real,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "group_id"):
                load_manifest(manifest)

    def test_manifest_valid(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "clip.mp4").write_bytes(b"test")
            manifest = folder / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["video_path", "label", "group_id"])
                writer.writeheader()
                writer.writerow({"video_path": "clip.mp4", "label": "Real", "group_id": "actor_a"})
            result = load_manifest(manifest)
            self.assertEqual(result[0]["label"], 0)
            self.assertEqual(result[0]["group_id"], "actor_a")

    def test_manifest_duplicate_source_path(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "clip.mp4").write_bytes(b"test")
            manifest = folder / "manifest.csv"
            manifest.write_text(
                "video_path,label,group_id\nclip.mp4,Real,one\nclip.mp4,Fake,two\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
