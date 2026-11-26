"""Lightweight Stage 7 upload validation tests."""
import unittest

from app.streamlit_app import check_upload


class TestUploadValidation(unittest.TestCase):
    def test_valid_mp4(self):
        check_upload("demo.mp4", 1024)

    def test_valid_case_insensitive_extension(self):
        check_upload("DEMO.MOV", 1024)

    def test_reject_executable(self):
        with self.assertRaisesRegex(ValueError, "MP4"):
            check_upload("demo.exe", 1024)

    def test_reject_large_file(self):
        with self.assertRaisesRegex(ValueError, "between"):
            check_upload("demo.mp4", 100 * 1024 * 1024 + 1)

    def test_reject_empty(self):
        with self.assertRaisesRegex(ValueError, "between"):
            check_upload("demo.mp4", 0)


if __name__ == "__main__":
    unittest.main()
