"""Logger cleanup regression tests."""

import logging
import tempfile
import unittest

from src.utils.logger import create_logger


class TestLoggerCleanup(unittest.TestCase):
    def test_previous_file_handler_closed_on_reconfigure(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = create_logger(log_dir=folder, log_name="old.log")
            previous = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
            self.assertIsNotNone(previous.stream)

            logger = create_logger(log_dir=folder, log_name="new.log")
            self.assertIsNone(previous.stream)
            self.assertEqual(
                sum(isinstance(h, logging.FileHandler) for h in logger.handlers), 1
            )
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()


if __name__ == "__main__":
    unittest.main()
