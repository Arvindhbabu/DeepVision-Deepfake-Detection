import unittest
import logging
from src.utils.logger import create_logger


class TestLogger(unittest.TestCase):

    def test_create_logger(self):
        logger = create_logger()
        self.assertIsInstance(logger, logging.Logger)
        logger.info("Test log message")


if __name__ == "__main__":
    unittest.main()