import unittest
from src.exceptions.exceptions import PathError, ClassifierOptionError


class TestExceptions(unittest.TestCase):
    """Class to test exceptions raised by application."""

    def test_PathError_init(self):
        """Method to test PathError Exception initialization."""
        e = PathError()
        self.assertEqual(e.message, "Provided path is invalid.")
        self.assertEqual(e.code, "DL_FILE_NOT_FOUND_001")
        del e

    def test_PathError_get_message(self):
        """Method to test PathError Exception get_message method."""
        e = PathError()
        self.assertEqual(e.get_message(), "Provided path is invalid.")
        del e

    def test_PathError_get_code(self):
        """Method to test PathError Exception get_code method."""
        e = PathError()
        self.assertEqual(e.get_code(), "DL_FILE_NOT_FOUND_001")
        del e

    def test_ClassifierOptionError_init(self):
        """Method to test ClassifierOptionError Exception initialization."""
        e = ClassifierOptionError()
        self.assertEqual(e.message, "Provided classifier option is invalid.")
        self.assertEqual(e.code, "CLASSIFIER_NOT_FOUND_002")
        del e

    def test_ClassifierOptionError_get_message(self):
        """Method to test ClassifierOptionError Exception get_message method."""
        e = ClassifierOptionError()
        self.assertEqual(e.get_message(), "Provided classifier option is invalid.")
        del e

    def test_ClassifierOptionError_get_code(self):
        """Method to test ClassifierOptionError Exception get_code method."""
        e = ClassifierOptionError()
        self.assertEqual(e.get_code(), "CLASSIFIER_NOT_FOUND_002")
        del e

if __name__ == "__main__":
    unittest.main()
