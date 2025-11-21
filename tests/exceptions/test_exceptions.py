import unittest
from src.exceptions.exceptions import (
    PathError,
    ClassifierOptionError,
    ModelNotFound,
    InputDataError,
    VotingOptionForVotingClassifierError,
    VotingClassifierNotSupported,
    EstimatorOptionError,
    StackingClassifierNotSupported,
)


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

    def test_ModelNotFound_init(self):
        """Method to test ModelNotFound Exception initialization."""
        e = ModelNotFound()
        self.assertEqual(e.message, "Model has not been initialized.")
        self.assertEqual(e.code, "MODEL_NOT_FOUND_003")
        del e

    def test_ModelNotFound_get_message(self):
        """Method to test ModelNotFound Exception get_message method."""
        e = ModelNotFound()
        self.assertEqual(e.get_message(), "Model has not been initialized.")
        del e

    def test_ModelNotFound_get_code(self):
        """Method to test ModelNotFound Exception get_code method."""
        e = ModelNotFound()
        self.assertEqual(e.get_code(), "MODEL_NOT_FOUND_003")
        del e

    def test_InputDataError_init(self):
        """Method to test InputDataError Exception initialization."""
        e = InputDataError()
        self.assertEqual(e.message, "Input data is not a string or list of strings.")
        self.assertEqual(e.code, "INVALID_DATA_004")
        del e

    def test_InputDataError_get_message(self):
        """Method to test InputDataError Exception get_message method."""
        e = InputDataError()
        self.assertEqual(
            e.get_message(), "Input data is not a string or list of strings."
        )
        del e

    def test_InputDataError_get_code(self):
        """Method to test InputDataError Exception get_code method."""
        e = InputDataError()
        self.assertEqual(e.get_code(), "INVALID_DATA_004")
        del e

    def test_VotingOptionForVotingClassifierError_init(self):
        """Method to test VotingOptionForVotingClassifierError Exception initialization."""
        e = VotingOptionForVotingClassifierError()
        self.assertEqual(e.message, "Voting option for VotingClassifier is invalid.")
        self.assertEqual(e.code, "INVALID_VOTING_OPTION_005")
        del e

    def test_VotingOptionForVotingClassifierError_get_message(self):
        """Method to test VotingOptionForVotingClassifierError Exception get_message method."""
        e = VotingOptionForVotingClassifierError()
        self.assertEqual(
            e.get_message(), "Voting option for VotingClassifier is invalid."
        )
        del e

    def test_VotingOptionForVotingClassifierError_get_code(self):
        """Method to test VotingOptionForVotingClassifierError Exception get_code method."""
        e = VotingOptionForVotingClassifierError()
        self.assertEqual(e.get_code(), "INVALID_VOTING_OPTION_005")
        del e

    def test_VotingClassifierNotSupported_init(self):
        """Method to test VotingClassifierNotSupported Exception initialization."""
        e = VotingClassifierNotSupported()
        self.assertEqual(
            e.message, "VotingClassifier is not supported in this environment."
        )
        self.assertEqual(e.code, "VOTING_CLASSIFIER_NOT_SUPPORTED_006")
        del e

    def test_VotingClassifierNotSupported_get_message(self):
        """Method to test VotingClassifierNotSupported Exception get_message method."""
        e = VotingClassifierNotSupported()
        self.assertEqual(
            e.get_message(), "VotingClassifier is not supported in this environment."
        )
        del e

    def test_VotingClassifierNotSupported_get_code(self):
        """Method to test VotingClassifierNotSupported Exception get_code method."""
        e = VotingClassifierNotSupported()
        self.assertEqual(e.get_code(), "VOTING_CLASSIFIER_NOT_SUPPORTED_006")
        del e

    def test_EstimatorOptionError_init(self):
        """Method to test EstimatorOptionError Exception initialization."""
        e = EstimatorOptionError()
        self.assertEqual(
            e.message,
            "Provided estimator option is invalid. Default estimators will be used instead.",
        )
        self.assertEqual(e.code, "ESTIMATOR_NOT_FOUND_007")
        del e

    def test_EstimatorOptionError_get_message(self):
        """Method to test EstimatorOptionError Exception get_message method."""
        e = EstimatorOptionError()
        self.assertEqual(
            e.get_message(),
            "Provided estimator option is invalid. Default estimators will be used instead.",
        )
        del e

    def test_EstimatorOptionError_get_code(self):
        """Method to test EstimatorOptionError Exception get_code method."""
        e = EstimatorOptionError()
        self.assertEqual(e.get_code(), "ESTIMATOR_NOT_FOUND_007")
        del e

    def test_StackingClassifierNotSupported_init(self):
        """Method to test StackingClassifierNotSupported Exception initialization."""
        e = StackingClassifierNotSupported()
        self.assertEqual(e.message, "StackingClassifier is not supported in this environment.")
        self.assertEqual(e.code, "STACKING_CLASSIFIER_NOT_SUPPORTED_006")
        del e

    def test_StackingClassifierNotSupported_get_message(self):
        """Method to test StackingClassifierNotSupported Exception get_message method."""
        e = StackingClassifierNotSupported()
        self.assertEqual(e.get_message(), "StackingClassifier is not supported in this environment.")
        del e

    def test_StackingClassifierNotSupported_get_code(self):
        """Method to test StackingClassifierNotSupported Exception get_code method."""
        e = StackingClassifierNotSupported()
        self.assertEqual(e.get_code(), "STACKING_CLASSIFIER_NOT_SUPPORTED_006")
        del e


if __name__ == "__main__":
    unittest.main()
