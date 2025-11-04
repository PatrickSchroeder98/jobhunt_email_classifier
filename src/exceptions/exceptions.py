class PathError(Exception):
    """Exception raised for errors when path is invalid."""

    def __init__(self, message="Provided path is invalid.", code="DL_FILE_NOT_FOUND_001"):
        """Setting custom exception message."""
        self.message = message
        self.code = code
        super().__init__(self.message, self.code)

    def get_message(self):
        """Gets the exception message."""
        return self.message

    def get_code(self):
        """Gets the exception code."""
        return self.code

class ClassifierOptionError(Exception):
    """Exception raised for errors when classifier option is invalid."""

    def __init__(self, message="Provided classifier option is invalid.", code="CLASSIFIER_NOT_FOUND_002"):
        """Setting custom exception message."""
        self.message = message
        self.code = code
        super().__init__(self.message, self.code)

    def get_message(self):
        """Gets the exception message."""
        return self.message

    def get_code(self):
        """Gets the exception code."""
        return self.code

class ModelNotFound(Exception):
    """Exception raised for errors when model is not properly initialized."""

    def __init__(self, message="Model has not been initialized.", code="MODEL_NOT_FOUND_003"):
        """Setting custom exception message."""
        self.message = message
        self.code = code
        super().__init__(self.message, self.code)

    def get_message(self):
        """Gets the exception message."""
        return self.message

    def get_code(self):
        """Gets the exception code."""
        return self.code

class InputDataError(Exception):
    """Exception raised for errors when input data is invalid."""

    def __init__(self, message="Input data is not a string or list of strings.", code="INVALID_DATA_004"):
        """Setting custom exception message."""
        self.message = message
        self.code = code
        super().__init__(self.message, self.code)

    def get_message(self):
        """Gets the exception message."""
        return self.message

    def get_code(self):
        """Gets the exception code."""
        return self.code

class VotingOptionForVotingClassifierError(Exception):
    """Exception raised for errors when voting option is invalid."""

    def __init__(self, message="Voting option for VotingClassifier is invalid.", code="INVALID_VOTING_OPTION_005"):
        """Setting custom exception message."""
        self.message = message
        self.code = code
        super().__init__(self.message, self.code)

    def get_message(self):
        """Gets the exception message."""
        return self.message

    def get_code(self):
        """Gets the exception code."""
        return self.code

class VotingClassifierNotSupported(Exception):
    """Exception raised for errors when voting classifier is not supported."""

    def __init__(self, message="VotingClassifier is not supported in this environment.", code="VOTING_CLASSIFIER_NOT_SUPPORTED_006"):
        """Setting custom exception message."""
        self.message = message
        self.code = code
        super().__init__(self.message, self.code)

    def get_message(self):
        """Gets the exception message."""
        return self.message

    def get_code(self):
        """Gets the exception code."""
        return self.code

class EstimatorOptionError(Exception):
    """Exception raised for errors when estimator option is invalid."""

    def __init__(self, message="Provided estimator option is invalid. Default estimators will be used instead.", code="ESTIMATOR_NOT_FOUND_007"):
        """Setting custom exception message."""
        self.message = message
        self.code = code
        super().__init__(self.message, self.code)

    def get_message(self):
        """Gets the exception message."""
        return self.message

    def get_code(self):
        """Gets the exception code."""
        return self.code