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