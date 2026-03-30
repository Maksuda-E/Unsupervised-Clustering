# This class creates a custom exception for the project
class ProjectException(Exception):
    # This function initializes the custom exception
    def __init__(self, message: str):
        # This line calls the parent Exception class
        super().__init__(message)

        # This line stores the error message
        self.message = message