# Create a custom exception class for the project.
class ProjectException(Exception):
    # Define the constructor.
    def __init__(self, message: str):
        # Pass the message to the parent Exception class.
        super().__init__(message)

        # Save the message in the object.
        self.message = message

    # Define how the exception is displayed as text.
    def __str__(self) -> str:
        # Return the custom error message.
        return self.message