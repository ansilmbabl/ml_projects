# Import the sys module to access system-related functionality
import sys

# Define a function to create a detailed error message including filename and line number
def error_message_details(error):
    # Get information about the current exception using the sys.exc_info() function
    _, _, exc_tb = sys.exc_info()
    
    # Extract the filename and line number from the exception traceback
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error raised in python script name {file_name} line number {exc_tb.tb_lineno} error message {str(error)}"
    return error_message

# Define a custom exception class that inherits from the built-in Exception class
class CustomException(Exception):
    def __init__(self, error_message):
        # Call the constructor of the parent Exception class to initialize the exception
        super().__init__(error_message)
        
        # Generate a detailed error message using the error_message_details function
        self.error_message = error_message_details(error_message)
       
    def __str__(self) -> str:
        # Return the custom error message when the exception is converted to a string
        return self.error_message
