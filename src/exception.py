import sys
from typing import Any


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error information including filename and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    
    # Safely extract traceback info
    file_name = "Unknown"
    line_number = "Unknown"
    
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in Python script: "
        f"[{file_name}] at line [{line_number}]\n"
        f"Error Message: {str(error)}"
    )
    return error_message


class CustomException(Exception):
    """
    Custom exception with detailed error context (file, line, message).
    Usage:
        try:
            1/0
        except Exception as e:
            raise CustomException(e, sys)
    """

    def __init__(self, error: Exception, error_detail: sys):
        self.error_message_detail = error_message_detail(error, error_detail)
        super().__init__(self.error_message_detail)

    def __str__(self) -> str:
        return self.error_message_detail