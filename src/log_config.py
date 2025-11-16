import logging
import os
from datetime import datetime

def setup_logger():
    # Create logs directory
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create unique log filename
    log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    log_path = os.path.join(logs_dir, log_file)

    # Configure logging
    logging.basicConfig(
        filename=log_path,
        format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True  # Important: allows re-import without error
    )

    # Return a named logger
    return logging.getLogger("MyAppLogger")

# Initialize the logger
logger = setup_logger()