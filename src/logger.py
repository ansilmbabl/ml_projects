import logging
import os
from datetime import datetime

# Generate a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Create a path to the 'logs' directory within the current working directory
logs_dir = os.path.join(os.getcwd(), "logs")

# Create the 'logs' directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# Create the full path to the log file within the 'logs' directory
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging with the specified settings
logging.basicConfig(
    filename=LOG_FILE_PATH,                  # Use the full path to the log file
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,                      # Log messages at the INFO level and higher
)
