import os
import logging
from datetime import datetime

def setup_logger(name, log_dir='logs/', level=logging.DEBUG):
    """Function to set up a logger with timestamped and sequenced log files."""
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a timestamp and sequence number for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'application_{timestamp}.log')

    # Set up the file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)

    # Set the log format
    formatter = logging.Formatter('%(asctime)s:%(filename)s:%(lineno)d:%(name)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)

    # Create and configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Initialize the logger for the application
logging = setup_logger('logging')