"""Set up logging for the VarClus class."""

import logging

__all__ = ["logger"]


LOG_LEVEL = logging.DEBUG
LOG_FILE_MODE = "w"

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# Create handlers
file_handler = logging.FileHandler("varclus.log", mode=LOG_FILE_MODE)
file_handler.setLevel(LOG_LEVEL)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOG_LEVEL)

# Create formatters and add them to handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
