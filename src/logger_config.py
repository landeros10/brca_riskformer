import os
import logging
from datetime import datetime

def logger_setup(debug=False):
    """Sets up logging globally and ensures it is only configured once."""
    LOG_PATH = "./logs"
    os.makedirs(LOG_PATH, exist_ok=True)
    log_filename = f"log_{datetime.now().strftime('%m_%d_%y')}.log"

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.info("Logger already configured, skipping setup.")
        return

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_PATH, log_filename)),
            logging.StreamHandler()
        ]
    )

    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.debug("Logger setup complete.")
