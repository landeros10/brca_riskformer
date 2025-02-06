import os
import logging
from datetime import datetime


def logger_setup(debug=False):
    """ Setus up loggin configuration. Applied Globally."""
    LOG_PATH = "./logs"
    os.makedirs(LOG_PATH, exist_ok=True)
    log_filename = f"log_{datetime.now().strftime('%m_%d_%y')}.log"

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        return

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_PATH, log_filename)),
            logging.StreamHandler()
        ]
    )