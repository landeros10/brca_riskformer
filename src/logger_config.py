import os
import logging
from datetime import datetime


def logger_setup():
    LOG_PATH = "./logs"
    os.makedirs(LOG_PATH, exist_ok=True)
    log_filename = f"log_{datetime.now().strftime('%m_%d_%y')}.log"

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_PATH, log_filename)),
            logging.StreamHandler()
        ]
    )