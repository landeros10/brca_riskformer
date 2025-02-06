import os
import logging
from datetime import datetime

def logger_setup(debug=False):
    """Configures logging to apply DEBUG only to 'src.*' modules.
    Keeps external libraries at default levels."""
    
    LOG_PATH = "./logs"
    os.makedirs(LOG_PATH, exist_ok=True)
    log_filename = f"log_{datetime.now().strftime('%m_%d_%y')}.log"

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.info("Logger already configured, skipping setup.")
        return
    


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_PATH, log_filename)),
            logging.StreamHandler()
        ]
    )

    if debug:
        logging.getLogger("src").setLevel(logging.DEBUG)
        root_logger.debug("Debugging enabled for 'src' modules.")
