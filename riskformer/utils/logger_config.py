import os
import logging
from datetime import datetime

def logger_setup(log_group="general", debug=False):
    """Configures logging to apply DEBUG only to 'src.*' modules.
    Keeps external libraries at default levels."""
    
    LOG_PATH = "/opt/ml/processing/logs"
    os.makedirs(LOG_PATH, exist_ok=True)
    log_filename = f"{log_group}.log"

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
        root_logger.info("Setting up debugging for 'src' modules.")
        logging.getLogger("riskformer").setLevel(logging.DEBUG)
        logging.getLogger("entrypoints").setLevel(logging.DEBUG)
        root_logger.debug("Debugging enabled for 'src' modules.")


def log_config(logger, config, tag):
    """ Logs the configuration parameters.
    
    """
    if not isinstance(config, dict):
        try:
            config = config.dict()
        except Exception as e:
            logger.warning(f"Failed to load config dict. Error: {e}")
            raise e

    logger.info(f"{tag} configuration:" + "=" * 20)
    for key, value in config.items():
        logger.info(f"{key}: {value}")
