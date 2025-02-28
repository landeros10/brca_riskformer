import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_yaml_config(config_path, schema):
    """Load a YAML config file and validate it against a schema."""
    
    config_path = os.path.abspath(config_path)
    if not config_path:
        logger.warning(f"Config file {config_path} not given. Using defaults.")
        return schema()
    
    if not os.path.isfile(config_path):
        logger.warning(f"Config file {config_path} not valid. Using defaults.")
        return schema()

    try:
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            logger.debug(f"Successfully loaded YAML config from {config_path}")
            if not isinstance(yaml_config, dict):
                logger.warning(f"Invalid YAML format in {config_path}. Using defaults.")
                return schema()
    except Exception as e:
        logger.warning(f"Failed to load YAML config {config_path}. Error: {e}. Using defaults.")
        return schema()
    try:
        return schema(**yaml_config)
    except Exception as e:
        logger.warning(f"Invalid values in {config_path}. Using defaults. Error: {e}")
        return schema()
    
