import os
import logging
import watchtower
from riskformer.utils.aws_utils import initialize_boto3_session

def setup_cloudwatch_handler(
        log_group: str,
        profile_name: str,
        region_name: str,
) -> watchtower.CloudWatchLogHandler:
    """
    Initializes and returns a CloudWatchLogHandler for the given log_group.
    
    Args:
        log_group (str): The name of the CloudWatch log group.
        profile_name (str): AWS profile name with permissions for CloudWatch Logs.
        region_name (str): AWS region where CloudWatch Logs will be created.

    Returns:
        watchtower.CloudWatchLogHandler or None
    """
    session = initialize_boto3_session(profile_name, region_name)
    if not session:
        logging.getLogger(__name__).error(
            "Failed to initialize AWS session. CloudWatch logging will not be enabled."
        )
        return None
    try:
        client = session.client("logs")
        cw_handler = watchtower.CloudWatchLogHandler(
            log_group=log_group,
            boto3_client = client,
            use_queues=True,
            send_interval=5,
            stream_name=f"riskformer-{os.getenv('USER', 'unknown')}-{os.getenv('HOSTNAME', 'unknown')}-{log_group}"
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create CloudWatch handler: {e}")
        return None
    return cw_handler


def logger_setup(
        log_group: str = "riskformer",
        debug: bool = False,
        use_cloudwatch: bool = False,
        profile_name: str = None,
        region_name: str = None,
) -> None:
    """
    Sets up Python logging. If `use_cloudwatch` is True and AWS parameters 
    are provided, also logs to AWS CloudWatch.

    Args:
        log_group (str): Name of the log group in CloudWatch (if enabled).
        debug (bool): Whether to enable debug-level logging.
        use_cloudwatch (bool): Whether to enable CloudWatch logging.
        profile_name (str): AWS profile name for CloudWatch logging.
        region_name (str): AWS region for CloudWatch logging.
    """    
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
        root_logger.info("Setting up debugging for 'riskformer' modules.")
        logging.getLogger("riskformer").setLevel(logging.DEBUG)
        logging.getLogger("entrypoints").setLevel(logging.DEBUG)
        root_logger.debug("Debugging enabled for 'src' modules.")
    
    if use_cloudwatch:
        if not profile_name or not region_name:
            root_logger.error(
                "CloudWatch logging requested, but 'profile_name' or 'region_name' not provided. "
                "Falling back to local logs only."
            )
            raise ValueError("'profile_name' and 'region_name' must be provided for CloudWatch logging.")
        try:
            cw_handler = setup_cloudwatch_handler(log_group, profile_name, region_name)
        except Exception as e:
            root_logger.error(f"Unexpected error setting up CloudWatch handler: {e}")
            raise e
        if cw_handler:
            root_logger.addHandler(cw_handler)
            root_logger.info(
                f"CloudWatch logging enabled. Log Group: '{log_group}', Stream: '{cw_handler.log_stream_name}'"
            )
        else:
            raise ValueError("Failed to initialize CloudWatch handler; check AWS credentials and permissions.")


def log_config(logger, config, tag):
    """ Logs the configuration parameters.
    
    """
    if not isinstance(config, dict):
        try:
            config = config.dict()
        except Exception as e:
            logger.warning(f"Failed to load config dict. Error: {e}")
            raise e

    logger.info(f"[{tag} configuration]")
    for key, value in config.items():
        logger.info(f"\t{key}:\t{value}")


def log_event(level, event, status, **kwargs):
    log_data = {"event": event, "status": status, "timestamp": time.time(), **kwargs}
    
    # Human-readable log for local debugging
    log_text = f"[{event}] {status} | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())

    if level == "info":
        logger.info(log_text)
        logger.info(json.dumps(log_data))  # JSON for CloudWatch
    elif level == "debug":
        logger.debug(log_text)
        logger.debug(json.dumps(log_data))
    elif level == "warning":
        logger.warning(log_text)
        logger.warning(json.dumps(log_data))
    elif level == "error":
        logger.error(log_text)
        logger.error(json.dumps(log_data))