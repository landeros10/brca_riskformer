import os
import yaml
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class AWSConfig:
    profile: str
    region: str
    ecr_id: str
    credentials_path: str

@dataclass
class S3Config:
    model_bucket: str
    data_bucket: str
    input_dir: str
    output_dir: str

@dataclass
class DockerConfig:
    image_name: str
    workspace_root: str
    runtime: str
    user: str
    memory: str
    cpus: str
    capabilities: List[str]
    devices: List[str]

@dataclass
class ProjectDirectories:
    resources: str
    configs: str
    outputs: str
    riskformer: str
    entrypoints: str
    orchestrators: str
    logs: str

@dataclass
class ProjectConfig:
    root: str
    directories: ProjectDirectories

@dataclass
class ConfigFiles:
    metadata: str
    foreground: str
    foreground_cleanup: str
    tiling: str

@dataclass
class ModelConfig:
    type: str
    key: str

@dataclass
class ProcessingConfig:
    batch_size: int
    num_workers: int
    prefetch_factor: int
    stop_on_fail: bool
    use_cloudwatch: bool
    debug: bool

@dataclass
class PreprocessingConfig:
    aws: AWSConfig
    s3: S3Config
    docker: DockerConfig
    project: ProjectConfig
    config_files: ConfigFiles
    model: ModelConfig
    processing: ProcessingConfig


def _dataclass_to_dict(obj: Any) -> Dict:
    """Convert a dataclass instance to a nested dictionary."""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            result[field] = _dataclass_to_dict(value) if hasattr(value, '__dataclass_fields__') else value
        return result
    return obj


def load_preprocessing_config(config_path: str) -> Dict:
    """Load preprocessing configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
        
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        # Convert nested dictionaries to appropriate dataclass objects
        aws_config = AWSConfig(**config_dict["aws"])
        s3_config = S3Config(**config_dict["s3"])
        docker_config = DockerConfig(**config_dict["docker"])
        project_dirs = ProjectDirectories(**config_dict["project"]["directories"])
        project_config = ProjectConfig(root=config_dict["project"]["root"], directories=project_dirs)
        config_files = ConfigFiles(**config_dict["config_files"])
        model_config = ModelConfig(**config_dict["model"])
        processing_config = ProcessingConfig(**config_dict["processing"])
        
        config = PreprocessingConfig(
            aws=aws_config,
            s3=s3_config,
            docker=docker_config,
            project=project_config,
            config_files=config_files,
            model=model_config,
            processing=processing_config
        )
        
        # Convert the dataclass to a dictionary
        return _dataclass_to_dict(config)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise


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
    

def load_train_config(config_path: str) -> Dict:
    """Load training configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
        
    Raises:
        ValueError: If the configuration file is invalid
        FileNotFoundError: If the configuration file does not exist
    """
    # Validate the config file exists
    if not os.path.isfile(config_path):
        logger.error(f"Config file {config_path} not found")
        raise FileNotFoundError(f"Config file {config_path} not found")

    try:
        # Load the config
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Validate the config
        _validate_training_config(config_dict)
            
        logger.info(f"Successfully loaded and validated config from {config_path}")
        return config_dict
    except ValueError as e:
        # Re-raise validation errors
        logger.error(f"Invalid config file: {str(e)}")
        raise
    except Exception as e:
        # Handle other errors like YAML parsing issues
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        raise ValueError(f"Failed to load config: {str(e)}")
    

def _validate_training_config(config: Dict[str, Any]) -> None:
    """
    Validate a training configuration dictionary.
    
    Args:
        config: Training configuration dictionary to validate
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Verify it's a dictionary
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config).__name__}")
    
    # Required fields for training
    required_fields = [
        "s3_bucket",
        "metadata_file",
        "tasks"
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from config")
    
    # Validate tasks field
    if not isinstance(config["tasks"], dict):
        raise ValueError("'tasks' must be a dictionary")
    
    if not config["tasks"]:
        raise ValueError("'tasks' cannot be empty")
    
    # Validate each task
    for task_name, task_config in config["tasks"].items():
        if not isinstance(task_config, dict):
            raise ValueError(f"Task '{task_name}' must be a dictionary")
        
        # Required task fields
        if "type" not in task_config:
            raise ValueError(f"Task '{task_name}' missing required field 'type'")
        
        task_type = task_config["type"]
        if task_type not in ["binary", "multiclass", "regression"]:
            raise ValueError(f"Task '{task_name}' has invalid type '{task_type}', must be one of: binary, multiclass, regression")
    
    # Optional numeric fields
    numeric_fields = [
        "batch_size", "num_workers", "max_epochs", "min_epochs",
        "val_split", "test_split", "seed", "max_dim", "overlap"
    ]
    
    for field in numeric_fields:
        if field in config and not isinstance(config[field], (int, float)):
            raise ValueError(f"Field '{field}' must be a number")
    
    # Validate specific numeric constraints
    if "batch_size" in config and config["batch_size"] <= 0:
        raise ValueError("'batch_size' must be positive")
    
    if "num_workers" in config and config["num_workers"] < 0:
        raise ValueError("'num_workers' cannot be negative")
    
    if "val_split" in config and (config["val_split"] < 0 or config["val_split"] >= 1):
        raise ValueError("'val_split' must be between 0 and 1 (exclusive)")
    
    if "test_split" in config and (config["test_split"] < 0 or config["test_split"] >= 1):
        raise ValueError("'test_split' must be between 0 and 1 (exclusive)")
    
