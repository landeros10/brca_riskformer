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
        
        # Add default values for required fields if not present
        _add_defaults_to_config(config_dict)
        
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


def _add_defaults_to_config(config: Dict[str, Any]) -> None:
    """
    Add default values to the config if they're not present.
    This ensures that all required fields have reasonable defaults.
    
    Args:
        config: Configuration dictionary to modify
    """
    # Default values for common fields
    defaults = {
        # Optimizer defaults
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "scheduler": "plateau",
        "regional_coeff": 0.0,
        
        # Training defaults
        "batch_size": 32,
        "num_workers": 4,
        "max_epochs": 100,
        "min_epochs": 10,
        "patience": 10,
        "accumulate_grad_batches": 1,
        
        # Data defaults
        "val_split": 0.2,
        "test_split": 0.1,
        "pin_memory": True,
        "s3_prefix": "",
        "max_dim": 32,
        "overlap": 0.0,
        "seed": 42,
        "cache_dir": "/tmp/riskformer_cache", 
        
        # RiskFormer model defaults if not specified
        "use_phi": True,
        "drop_path_rate": 0.1,
        "drop_rate": 0.1,
        "use_attn_mask": False,
        "use_class_token": True,
        "encoding_method": "sinusoidal",
        
        # Required model parameters - these should be set by the user,
        # but providing sensible defaults helps in development
        "input_embed_dim": 1024,
        "output_embed_dim": 512,
        "depth": 4,
        "global_depth": 2,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "attn_global_hidden_dim": 128
    }
    
    # Apply defaults where values are missing
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            logger.info(f"Added default value for '{key}': {default_value}")
    
    # Make sure tasks is defined, even if empty
    if "tasks" not in config:
        config["tasks"] = {}
        
    # Apply defaults to each task
    for task_name, task_config in config.get("tasks", {}).items():
        # Default task configuration
        task_defaults = {
            "weight": 1.0,
            "metrics": ["accuracy"] if task_config.get("type") in ["binary", "multiclass"] else ["mse", "mae"]
        }
        
        # Apply defaults where values are missing in tasks
        for key, default_value in task_defaults.items():
            if key not in task_config:
                task_config[key] = default_value
                logger.info(f"Added default value for task '{task_name}.{key}': {default_value}")


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
    
    # Validate s3_bucket is a non-empty string
    if not isinstance(config["s3_bucket"], str) or not config["s3_bucket"].strip():
        raise ValueError("'s3_bucket' must be a non-empty string")
        
    # Validate s3_prefix if present
    if "s3_prefix" in config and not isinstance(config["s3_prefix"], str):
        raise ValueError("'s3_prefix' must be a string")
    
    # Validate cache_dir if present
    if "cache_dir" in config:
        if not isinstance(config["cache_dir"], str):
            raise ValueError("'cache_dir' must be a string")
        if not config["cache_dir"].strip():
            raise ValueError("'cache_dir' cannot be empty")
    
    # Validate AWS-related fields if present
    if "profile_name" in config and not isinstance(config["profile_name"], str):
        raise ValueError("'profile_name' must be a string")
        
    if "region_name" in config and not isinstance(config["region_name"], str):
        raise ValueError("'region_name' must be a string")
    
    # Required fields for RiskFormerLightningModule model initialization
    model_required_fields = [
        "input_embed_dim",
        "output_embed_dim",
        "depth",
        "global_depth",
        "num_heads",
        "mlp_ratio",
        "regional_coeff"  # Required for loss calculation
    ]
    
    for field in model_required_fields:
        if field not in config:
            raise ValueError(f"Required model field '{field}' missing from config")
            
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
        task_required_fields = ["type", "num_classes", "loss_fn", "weight", "metrics"]
        for field in task_required_fields:
            if field not in task_config:
                raise ValueError(f"Task '{task_name}' missing required field '{field}'")
        
        task_type = task_config["type"]
        if task_type not in ["binary", "multiclass", "regression"]:
            raise ValueError(f"Task '{task_name}' has invalid type '{task_type}', must be one of: binary, multiclass, regression")
        
        # Validate metrics is a list
        if not isinstance(task_config["metrics"], list):
            raise ValueError(f"Task '{task_name}' field 'metrics' must be a list")
        
        # Validate numeric fields in task config
        if not isinstance(task_config["num_classes"], int) or task_config["num_classes"] <= 0:
            raise ValueError(f"Task '{task_name}' field 'num_classes' must be a positive integer")
        
        if not isinstance(task_config["weight"], (int, float)) or task_config["weight"] <= 0:
            raise ValueError(f"Task '{task_name}' field 'weight' must be a positive number")
        
        # Validate loss function
        valid_loss_functions = ["MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "L1Loss"]
        if task_config["loss_fn"] not in valid_loss_functions:
            raise ValueError(f"Task '{task_name}' has invalid loss_fn '{task_config['loss_fn']}', must be one of: {', '.join(valid_loss_functions)}")
    
    # Optional numeric fields
    numeric_fields = [
        "batch_size", "num_workers", "max_epochs", "min_epochs",
        "val_split", "test_split", "seed", "max_dim", "overlap",
        "learning_rate", "weight_decay", "patience", "accumulate_grad_batches",
        "input_embed_dim", "output_embed_dim", "drop_path_rate", "drop_rate",
        "depth", "global_depth", "num_heads", "mlp_ratio", "regional_coeff",
        "downscale_stride_q", "downscale_stride_k", "downscale_multiplier",
        "attn_global_hidden_dim"
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
    
    if "max_dim" in config and config["max_dim"] <= 0:
        raise ValueError("'max_dim' must be positive")
        
    if "overlap" in config and (config["overlap"] < 0 or config["overlap"] >= 1):
        raise ValueError("'overlap' must be between 0 and 1 (exclusive)")
    
    if "input_embed_dim" in config and config["input_embed_dim"] <= 0:
        raise ValueError("'input_embed_dim' must be positive")
        
    if "output_embed_dim" in config and config["output_embed_dim"] <= 0:
        raise ValueError("'output_embed_dim' must be positive")
    
    # Validate boolean fields
    boolean_fields = [
        "pin_memory", "use_phi", "use_attn_mask", "use_class_token", "use_wandb",
        "hflip_prob", "vflip_prob", "rotate_prob", "noise_aug_prob"
    ]
    for field in boolean_fields:
        if field in config and not isinstance(config[field], bool):
            raise ValueError(f"Field '{field}' must be a boolean")
    
    # Validate optimizer and scheduler
    if "optimizer" in config and config["optimizer"] not in ["adam", "adamw", "sgd"]:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    if "scheduler" in config and config["scheduler"] not in ["plateau", "cosine", "linear", "step", "none", "onecycle"]:
        raise ValueError(f"Unsupported scheduler: {config['scheduler']}")
    
    # Validate encoding method
    if "encoding_method" in config and config["encoding_method"] not in ["sinusoidal", "learned"]:
        raise ValueError(f"Unsupported encoding_method: {config['encoding_method']}")
    
    # Validate precision field
    if "precision" in config and config["precision"] not in ["32", "16", "bf16"]:
        raise ValueError(f"Unsupported precision: {config['precision']}")
    
