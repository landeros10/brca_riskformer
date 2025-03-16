"""
This module is deprecated. All configuration utilities have been moved to preprocess_utils.py.
This file is kept for backward compatibility and will be removed in a future release.
"""

import logging
import warnings
from riskformer.utils.preprocess_utils import (
    # Configuration dataclasses
    AWSConfig,
    S3Config,
    DockerConfig,
    ProjectDirectories,
    ProjectConfig,
    ConfigFiles,
    ModelConfig,
    ProcessingConfig,
    PreprocessingConfig,
    # Utility functions
    _dataclass_to_dict,
    load_preprocessing_config,
    load_yaml_config,
)

logger = logging.getLogger(__name__)
warnings.warn(
    "The config_utils module is deprecated. Use preprocess_utils instead.",
    DeprecationWarning,
    stacklevel=2
)
    