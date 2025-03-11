from abc import abstractmethod
import logging
from datetime import datetime, timedelta
from os.path import join, abspath, exists
from os import makedirs

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Any, Optional, Union, List, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.optim import Adam, SGD, AdamW
from torchvision import transforms
import torch.nn.functional as F
import yaml

from riskformer.training.layers import SinusoidalPositionalEncoding2D, MultiScaleBlock, GlobalMaxPoolLayer
from riskformer.utils.training_utils import create_slide_level_loss

logger = logging.getLogger(__name__)

class RiskFormer_Head(nn.Module):
    """ Multi-task head for RiskFormer with task-specific architectures. """
    def __init__(self, tasks_config, input_dim, drop_rate, hidden_dim=None):
        """
        Initialize task-specific heads based on task configurations.
        
        Args:
            tasks_config: Dictionary of task configurations with keys as task names and values as dicts:
                - type: Task type ("binary", "regression", "multiclass")
                - num_classes: Number of output classes/values
            input_dim: Dimension of input features
            drop_rate: Dropout rate
            hidden_dim: Optional hidden dimension for more complex heads
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim // 2
        self.drop_rate = drop_rate
        self.tasks_config = tasks_config
        
        # Create mapping from task name to index for output slicing
        self.task_indices = {}
        start_idx = 0
        for task_name, task_config in tasks_config.items():
            num_classes = task_config['num_classes']
            end_idx = start_idx + num_classes
            self.task_indices[task_name] = (start_idx, end_idx)
            start_idx = end_idx
        
        # Create task-specific heads
        self.heads = nn.ModuleList([
            self._initialize_head(task_config) 
            for _, task_config in tasks_config.items()
        ])
        
    def _head_activation(self, activation):
        if activation is None or activation == "None":
            return nn.Identity()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax":
            return nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _initialize_head(self, task_config):
        """
        Initialize task-specific head architecture.
        
        Args:
            task_config: Configuration for this task
            
        Returns:
            Task-specific neural network head
        """
        task_type = task_config['type']
        num_classes = task_config['num_classes']
        activation = task_config['activation']

        layers = [
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
        ]
        
        if task_type == "binary" or task_type == "regression":
            layers.append(nn.Linear(self.hidden_dim, 1))
        
        elif task_type == "multiclass":
            layers.append(nn.Linear(self.hidden_dim, num_classes))
        
        else:
            layers.append(nn.Linear(self.hidden_dim, num_classes))

        layers.append(self._head_activation(activation))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through all task heads.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Concatenated outputs from all heads
        """
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=-1)
    
    def get_task_output(self, x, task_name):
        """
        Get output for a specific task.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            task_name: Name of the task to get output for
            
        Returns:
            Output for the specified task
        """
        all_outputs = self.forward(x)
        start_idx, end_idx = self.task_indices[task_name]
        return all_outputs[:, start_idx:end_idx]


class RiskFormer_ViT(nn.Module):
    """
    Vision Transformer for Whole Slide Image processing with multiscale attention.
    
    Args:
        input_embed_dim: Input embedding dimension
        output_embed_dim: Output embedding dimension
        use_phi: Whether to use phi network
        drop_path_rate: Drop path rate
        drop_rate: Dropout rate
        tasks: Dictionary mapping task names to task configurations. Each task config should include:
            - type: Task type ("binary", "regression", "multiclass")
            - num_classes: Number of output classes/values
            - metrics: Optional list of metrics to track
            - weight: Optional task weight for loss calculation
        max_dim: Maximum dimension
        depth: Number of transformer layers
        global_depth: Number of global transformer layers
        encoding_method: Position encoding method
        num_heads: Number of attention heads
        use_attn_mask: Whether to use attention mask
        mlp_ratio: MLP ratio
        use_class_token: Whether to use class token
        attn_global_hidden_dim: Global attention hidden dimension
        phi_dim: Phi dimension
        downscale_depth: Downscale depth
        downscale_multiplier: Downscale multiplier
        downscale_stride_q: Downscale stride for query
        downscale_stride_k: Downscale stride for key
        noise_aug: Noise augmentation parameter
        attnpool_mode: Attention pooling mode
        name: Model name
        hflip_prob: Probability of horizontal flip
        vflip_prob: Probability of vertical flip
        rotate_prob: Probability of rotation
        noise_aug_prob: Probability of noise augmentation
        **kwargs: Additional arguments        
    """
    
    @staticmethod
    def load_config(config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            A dictionary containing the configuration.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @classmethod
    def from_config_file(cls, config_path):
        """
        Create a RiskFormer_ViT model from a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            An initialized RiskFormer_ViT model.
        """
        config = cls.load_config(config_path)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config):
        """
        Create a RiskFormer_ViT model from a configuration dictionary.
        
        Args:
            config: A dictionary containing configuration parameters.
            
        Returns:
            An initialized RiskFormer_ViT model.
        """
        # Required parameters with their keys in the config
        required_params = {
            'input_embed_dim': 'input_embed_dim',
            'output_embed_dim': 'output_embed_dim',
            'use_phi': 'use_phi',
            'drop_path_rate': 'drop_path_rate',
            'drop_rate': 'drop_rate',
            'tasks': 'tasks',
            'max_dim': 'max_dim',
            'depth': 'depth',
            'global_depth': 'global_depth',
            'encoding_method': 'encoding_method',
            'num_heads': 'num_heads',
            'use_attn_mask': 'use_attn_mask',
            'mlp_ratio': 'mlp_ratio',
            'use_class_token': 'use_class_token',
            'attn_global_hidden_dim': 'attn_global_hidden_dim',
        }
        
        # Optional parameters with default values
        optional_params = {
            'phi_dim': None,
            'downscale_depth': 1,
            'downscale_multiplier': 1.25,
            'downscale_stride_q': 2,
            'downscale_stride_k': 2,
            'noise_aug': 0.1,
            'attnpool_mode': 'conv',
            'name': None,
            'hflip_prob': 0.5,
            'vflip_prob': 0.5,
            'rotate_prob': 0.5,
            'noise_aug_prob': 0.5,
        }
        
        # Extract required parameters from config
        model_args = {}
        for param, config_key in required_params.items():
            if config_key not in config:
                raise ValueError(f"Required parameter '{config_key}' not found in config")
            model_args[param] = config[config_key]
        
        # Extract optional parameters from config
        for param, default_value in optional_params.items():
            model_args[param] = config.get(param, default_value)
                
        # Pass any other parameters from config
        for key, value in config.items():
            if key not in required_params.values() and key not in optional_params:
                model_args[key] = value
        
        logger.info(f"Initializing RiskFormer_ViT from config with parameters: {model_args}")
        
        return cls(**model_args)

    def __init__(
        self,
        input_embed_dim: int,
        output_embed_dim: int,
        use_phi: bool,
        drop_path_rate: float,
        drop_rate: float,
        tasks: Dict[str, Dict[str, Any]],  # Dictionary mapping task names to task configurations
        max_dim: int,
        depth: int,
        global_depth: int,
        encoding_method: str,
        num_heads: int,
        use_attn_mask: bool,
        mlp_ratio: float,
        use_class_token: bool,
        attn_global_hidden_dim: int,
        phi_dim: Optional[int] = None,
        downscale_depth: int = 1,
        downscale_multiplier: float = 1.25,
        downscale_stride_q: int = 2,
        downscale_stride_k: int = 2,
        noise_aug: float = 0.1,
        attnpool_mode: str = "conv",
        name: Optional[str] = None,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        noise_aug_prob: float = 0.5,
        **kwargs
    ):
        """
        Initialize the RiskFormer Vision Transformer model.
        
        Args:
            input_embed_dim: Dimension of input embeddings
            output_embed_dim: Dimension of output embeddings
            use_phi: Whether to use the phi network
            drop_path_rate: Drop path rate
            drop_rate: Dropout rate
            tasks: Dictionary mapping task names to task configurations. Each task config should include:
                - type: Task type ("binary", "regression", "multiclass")
                - num_classes: Number of output classes/values
                - metrics: Optional list of metrics to track
                - weight: Optional task weight for loss calculation
            max_dim: Maximum dimension
            depth: Number of transformer layers
            global_depth: Number of global transformer layers
            encoding_method: Position encoding method
            num_heads: Number of attention heads
            use_attn_mask: Whether to use attention mask
            mlp_ratio: MLP ratio
            use_class_token: Whether to use class token
            attn_global_hidden_dim: Global attention hidden dimension
            phi_dim: Phi dimension
            downscale_depth: Downscale depth
            downscale_multiplier: Downscale multiplier
            downscale_stride_q: Downscale stride for query
            downscale_stride_k: Downscale stride for key
            noise_aug: Noise augmentation
            attnpool_mode: Attention pooling mode
            name: Model name
            hflip_prob: Probability of horizontal flip
            vflip_prob: Probability of vertical flip
            rotate_prob: Probability of rotation
            noise_aug_prob: Probability of noise augmentation
            **kwargs: Additional arguments        
        """
        super().__init__()
        
        # Save configuration
        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim
        self.use_phi = use_phi
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.token_array_dim = max_dim
        self.depth = depth
        self.global_depth = global_depth
        self.encoding_method = encoding_method
        self.num_heads = num_heads
        self.use_attn_mask = use_attn_mask
        self.mlp_ratio = mlp_ratio
        self.use_class_token = use_class_token
        self.attn_global_hidden_dim = attn_global_hidden_dim
        self.phi_dim = phi_dim if phi_dim is not None else output_embed_dim
        self.downscale_depth = downscale_depth
        self.downscale_multiplier = downscale_multiplier
        self.downscale_stride_q = downscale_stride_q
        self.downscale_stride_k = downscale_stride_k
        self.noise_aug = noise_aug
        self.attnpool_mode = attnpool_mode
        self.name = name
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob
        self.noise_aug_prob = noise_aug_prob
        self.tasks = tasks
        
        # Define Model Dimensions
        self.blocks_input_dim = self.phi_dim if use_phi else self.output_embed_dim
        self.blocks_output_dim = self.blocks_input_dim

        self.downscale_output_dims = []
        current_dim = self.blocks_input_dim
        for i in range(self.downscale_depth):
            current_dim = current_dim * self.downscale_multiplier
            self.downscale_output_dims.append(int(current_dim))
        self.downscale_output_dims = [
            (dim + self.num_heads - 1) // self.num_heads * self.num_heads
            for dim in self.downscale_output_dims
        ]
        self.blocks_output_dim = self.downscale_output_dims[-1] if len(self.downscale_output_dims) > 0 else self.blocks_input_dim
        
        # Define input sizes for each block
        s_q = self.downscale_stride_q
        s_k = self.downscale_stride_k
        self.input_sizes = [int(self.token_array_dim / (s_q**i)) for i in range(self.downscale_depth + 1)]
        self.input_sizes = [(s, s) for s in self.input_sizes]

        # Initialize drop path rates
        self.total_blocks = self.depth + self.downscale_depth
        self.drop_path_rates = torch.linspace(0, self.drop_path_rate, self.total_blocks)


        # Initialize phi network if used
        self.initialize_phi()
        
        # Number of prefix tokens (e.g., class token)
        self.num_prefix_tokens = 1 if use_class_token else 0
        
        # Global pooling method
        self.global_pool = "token" if use_class_token else "avg"
        
        # Initialize class tokens if needed
        self.initialize_class_token()
        
        # Initialize position encodings
        self.initialize_position_encodings()
        
        # Initialize blocks
        self.initialize_downscale_blocks()
        self.initialize_local_blocks()
        self.initialize_global_blocks()
        self.initialize_global_attn()
        
        # Initialize normalization layers
        self.initialize_norm_layers()
        # Create head for predictions
        self.initialize_heads()

        # Apply weight initialization
        self.apply(self.initialize_weights)

    def initialize_phi(self):
        """Initialize phi network."""
        if self.use_phi:
            self.phi = nn.Sequential(
                nn.Linear(self.input_embed_dim, self.phi_dim, bias=False),
                nn.GELU()
            )
        else:
            self.phi = None

    def initialize_class_token(self):
        """Generate class token with specified dimension."""
        # Create a trainable class token parameter
        self.cls_token = None
        if self.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.blocks_input_dim))

    def initialize_position_encodings(self):
        """Initialize position encodings based on the specified method."""
        num_patches = int(self.token_array_dim ** 2)
        height = width = self.token_array_dim
        
        if self.encoding_method == "standard" or self.encoding_method == "":
            pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if self.use_class_token else 0), self.blocks_input_dim))
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embed = pos_embed
            self.pos_drop = nn.Dropout(p=self.drop_rate)
            
        elif self.encoding_method == "sinusoidal":
            # Use fixed sinusoidal embeddings
            self.pos_encoding = SinusoidalPositionalEncoding2D(
                channels=self.blocks_input_dim,
                height=height,
                width=width
            )
            self.pos_drop = nn.Dropout(p=self.drop_rate)
        else:
            raise ValueError(f"Unknown position encoding method: {self.encoding_method}")

    def apply_positional_encoding(self, x, height, width):
        """Apply positional encoding to the input tensor based on the specified method.
        
        This method consolidates all positional encoding implementations using standard
        libraries where possible.
        
        Args:
            x: Input tensor of shape [B, N, C]
            height: Height of the 2D grid
            width: Width of the 2D grid
            
        Returns:
            Tensor with positional encoding applied
        """
        batch_size, seq_len, channels = x.shape
        
        # Apply positional encoding based on method
        if self.encoding_method == "standard" or self.encoding_method == "":
            if self.use_class_token:
                x = x + self.pos_embed[:, 1:, :]
            else:
                x = x + self.pos_embed
                    
        elif self.encoding_method == "sinusoidal":
            x = self.pos_encoding(x)
                    
        # Apply dropout
        x = self.pos_drop(x)
        return x
        
    def initialize_weights(self, m):
        """Initialize weights for the model."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def initialize_downscale_blocks(self):
        """Initialize blocks that downscale spatial dimensions."""
        self.downscale_blocks = nn.ModuleList()
            
        # Create downscale blocks
        input_dim = self.blocks_input_dim
        for i in range(self.downscale_depth):
            dim_out = self.downscale_output_dims[i]
            self.downscale_blocks.append(
                MultiScaleBlock(
                    dim=input_dim,
                    dim_out=dim_out,
                    input_size=self.input_sizes[i],
                    num_heads=1,  # Fixed to 1 head for downscale blocks
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=self.drop_rate,  # Fixed to 0.0 for downscale blocks
                    attn_drop=self.drop_rate,  # Fixed to 0.0 for downscale blocks
                    drop_path=self.drop_path_rates[i],
                    norm_layer=nn.LayerNorm,
                    kernel_q=(self.downscale_stride_q + 1, self.downscale_stride_q + 1),
                    kernel_kv=(self.downscale_stride_k + 1, self.downscale_stride_k + 1),
                    stride_q=(self.downscale_stride_q, self.downscale_stride_q),
                    stride_kv=(self.downscale_stride_k, self.downscale_stride_k),
                    mode=self.attnpool_mode,
                    has_cls_embed=self.use_class_token,
                    rel_pos_spatial=True
                )
            )
            input_dim = dim_out
    
    def initialize_local_blocks(self):
        """Initialize the blocks for local processing of patches."""
        self.local_blocks = nn.ModuleList()
                
        input_dim = self.blocks_output_dim
        input_size = self.input_sizes[-1]

        for i in range(self.depth):
            self.local_blocks.append(
                MultiScaleBlock(
                    dim=input_dim,
                    dim_out=input_dim,
                    input_size=input_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=self.drop_rate,
                    attn_drop=self.drop_rate,
                    drop_path=self.drop_path_rates[i + self.downscale_depth].item(),
                    norm_layer=nn.LayerNorm,
                    kernel_q=(1, 1),
                    kernel_kv=(1, 1),
                    stride_q=(1, 1),
                    stride_kv=(1, 1),
                    mode=self.attnpool_mode,
                    has_cls_embed=self.use_class_token,
                    rel_pos_spatial=True
                )
            )

    def initialize_global_blocks(self):
        """Initialize blocks for generating region-level tokens."""
        self.global_blocks = nn.ModuleList()
        
        # Add GlobalMaxPoolLayer as the first global block
        self.global_blocks.append(
            GlobalMaxPoolLayer(use_class_token=self.use_class_token)
        )
            
    def initialize_global_attn(self):
        """Initialize attn layer for weighing region-level tokens."""
        # Create a global attention mechanism similar to TF implementation
        self.attn_global = nn.Sequential(
            nn.Linear(self.blocks_output_dim, self.attn_global_hidden_dim),
            nn.GELU(),
            nn.Linear(self.attn_global_hidden_dim, 1)
        )

    def initialize_norm_layers(self):
        """Initialize normalization layers."""
        self.norm = nn.LayerNorm(self.blocks_input_dim)
        self.norm_local = nn.LayerNorm(self.blocks_output_dim)
        self.norm_global = nn.LayerNorm(self.blocks_output_dim)

    def initialize_heads(self):
        """Initialize heads for predictions."""        
        self.head = RiskFormer_Head(self.tasks, self.blocks_output_dim, self.drop_rate)
        
    def generate_masks(self, x):
        """Generate attention masks for a batch of tensors.
        
        Args:
            x: Input tensor of shape [B, C, H, W] or [C, H, W].
        
        Returns:
            Boolean masks of same shape as input.
        """
        # Handle batched and un-batched inputs
        unbatched = False
        if x.ndim == 3:
            x = x.unsqueeze(0)  # Shape: [1, C, H, W]
            unbatched = True

        mask = torch.any(x != 0, dim=1)  # Shape: [B, H, W] or [1, H, W]
        if unbatched:
            return mask.squeeze(0)  # Shape: [H, W]
        return mask  # Shape: [B, H, W]
        
    def random_apply(self, x, transform_fn, p=0.5, batch_wise=True, **kwargs):
        """Apply a transformation with probability p.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            transform_fn: Function to apply to the tensor
            p: Probability of applying the transformation
            batch_wise: Whether to apply the same transformation to all samples in the batch
            **kwargs: Additional arguments to pass to transform_fn
            
        Returns:
            Transformed tensor with same shape as input
        """
        if p <= 0:
            return x
            
        batch_size = x.shape[0]
        device = x.device
        
        if batch_wise:
            # Apply same transformation to all samples in batch with probability p
            if torch.rand(1, device=device).item() < p:
                return transform_fn(x, **kwargs)
            return x
        else:
            # Apply transformation to each sample independently with probability p
            mask = torch.rand(batch_size, 1, 1, 1, device=device) < p
            if not mask.any():
                return x
                
            # Apply transformation only to selected samples
            x_transformed = transform_fn(x, **kwargs)
            return torch.where(mask, x_transformed, x)
    
    def apply_noise(self, x, noise_level=0.1):
        """Apply random noise to a tensor.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            noise_level: Standard deviation of the noise
            
        Returns:
            Noisy tensor
        """
        if noise_level <= 0:
            return x
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    def random_rotate(self, x, angles=[1, 2, 3]):
        """Apply random rotation to each sample in the batch.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            angles: List of rotation angles in multiples of 90 degrees
            
        Returns:
            Rotated tensor
        """
        batch_size = x.shape[0]
        device = x.device
        
        x_rotated = x.clone()
        rot_indices = torch.randint(0, len(angles) + 1, (batch_size,), device=device)
        for angle_idx, angle in enumerate(angles):                
            mask = (rot_indices == angle_idx)
            if not mask.any():
                continue
                
            samples = x[mask]
            rotated_samples = torch.rot90(samples, k=angle, dims=[2, 3])  # Note: dims are [2,3] for batched input
            x_rotated[mask] = rotated_samples
                
        return x_rotated
    
    def apply_token_augment(self, x):
        """ Apply augmentations to tokens in a vectorized manner.
        
        Args:
            x: Input tensor of shape [B, C, H, W] or [C, H, W].
            
        Returns:
            Augmented tensor with same shape as input.
        """
        # TODO: use augment_dict to implement dynamic
        # augmentation function with variable steps

        unbatched = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            unbatched = True
            
        if not self.training:
            if unbatched:
                return x.squeeze(0)
            return x
            
        batch_size, _, _, _ = x.shape
        x = self.random_apply(x, lambda x: torch.flip(x, dims=[2]), p=self.vflip_prob)
        x = self.random_apply(x, lambda x: torch.flip(x, dims=[3]), p=self.hflip_prob)
        x = self.random_apply(x, self.random_rotate, p=self.rotate_prob)
        x = self.random_apply(x, self.apply_noise, p=self.noise_aug_prob, noise_level=self.noise_aug)
        
        if unbatched:
            return x.squeeze(0)
        return x
    
    def forward_phi(self, x, masks=None):
        """Apply phi network to tokens.
        
        Args:
            x: Input tensor of shape [B, C, S, S] or [C, S, S].
            masks (optional): Attention masks

        Returns:
            Input tensor with reduced dimensionality in shape [B, D, S, S]
        """
        batched = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            batched = True
            
        batch_size, channels, height, width = x.shape
        
        x_flat = x.reshape(-1, channels)  # [B*H*W, C]
        if masks is not None:
            x_flat = self.phi(x_flat) * masks.reshape(-1, 1).to(torch.float32) # [B*H*W, D]
        else:
            x_flat = self.phi(x_flat) # [B*H*W, D]
        
        x_reduced = x_flat.reshape(batch_size, -1, height, width)  # [B, D, S, S]
        
        if batched:
            return x_reduced.squeeze(0)
        return x_reduced
    
    def add_class_token(self, x, masks=None):
        """Add class token to tokens.
        
        Args:
            x: Input tensor of shape [B, D, S, S]
            masks: Attention masks  
            
        Returns:
            Input tensor with class token added and updated masks
        """
        if not self.use_class_token:
            return x, masks
        
        batch_size = x.shape[0]

        # self.cls_token_local: [D] -> [bs, D]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Update masks to include class token if using attention masks
        if masks is not None:
            cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=masks.device)
            masks = torch.cat((cls_mask, masks), dim=1)
        return x, masks
    
    def prepare_tokens(self, x):
        """Prepare input tokens for transformer processing.
        Input x is a single or batch of 2-D patch embedding arrays and
        has shape [B, C, S, S] or [C, S, S].
        
        This method handles:
        1. Data augmentation (flip/rotate, noise)
        2. Generating attention masks
        3. Applying positional encoding
        4. Adding class token if required
        
        Args:
            x: Input tensor of shape [B, C, S, S] or [C, S, S].
            
        Returns:
            Processed tensor and attention masks
        """
        
        batch_size = x.shape[0]
        if self.training:
            x = self.apply_token_augment(x) # [B, C, S, S]
        
        # Generate attention masks if needed
        if self.use_attn_mask:
            attn_mask = self.generate_masks(x) # [B, H, W]
        else:
            attn_mask = None
        
        # Apply phi network if used (dimensionality adjustment)
        if self.use_phi:
            x = self.forward_phi(x, attn_mask) # [B, D, S, S]

        # Reshape into sequence format [B, N, D] for transformer
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, -1, channels) # [B, H*W, D]
        attn_mask = attn_mask.reshape(batch_size, -1) if attn_mask is not None else None # [B, H*W]
        if self.use_class_token:
            x, attn_mask = self.add_class_token(x, attn_mask)

        # Apply positional encoding
        x = self.apply_positional_encoding(x, height, width)
            
        # Add class token if required
        return x, attn_mask, (height, width)
        
    def process_downscale_blocks(self, x, hw_shape):
        """Process through downscale blocks.
        
        Args:
            x: Input tensor of shape [B, H*W, D]
            hw_shape: Height and width shape tuple (h, w)
            attn_mask: Attention masks of shape [B, H*W]
            
        Returns:
            Processed features and new hw_shape
        """
        # Process through downscale blocks
        for i, block in enumerate(self.downscale_blocks):
            x, _, hw_shape = block(x, hw_shape)
        return x, hw_shape
    
    def process_local_blocks(self, x, hw_shape):
        """Process through local transformer blocks.
        
        Args:
            x: Input tensor of shape [B, H*W, D]
            hw_shape: Height and width shape tuple (h, w)
            attn_mask: Attention masks of shape [B, H*W]
            
        Returns:
            Processed features, new hw_shape, and attention weights
        """
        attns = []
        
        # Process through local blocks
        for i, block in enumerate(self.local_blocks):
            x, attn, hw_shape = block(x, hw_shape)
            attns.append(attn)
                        
        return x, hw_shape, torch.stack(attns) if attns else None
    
    def process_global_blocks(self, x, hw_shape, mask=None):
        """Process through global transformer blocks.
        
        Args:
            x: Input tensor of shape [B, H*W, D]
            hw_shape: Height and width shape tuple (h, w)
            mask: Attention masks of shape [B, H*W]
            
        Returns:
            Processed features
        """
        # Process through global blocks (which are part of local processing in TF)
        for i, block in enumerate(self.global_blocks):
            x = block(x, mask=mask)

        return x, hw_shape    

    def produce_preds(self, x, return_weights=False):
        """Create predictions from global transformer blocks.
        
        Args:
            x: Input tensor of shape (B, D)
            masks: Attention masks
            return_weights: Whether to return attention weights
            
        Returns:
            Global predictions (and optionally attention weights)
        """
        # Apply global normalization
        x = self.norm_global(x) # (B, D)
        
        # Calculate attention weights
        weights = self.attn_global(x) # (B, 1)
        weights = F.softmax(weights, dim=0)
        
        # Apply attention pooling
        x_avg = torch.sum(x * weights, dim=0) # (D,)
        
        # Get predictions
        global_pred = self.head(x_avg) # (sum(num_classes),)
        global_pred = global_pred.unsqueeze(0) # (1, sum(num_classes))
        
        if return_weights:
            return global_pred, weights
        return global_pred
    
    def forward_features(self, x, return_weights=False):
        """Process features through all stages. Expects a single pre-processed slide as input.
        Each pre-processed slide produces B region-level token arrays of shape (C, S, S) where C
        is the embedding dimension and S is the region-level token array size.

        Args:
            x: Input tensor of shape [B, C, S, S] representing a patch token array.
            return_weights: Whether to return attention weights
            
        Returns:
            Processed features, masks, and optionally attention weights
        """
        # Prepare tokens - handles embedding, masking, etc.
        x, attn_mask, hw_shape = self.prepare_tokens(x)
        
        # Spatially consolidate tokens of shape (bs, H * W, D)
        x, hw_shape = self.process_downscale_blocks(x, hw_shape)

        # Process spatially consolidated tokens of shape (bs, h * w, D')
        x, hw_shape, attns = self.process_local_blocks(x, hw_shape)

        # Create (bs) region-level tokens
        x, hw_shape = self.process_global_blocks(x, hw_shape, mask=attn_mask)

        
        # Handle class token for bag predictions
        norm_x = self.norm_local(x) # (bs, D')
        bag_preds = self.head(norm_x) # (bs, sum(num_classes))
        
        # Process through global blocks
        if return_weights:
            global_pred, global_weights = self.produce_preds(x, return_weights=True)
            return bag_preds, global_pred, attns, global_weights
        else:
            global_pred = self.produce_preds(x)
            return bag_preds, global_pred
            
    def forward(self, x, return_weights=False):
        """Forward pass.
        
        Args:
            x: Input tensor
            return_weights: Whether to return attention weights
            
        Returns:
            Dictionary mapping task names to outputs (and optionally attention weights)
            When return_weights=True: (task_outputs, attns, global_weights)
            When return_weights=False: task_outputs
        """
        if return_weights:
            bag_preds, global_pred, attns, global_weights = self.forward_features(
                x, return_weights=True
            )
            all_preds = torch.cat([global_pred, bag_preds], dim=0)
            
            # Convert to task dictionary
            task_outputs = {}
            for task_name, (start_idx, end_idx) in self.head.task_indices.items():
                task_outputs[task_name] = all_preds[:, start_idx:end_idx]
                
            return task_outputs, attns, global_weights
        else:
            bag_preds, global_pred = self.forward_features(x)
            all_preds = torch.cat([global_pred, bag_preds], dim=0)
            
            # Convert to task dictionary
            task_outputs = {}
            for task_name, (start_idx, end_idx) in self.head.task_indices.items():
                task_outputs[task_name] = all_preds[:, start_idx:end_idx]
                
            return task_outputs
    
    def get_config(self):
        """Get model configuration as a dictionary."""
        config = {
            "input_embed_dim": self.input_embed_dim,
            "output_embed_dim": self.output_embed_dim,
            "use_phi": self.use_phi,
            "drop_path_rate": self.drop_path_rate,
            "drop_rate": self.drop_rate,
            "tasks": self.tasks,
            "token_array_dim": self.token_array_dim,
            "depth": self.depth,
            "global_depth": self.global_depth,
            "encoding_method": self.encoding_method,
            "num_heads": self.num_heads,
            "use_attn_mask": self.use_attn_mask,
            "mlp_ratio": self.mlp_ratio,
            "use_class_token": self.use_class_token,
            "attn_global_hidden_dim": self.attn_global_hidden_dim,
            "phi_dim": self.phi_dim,
            "downscale_depth": self.downscale_depth,
            "downscale_multiplier": self.downscale_multiplier,
            "downscale_stride_q": self.downscale_stride_q,
            "downscale_stride_k": self.downscale_stride_k,
            "noise_aug": self.noise_aug,
            "attnpool_mode": self.attnpool_mode,
            "name": self.name,
            "hflip_prob": self.hflip_prob,
            "vflip_prob": self.vflip_prob,
            "rotate_prob": self.rotate_prob,
            "noise_aug_prob": self.noise_aug_prob,
        }
        return config
        

class RiskFormerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for RiskFormer model.
    
    This module wraps the RiskFormer_ViT model and provides the training, validation,
    and test steps for PyTorch Lightning.
    """
    
    @classmethod
    def from_config(cls, config, task_configs=None, regional_coeff=None):
        """
        Create a RiskFormerLightningModule from a configuration dictionary.
        
        Args:
            config: A dictionary containing model and optimizer configuration parameters.
            task_configs: Optional dictionary of task configurations with unified information.
                If not provided, will be extracted from config['tasks'].
            regional_coeff: Optional regional loss coefficient.
            
        Returns:
            An initialized RiskFormerLightningModule.
        """
        # Create model config from main config
        model_config = {k: v for k, v in config.items() if k not in [
            'optimizer', 'learning_rate', 'weight_decay', 'scheduler',
            'batch_size', 'num_workers', 'max_epochs', 'min_epochs', 'patience'
        ]}
        
        # Add tasks configuration directly to model_config
        if task_configs is None:
            # Get tasks from config if not provided as an argument
            if 'tasks' not in config:
                raise ValueError("tasks configuration must be provided either in config['tasks'] or as task_configs argument")
            task_configs = config['tasks']
        
        model_config['tasks'] = task_configs
        
        # Create optimizer config
        optimizer_config = {
            'optimizer': config.get('optimizer', 'adam'),
            'learning_rate': config.get('learning_rate', 1e-4),
            'weight_decay': config.get('weight_decay', 1e-6),
            'scheduler': config.get('scheduler', 'plateau')
        }
        
        # Use provided regional_coeff or get from config
        if regional_coeff is None:
            regional_coeff = config.get('regional_coeff', 0.0)
        
        return cls(
            model_config=model_config,
            optimizer_config=optimizer_config,
            regional_coeff=regional_coeff
        )
    
    @classmethod
    def from_config_file(cls, config_path, task_configs=None, regional_coeff=None):
        """
        Create a RiskFormerLightningModule from a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file.
            task_configs: Optional dictionary of task configurations with unified information.
                If not provided, will be extracted from the config file.
            regional_coeff: Optional regional loss coefficient.
            
        Returns:
            An initialized RiskFormerLightningModule.
        """
        config = RiskFormer_ViT.load_config(config_path)
        return cls.from_config(config, task_configs, regional_coeff)
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        regional_coeff: float = 0.0,
    ):
        """
        Initialize the RiskFormer Lightning Module.
        
        Args:
            model_config: Configuration for the RiskFormer_ViT model
            optimizer_config: Configuration for the optimizer
            regional_coeff: Coefficient for weighting local vs global loss
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store all configurations as instance attributes
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.regional_coeff = regional_coeff
        self.task_configs = model_config['tasks']
        
        # Create the model
        self.model = RiskFormer_ViT(**model_config)
        
        # Extract essential information from task_configs
        self.tasks = list(model_config['tasks'].keys())
        self.task_types = {task: cfg['type'] for task, cfg in model_config['tasks'].items()}
        self.task_weights = {task: cfg.get('weight', 1.0) for task, cfg in model_config['tasks'].items()}
        
        # Create loss function
        self.loss = create_slide_level_loss(
            self.task_configs, 
            self.regional_coeff
        )
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metrics for tracking model performance."""
        self.metrics = {}
        
        for task, task_type in self.task_types.items():
            task_metrics = {}
            
            if task_type == "binary":
                # Binary classification metrics
                num_classes = 1  # Binary has one output node
                task_metrics["train_acc"] = torchmetrics.Accuracy(task="binary")
                task_metrics["val_acc"] = torchmetrics.Accuracy(task="binary")
                task_metrics["test_acc"] = torchmetrics.Accuracy(task="binary")
                
                task_metrics["train_auc"] = torchmetrics.AUROC(task="binary")
                task_metrics["val_auc"] = torchmetrics.AUROC(task="binary")
                task_metrics["test_auc"] = torchmetrics.AUROC(task="binary")
            elif task_type == "multiclass":
                # Multiclass classification metrics
                num_classes = 2  # Default to binary (2 classes) if not specified
                # For testing, ensure at least 2 classes for torchmetrics
                if "num_classes" in self.model_config and self.model_config["num_classes"] > 1:
                    num_classes = self.model_config["num_classes"]
                    
                task_metrics["train_acc"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                task_metrics["val_acc"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                task_metrics["test_acc"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                
                # F1 Score for multiclass
                task_metrics["train_f1"] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
                task_metrics["val_f1"] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
                task_metrics["test_f1"] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
                
                # AUROC for multiclass
                task_metrics["train_auc"] = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
                task_metrics["val_auc"] = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
                task_metrics["test_auc"] = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
            else:
                # Regression metrics
                task_metrics["train_mse"] = torchmetrics.MeanSquaredError()
                task_metrics["val_mse"] = torchmetrics.MeanSquaredError()
                task_metrics["test_mse"] = torchmetrics.MeanSquaredError()
                
                task_metrics["train_mae"] = torchmetrics.MeanAbsoluteError()
                task_metrics["val_mae"] = torchmetrics.MeanAbsoluteError()
                task_metrics["test_mae"] = torchmetrics.MeanAbsoluteError()
            
            # Add metrics for this task to the metrics dictionary
            self.metrics[task] = torch.nn.ModuleDict(task_metrics)
    
    def forward(self, x, return_weights=False):
        """Forward pass through the model."""
        return self.model(x, return_weights)    
    
    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        x, metadata = batch
        predictions = self(x)  # This now correctly handles any return format from RiskFormer_ViT.forward
        
        # Get labels for all tasks
        if 'labels' in metadata:
            labels = metadata['labels']
        else:
            # For backward compatibility
            labels = {task: metadata.get(task, metadata.get('label', None)) for task in self.tasks}
        
        # Calculate loss using the new loss function
        losses = self.loss(predictions, labels)
        total_loss = losses['total']
        
        # Log individual task losses
        for task, loss in losses.items():
            if task != 'total':
                task_weight = self.task_weights.get(task, 1.0)
                self.log(f'train_{task}_loss', loss, on_step=True, on_epoch=True, prog_bar=(task == self.tasks[0]))
                
                # Update metrics
                self._update_metrics(predictions, labels, task, 'train')
        
        # Log total loss
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning."""
        x, metadata = batch
        predictions = self(x)  # This now correctly handles any return format from RiskFormer_ViT.forward
        
        # Get labels for all tasks
        if 'labels' in metadata:
            labels = metadata['labels']
        else:
            # For backward compatibility
            labels = {task: metadata.get(task, metadata.get('label', None)) for task in self.tasks}
        
        # Calculate loss using the new loss function
        losses = self.loss(predictions, labels)
        total_loss = losses['total']
        
        # Log individual task losses
        for task, loss in losses.items():
            if task != 'total':
                self.log(f'val_{task}_loss', loss, on_step=False, on_epoch=True, prog_bar=(task == self.tasks[0]))
                
                # Update metrics
                self._update_metrics(predictions, labels, task, 'val')
        
        # Log total loss
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step for Lightning."""
        x, metadata = batch
        predictions = self(x)  # This now correctly handles any return format from RiskFormer_ViT.forward
        
        # Get labels for all tasks
        if 'labels' in metadata:
            labels = metadata['labels']
        else:
            # For backward compatibility
            labels = {task: metadata.get(task, metadata.get('label', None)) for task in self.tasks}
        
        # Calculate loss using the new loss function
        losses = self.loss(predictions, labels)
        total_loss = losses['total']
        
        # Log individual task losses
        for task, loss in losses.items():
            if task != 'total':
                self.log(f'test_{task}_loss', loss, on_step=False, on_epoch=True)
                
                # Update metrics
                self._update_metrics(predictions, labels, task, 'test')
        
        # Log total loss
        self.log('test_loss', total_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        opt_config = self.optimizer_config
        
        # Get optimizer
        optimizer_name = opt_config.get('optimizer', 'adam').lower()
        lr = opt_config.get('learning_rate', 1e-4)
        weight_decay = opt_config.get('weight_decay', 1e-6)
        
        if optimizer_name == 'adam':
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = opt_config.get('momentum', 0.9)
            optimizer = SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Get scheduler
        scheduler_name = opt_config.get('scheduler', 'plateau').lower()
        
        if scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=opt_config.get('factor', 0.1),
                patience=opt_config.get('patience', 10),
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=opt_config.get('t_max', 10),
                eta_min=opt_config.get('min_lr', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif scheduler_name == 'onecycle':
            max_lr = opt_config.get('max_lr', lr * 10)
            steps_per_epoch = opt_config.get('steps_per_epoch', 100)
            epochs = opt_config.get('epochs', 10)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=steps_per_epoch * epochs
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        else:
            return optimizer

    def _update_metrics(self, predictions, labels, task, stage):
        """
        Update metrics for a specific task.
        
        Args:
            predictions: Model predictions
            labels: Dictionary of labels
            task: Task name
            stage: 'train', 'val', or 'test'
        """
        # Skip if task not in predictions, labels, or metrics
        if task not in predictions or task not in labels or task not in self.metrics:
            return
            
        # Get predictions for this task
        if isinstance(predictions, dict):
            task_predictions = predictions[task]
        else:
            task_predictions = predictions
            
        # Get labels for this task
        if isinstance(labels, dict):
            task_labels = labels[task]
        else:
            task_labels = labels
            
        # Ensure predictions have the right shape
        if len(task_predictions.shape) > 1:
            # If we have instance-level predictions, use the global prediction
            preds = task_predictions[0].unsqueeze(0)  # Select global prediction and add batch dimension
        else:
            preds = task_predictions.unsqueeze(0)  # Add batch dimension
            
        # Get task type
        task_type = self.task_types[task]
        
        # Update metrics based on task type
        if task_type == "binary":
            # Binary classification metrics
            if f"{stage}_acc" in self.metrics[task]:
                # Apply sigmoid for binary classification accuracy
                binary_preds = torch.sigmoid(preds)
                self.metrics[task][f"{stage}_acc"](binary_preds, task_labels)
                acc = self.metrics[task][f"{stage}_acc"].compute()
                self.log(f'{stage}_{task}_acc', acc, on_step=False, on_epoch=True, prog_bar=False)
                
            if f"{stage}_auc" in self.metrics[task]:
                try:
                    self.metrics[task][f"{stage}_auc"](preds, task_labels)
                    auc = self.metrics[task][f"{stage}_auc"].compute()
                    self.log(f'{stage}_{task}_auc', auc, on_step=False, on_epoch=True, prog_bar=False)
                except Exception as e:
                    # AUC calculation may fail with insufficient data
                    logger.warning(f"AUC calculation failed: {e}")
                
        elif task_type == "multiclass":
            # Multiclass classification metrics
            if f"{stage}_acc" in self.metrics[task]:
                self.metrics[task][f"{stage}_acc"](preds, task_labels)
                acc = self.metrics[task][f"{stage}_acc"].compute()
                self.log(f'{stage}_{task}_acc', acc, on_step=False, on_epoch=True, prog_bar=False)
                
            if f"{stage}_f1" in self.metrics[task]:
                self.metrics[task][f"{stage}_f1"](preds, task_labels)
                f1 = self.metrics[task][f"{stage}_f1"].compute()
                self.log(f'{stage}_{task}_f1', f1, on_step=False, on_epoch=True, prog_bar=False)
                
            if f"{stage}_auc" in self.metrics[task]:
                try:
                    self.metrics[task][f"{stage}_auc"](preds, task_labels)
                    auc = self.metrics[task][f"{stage}_auc"].compute()
                    self.log(f'{stage}_{task}_auc', auc, on_step=False, on_epoch=True, prog_bar=False)
                except Exception as e:
                    # AUC calculation may fail with insufficient data
                    logger.warning(f"AUC calculation failed: {e}")
                
        elif task_type == "regression":
            # Regression metrics
            if f"{stage}_mse" in self.metrics[task]:
                self.metrics[task][f"{stage}_mse"](preds, task_labels)
                mse = self.metrics[task][f"{stage}_mse"].compute()
                self.log(f'{stage}_{task}_mse', mse, on_step=False, on_epoch=True, prog_bar=False)
                
            if f"{stage}_mae" in self.metrics[task]:
                self.metrics[task][f"{stage}_mae"](preds, task_labels)
                mae = self.metrics[task][f"{stage}_mae"].compute()
                self.log(f'{stage}_{task}_mae', mae, on_step=False, on_epoch=True, prog_bar=False)

