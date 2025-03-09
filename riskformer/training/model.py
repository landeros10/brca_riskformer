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

from riskformer.training.layers import SinusoidalPositionalEncoding2D, MultiScaleBlock, GlobalMaxPoolLayer
from riskformer.utils.training_utils import slide_level_loss

logger = logging.getLogger(__name__)

class RiskFormer_ViT(nn.Module):
    """Vision Transformer for Whole Slide Image processing with multiscale attention."""
    
    def __init__(
        self,
        input_embed_dim: int,
        output_embed_dim: int,
        use_phi: bool,
        drop_path_rate: float,
        drop_rate: float,
        num_classes: int,
        max_dim: int,
        depth: int,
        global_depth: int,
        encoding_method: str,
        num_heads: int,
        use_attn_mask: bool,
        mlp_ratio: float,
        use_class_token: bool,
        global_k: int,
        phi_dim: Optional[int] = None,
        downscale_depth: int = 1,
        downscale_multiplier: float = 1.25,
        downscale_stride_q: int = 2,
        downscale_stride_k: int = 2,
        noise_aug: float = 0.1,
        attnpool_mode: str = "conv",
        name: Optional[str] = None,
        **kwargs
    ):
        """Initialize the model.
        
        Args:
            input_embed_dim: Input embedding dimension
            output_embed_dim: Output embedding dimension
            use_phi: Whether to use phi network
            drop_path_rate: Drop path rate
            drop_rate: Dropout rate
            num_classes: Number of classes
            max_dim: Maximum dimension
            depth: Depth of local blocks
            global_depth: Depth of global blocks
            encoding_method: Position encoding method
            mask_num: Number of masks
            mask_preglobal: Whether to mask before global blocks
            num_heads: Number of attention heads
            use_attn_mask: Whether to use attention mask
            mlp_ratio: MLP ratio
            use_class_token: Whether to use class token
            global_k: Global k value
            phi_dim: Phi dimension
            downscale_depth: Depth of downscale blocks
            downscale_multiplier: Multiplier for downscale blocks
            downscale_stride_q: Stride for query in downscale blocks
            downscale_stride_k: Stride for key/value in downscale blocks
            noise_aug: Noise augmentation level
            data_dir: Data directory
            attnpool_mode: Attention pool mode
            name: Model name
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Save configuration
        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim
        self.use_phi = use_phi
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.max_dim = max_dim
        self.depth = depth
        self.global_depth = global_depth
        self.encoding_method = encoding_method
        self.num_heads = num_heads
        self.use_attn_mask = use_attn_mask
        self.mlp_ratio = mlp_ratio
        self.use_class_token = use_class_token
        self.global_k = global_k
        self.phi_dim = phi_dim if phi_dim is not None else output_embed_dim
        self.downscale_depth = downscale_depth
        self.downscale_multiplier = downscale_multiplier
        self.downscale_stride_q = downscale_stride_q
        self.downscale_stride_k = downscale_stride_k
        self.noise_aug = noise_aug
        self.attnpool_mode = attnpool_mode
        self.name = name
        
        # Set model dimension
        self.model_dim = self.phi_dim if use_phi else input_embed_dim
        
        # Initialize phi network if used
        if self.use_phi:
            self.phi = nn.Sequential(
                nn.Linear(self.input_embed_dim, self.phi_dim, bias=False),
                nn.GELU()
            )
        else:
            # Ensure phi is None when not used
            self.phi = None
            # Use delattr to completely remove the phi attribute to match test expectations
            if hasattr(self, 'phi'):
                delattr(self, 'phi')
        
        # Number of prefix tokens (e.g., class token)
        self.num_prefix_tokens = 1 if use_class_token else 0
        
        # Global pooling method
        self.global_pool = "token" if use_class_token else "avg"
        
        # Initialize class tokens if needed
        if self.use_class_token:
            self.cls_token_local = self.generate_class_tokens(self.model_dim)
            self.cls_token_global = self.generate_class_tokens(self.output_embed_dim)
        else:
            self.cls_token_local = None
            self.cls_token_global = None
        
        # Initialize position encodings
        self._initialize_position_encodings()
        
        # Initialize blocks
        self.initialize_downscale_blocks()
        self.initialize_local_blocks()
        self.initialize_global_blocks()
        self.initialize_global_attn()
        
        # Initialize normalization layers
        self.norm = nn.LayerNorm(self.model_dim)
        self.norm_local = nn.LayerNorm(self.model_dim)
        self.norm_global = nn.LayerNorm(self.output_embed_dim)
        
        # Initialize projection layer
        self.global_proj = nn.Linear(self.model_dim, self.output_embed_dim)
        
        # Initialize head for predictions
        # Add softmax activation for classification consistency with TensorFlow
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.Linear(self.model_dim, num_classes),
                nn.Softmax(dim=-1)
            )
            self.head_global = nn.Sequential(
                nn.Linear(self.output_embed_dim, num_classes),
                nn.Softmax(dim=-1)
            )
        else:
            self.head = nn.Identity()
            self.head_global = nn.Identity()
            
        # Initialize global prediction layers
        self.attn_weights = nn.Linear(self.output_embed_dim, 128)
        self.attn_weights_activation = nn.GELU()
        self.attn_weights_projection = nn.Linear(128, 1)
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
    def generate_class_tokens(self, dim):
        """Generate class token with specified dimension.
        
        Args:
            dim: Dimension of the class token
            
        Returns:
            Class token parameter
        """
        # Create a trainable class token parameter
        cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Initialize with truncated normal distribution
        nn.init.trunc_normal_(cls_token, std=0.02)
        return cls_token

    def _initialize_position_encodings(self):
        """Initialize position encodings based on the specified method."""
        num_patches = int(math.sqrt(self.max_dim))
        height = width = num_patches
        
        if self.encoding_method == "standard" or self.encoding_method == "":
            # Use learnable positional embeddings
            pos_embed = nn.Parameter(torch.zeros(1, self.max_dim + (1 if self.use_class_token else 0), self.model_dim))
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embed = pos_embed
            self.pos_drop = nn.Dropout(p=self.drop_rate)
            
        elif self.encoding_method == "sinusoidal":
            # Use fixed sinusoidal embeddings
            self.pos_encoding = SinusoidalPositionalEncoding2D(
                channels=self.model_dim,
                height=height,
                width=width
            )
            self.pos_drop = nn.Dropout(p=self.drop_rate)
            
        elif self.encoding_method == "conditional":
            # Use conditional positional encoding through depth-wise convolution
            self.peg = nn.Conv2d(
                self.model_dim, 
                self.model_dim, 
                kernel_size=3, 
                padding=1, 
                groups=self.model_dim, 
                bias=False
            )
            nn.init.normal_(self.peg.weight, std=0.02)
            self.pos_drop = nn.Dropout(p=self.drop_rate)
            
        elif self.encoding_method == "ppeg":
            # Use Pyramid Positional Encoding
            self.ppeg = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.model_dim, self.model_dim, kernel_size=3, padding=1, groups=self.model_dim),
                    nn.GELU(),
                    nn.Conv2d(self.model_dim, self.model_dim, kernel_size=3, padding=1, groups=self.model_dim),
                    nn.GELU(),
                    nn.Conv2d(self.model_dim, self.model_dim, kernel_size=3, padding=1, groups=self.model_dim),
                )
                for _ in range(3)  # Use 3 levels for pyramid encoding
            ])
            self.pos_drop = nn.Dropout(p=self.drop_rate)
        else:
            raise ValueError(f"Unknown position encoding method: {self.encoding_method}")

    def _apply_positional_encoding(self, x, height, width):
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
            # For standard encoding, we add the position embedding
            if self.use_class_token:
                # If using class token later, just add position embedding to sequence
                if seq_len <= self.pos_embed.shape[1] - 1:
                    # Use the pre-computed position embedding
                    pos_embed = self.pos_embed[:, 1:seq_len+1, :]
                    x = x + pos_embed
                else:
                    # Need to interpolate position embedding for larger sequence
                    pos_embed = self.pos_embed[:, 1:, :]
                    pos_embed = F.interpolate(
                        pos_embed.permute(0, 2, 1).unsqueeze(0),
                        size=seq_len,
                        mode='linear'
                    ).squeeze(0).permute(0, 2, 1)
                    x = x + pos_embed
            else:
                # If no class token, just add position embedding
                if seq_len <= self.pos_embed.shape[1]:
                    # Use the pre-computed position embedding
                    x = x + self.pos_embed[:, :seq_len, :]
                else:
                    # Need to interpolate position embedding for larger sequence
                    pos_embed = self.pos_embed
                    pos_embed = F.interpolate(
                        pos_embed.permute(0, 2, 1).unsqueeze(0),
                        size=seq_len,
                        mode='linear'
                    ).squeeze(0).permute(0, 2, 1)
                    x = x + pos_embed
                    
        elif self.encoding_method == "sinusoidal":
            # For sinusoidal, we apply the encoding function
            x = self.pos_encoding(x)
            
        elif self.encoding_method == "conditional":
            # For conditional encoding, we need to reshape to 2D, apply conv, then reshape back
            x = x.reshape(batch_size, height, width, channels)
            # Convert to channels-first format for convolution
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
            # Apply conditional encoding
            x = self.peg(x)
            # Convert back to sequence format
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
            x = x.reshape(batch_size, height * width, channels)
            
        elif self.encoding_method == "ppeg":
            # For pyramid encoding, similar to conditional but with multiple levels
            x = x.reshape(batch_size, height, width, channels)
            # Convert to channels-first format
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Apply pyramid position encoding
            for encoder in self.ppeg:
                # Apply encoder at each level
                x = x + encoder(x)
                
            # Convert back to sequence format
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
            x = x.reshape(batch_size, height * width, channels)
        
        # Apply dropout
        x = self.pos_drop(x)
        
        return x
        
    def _init_weights(self, m):
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
        self.output_dims = []
        current_dim = self.output_embed_dim
        for i in range(self.downscale_depth):
            current_dim = int(current_dim * self.downscale_multiplier)
            self.output_dims.append(current_dim)
        self.output_dims = [
            (dim + self.num_heads - 1) // self.num_heads * self.num_heads
            for dim in self.output_dims
        ]
            
        # Calculate input sizes for each block
        s_q = self.downscale_stride_q
        s_k = self.downscale_stride_k
        self.input_sizes = [int(self.max_dim / (s_q**i)) for i in range(self.downscale_depth + 1)]
        self.input_sizes = [(s, s) for s in self.input_sizes]
        
        # Create downscale blocks
        for i in range(self.downscale_depth):
            input_dim = self.model_dim if i == 0 else self.output_dims[i-1]
            self.downscale_blocks.append(
                MultiScaleBlock(
                    dim=input_dim,
                    dim_out=self.output_dims[i],
                    input_size=self.input_sizes[i],
                    num_heads=1,  # Fixed to 1 head for downscale blocks
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,  # Fixed to 0.0 for downscale blocks
                    attn_drop=0.0,  # Fixed to 0.0 for downscale blocks
                    drop_path=0.0,  # Fixed to 0.0 for downscale blocks
                    norm_layer=nn.LayerNorm,
                    kernel_q=(s_q + 1, s_q + 1),
                    kernel_kv=(s_k + 1, s_k + 1),
                    stride_q=(s_q, s_q),
                    stride_kv=(s_k, s_k),
                    mode=self.attnpool_mode,
                    has_cls_embed=self.use_class_token,
                    rel_pos_spatial=True
                )
            )
    
    def initialize_local_blocks(self):
        """Initialize the blocks for local processing of patches."""
        # Calculate drop path rates
        total_depth = self.depth + self.downscale_depth
        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, total_depth)]
        self.local_blocks = nn.ModuleList()
                
        # Get input dimension and size based on downscale blocks
        input_dim = self.output_embed_dim if len(self.output_dims) == 0 else self.output_dims[-1]
        input_size = self.input_sizes[-1]  #

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
                    drop_path=self.dpr[i + self.downscale_depth],
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
        """Initialize blocks for global processing."""
        self.global_blocks = nn.ModuleList()
        
        # Add GlobalMaxPoolLayer as the first global block
        self.global_blocks.append(
            GlobalMaxPoolLayer(use_class_token=self.use_class_token)
        )
            
    def initialize_global_attn(self):
        """Initialize global attention layer."""
        # Create a global attention mechanism similar to TF implementation
        self.global_attn = nn.Sequential(
            nn.Linear(self.output_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
            
    def generate_masks(self, x):
        """Generate attention masks for a batch of tensors.
        
        Args:
            x: Input tensor of shape [B, H, W, C]
        
        Returns:
            Boolean masks of shape [B, H*W] or [B, H*W+1] if use_class_token is True
        """
        if not self.use_attn_mask:
            return None  # No need to generate masks
        
        # Check if any value in the feature dimension is non-zero
        # This creates a mask where True indicates a valid token
        mask = torch.any(x != 0, dim=-1)  # Shape: [B, H, W]
        
        # Reshape to [B, H*W]
        batch_size = x.shape[0]
        mask = mask.reshape(batch_size, -1)  # Shape: [B, H*W]
        
        # Note: We don't add class token mask here as it's added in prepare_tokens
        # after the positional encoding is applied
        
        return mask
        
    def apply_token_augment(self, x):
        """ Apply augmentations to tokens 
        
        Args:
            x: Input tensor of shape [B, C, S, S]
            
        Returns:
            Augmented tensor
        """
        # Random horizontal flip

        # Random vertical flip

        # Random 90-degree rotation

        # Random noise

        return x
        
    def forward_phi(self, x, masks=None):
        """Apply phi network to tokens.
        
        Args:
            x: Input tensor of shape [B, C, S, S]
            masks: Attention masks
            
        Returns:
            Input tensor with reduced dimensionality
        """
        batch_size, height, width, channels = x.shape
        x_flat = x.reshape(-1, channels)
        x_flat = self.phi(x_flat)
        
        if masks is not None:
            mask_flat = masks.reshape(-1, 1)
            x_flat = x_flat * mask_flat
        x = x_flat.reshape(batch_size, height, width, -1)
        return x
    
    def add_class_token(self, x, masks=None):
        """Add class token to tokens.
        
        Args:
            x: Input tensor of shape [B, C, S, S]
            masks: Attention masks  
            
        Returns:
            Input tensor with class token added and updated masks
        """
        batch_size = x.shape[0]
        cls_tokens = self.cls_token_local.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Update masks to include class token if using attention masks
        if masks is not None:
            cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=masks.device)
            masks = torch.cat((cls_mask, masks), dim=1)
        return x, masks
    
    def prepare_tokens(self, x):
        """Prepare input tokens for transformer processing.
        Input x is a 2-D array of small patch embeddings and
        has shape [B, C, S, S]
        
        This method handles:
        1. Reshaping input for transformer processing
        2. Data augmentation (flip/rotate, noise)
        3. Generating attention masks
        4. Applying positional encoding
        5. Adding class token if required
        
        Args:
            x: Input tensor of shape [B, C, S, S]
            
        Returns:
            Processed tensor and attention masks
        """
        
        batch_size = x.shape[0]
        if self.training:
            x = self.apply_token_augment(x)
        
        # Generate attention masks if needed
        if self.use_attn_mask:
            masks = self.generate_masks(x)
        else:
            masks = None
        
        # Apply phi network if used (dimensionality adjustment)
        if self.use_phi:
            x = self.forward_phi(x, masks)

        # Reshape into sequence format [B, N, C] for transformer
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        
        # Apply positional encoding
        x = self._apply_positional_encoding(x, height, width)
            
        # Add class token if required
        if self.use_class_token:
            x, masks = self.add_class_token(x, masks)
        return x, masks, (height, width)
        
    def flip_rotate(self, x):
        """Apply random flip and rotation augmentation.
        
        Args:
            x: Input tensor
            
        Returns:
            Augmented tensor
        """
        # Only apply augmentation during training
        if not self.training:
            return x
            
        batch_size, height, width, channels = x.shape
        augmented_batch = []
        
        for i in range(batch_size):
            img = x[i]  # [H, W, C]
            
            # Random horizontal flip (50% probability)
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])
                
            # Random vertical flip (50% probability)
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[0])
                
            # Random 90-degree rotation (25% probability for each rotation)
            rot_choice = torch.randint(0, 4, (1,)).item()
            if rot_choice == 1:  # 90 degrees
                img = img.permute(1, 0, 2).flip(dims=[0])
            elif rot_choice == 2:  # 180 degrees
                img = torch.flip(img, dims=[0, 1])
            elif rot_choice == 3:  # 270 degrees
                img = img.permute(1, 0, 2).flip(dims=[1])
                
            augmented_batch.append(img)
            
        # Stack back into a batch
        return torch.stack(augmented_batch)
    
    def random_noise(self, x, masks):
        """Apply random noise augmentation.
        
        Args:
            x: Input tensor
            masks: Attention masks
            
        Returns:
            Noisy tensor
        """
        # Only apply noise during training and if noise_aug > 0
        if not self.training or self.noise_aug <= 0:
            return x
            
        batch_size, height, width, channels = x.shape
        
        # Generate random noise
        noise = torch.randn_like(x) * self.noise_aug
        
        # If masks are provided, only apply noise to unmasked regions
        if masks is not None:
            # Expand mask to match x dimensions
            mask_expanded = masks.unsqueeze(-1).expand_as(x)
            # Apply noise only to unmasked regions
            noisy_x = x + noise * mask_expanded
        else:
            # Apply noise to all regions
            noisy_x = x + noise
            
        return noisy_x
    
    def process_downscale_blocks(self, x, hw_shape, masks=None):
        """Process through downscale blocks.
        
        Args:
            x: Input tensor
            hw_shape: Height and width shape tuple (h, w)
            masks: Attention masks
            
        Returns:
            Processed features and new hw_shape
        """
        # Process through downscale blocks
        for i, block in enumerate(self.downscale_blocks):
            x, _, hw_shape, _ = block(x, hw_shape)
            
        return x, hw_shape
    
    def process_local_blocks(self, x, hw_shape, masks=None):
        """Process through local transformer blocks.
        
        Args:
            x: Input tensor
            hw_shape: Height and width shape tuple (h, w)
            masks: Attention masks
            
        Returns:
            Processed features, new hw_shape, and attention weights
        """
        attns = []
        
        # Process through local blocks
        for i, block in enumerate(self.local_blocks):
            x, attn, hw_shape = block(x, hw_shape)
            attns.append(attn)
                        
        return x, hw_shape, torch.stack(attns) if attns else None
    
    def process_global_blocks(self, x, hw_shape, masks=None):
        """Process through global transformer blocks.
        
        Args:
            x: Input tensor
            hw_shape: Height and width shape tuple (h, w)
            masks: Attention masks
            
        Returns:
            Processed features
        """
        # Process through global blocks (which are part of local processing in TF)
        for i, block in enumerate(self.global_blocks):
            x = block(x, attention_mask=masks)

        return x, hw_shape
    

    def produce_preds(self, x, masks=None, return_weights=False):
        """Create predictions from global transformer blocks.
        
        Args:
            x: Input tensor
            masks: Attention masks
            return_weights: Whether to return attention weights
            
        Returns:
            Global predictions (and optionally attention weights)
        """
        # Apply global normalization
        x = self.norm_global(x)
        
        # Calculate attention weights
        weights = self.attn_weights(x)
        weights = self.attn_weights_activation(weights)
        weights = self.attn_weights_projection(weights)
        
        # Apply softmax to get normalized weights
        weights = F.softmax(weights, dim=1)
        
        # Apply attention pooling
        x_weighted = x * weights
        x_avg = torch.sum(x_weighted, dim=1)
        
        # Get predictions
        global_pred = self.head_global(x_avg)
        
        if return_weights:
            return global_pred, weights
        return global_pred
    
    def forward_features(self, x, return_weights=False):
        """Process features through all stages.

        Args:
            x: Input tensor of shape [B, C, S, S]
            return_weights: Whether to return attention weights
            
        Returns:
            Processed features, masks, and optionally attention weights
        """
        # Prepare tokens - handles embedding, masking, etc.
        x, masks, hw_shape = self.prepare_tokens(x)
        
        # Spatially consolidate tokens of (bs, h * w, D)
        x, hw_shape = self.process_downscale_blocks(x, hw_shape, masks)

        # Process spatially consolidated tokens of shape (bs, h' * w', D)
        x, hw_shape, attns = self.process_local_blocks(x, hw_shape, masks)

        # Create (bs) region-level tokens of expanded dim
        x, hw_shape = self.process_global_blocks(x, hw_shape, masks)
        embed_dim = x.shape[-1]
        
        # Handle class token for bag predictions
        norm_x = self.norm_local(x[:, 0, :])
        if self.use_class_token:
            bag_preds = self.head(norm_x)
            x = x[:, 1:, :].reshape(1, -1, embed_dim)  # Reshape non-class token per region
        else:
            bag_preds = self.head(norm_x)
            x = x.reshape(1, -1, embed_dim)  # Reshape single global token per region
        
        # Define global mask and prepare for global processing
        global_mask = self.define_global_mask(masks)
        x, global_mask = self.select_and_shuffle_unmasked(x, global_mask)
        
        # Process through global blocks
        if return_weights:
            global_pred, global_weights = self.produce_preds(x, global_mask, return_weights=True)
            global_weights = global_weights.reshape(-1, hw_shape[0], hw_shape[1])
            return bag_preds, global_pred, attns, global_weights
        else:
            global_pred = self.produce_preds(x, global_mask)
            return bag_preds, global_pred
    
    def define_global_mask(self, masks):
        """Define global mask from individual masks.
        
        Args:
            masks: Attention masks from prepare_tokens
            
        Returns:
            Combined global mask
        """
        # If no masks, return None
        if masks is None or not self.use_attn_mask:
            return None
            
        # Otherwise, process masks similar to TensorFlow implementation
        return masks

    def select_and_shuffle_unmasked(self, x, global_mask):
        """Select and shuffle unmasked tokens for global processing.
        
        Args:
            x: Input tokens
            global_mask: Global attention mask
            
        Returns:
            Selected tokens and updated mask
        """
        # If no mask, return as is
        if global_mask is None:
            return x, None
            
        # Get dimensions
        batch_size, seq_len, embed_dim = x.shape
        
        # Select only unmasked tokens
        unmasked_indices = torch.nonzero(global_mask.squeeze(-1), as_tuple=True)
        unmasked_x = x[unmasked_indices]
        
        # Reshape to [1, N', D] where N' is the number of unmasked tokens
        unmasked_x = unmasked_x.reshape(1, -1, embed_dim)
        
        # Create a new mask for the unmasked tokens (all 1s)
        new_mask = torch.ones(1, unmasked_x.shape[1], 1, device=x.device)
        
        # Shuffle tokens during training for better generalization
        if self.training:
            # Get number of tokens
            num_tokens = unmasked_x.shape[1]
            # Create random permutation
            perm = torch.randperm(num_tokens, device=x.device)
            # Apply permutation
            unmasked_x = unmasked_x[:, perm, :]
            
        return unmasked_x, new_mask
    
    def forward_head(self, x):
        """Apply head to class token or average pooled features."""
        if self.use_class_token:
            x = x[:, 0]
        
        return self.head(x)
    
    def forward(self, x, return_weights=False):
        """Forward pass.
        
        Args:
            x: Input tensor
            return_weights: Whether to return attention weights
            
        Returns:
            Model output (and optionally attention weights)
            When return_weights=True: (all_preds, attns, global_weights)
            When return_weights=False: all_preds
        """
        if return_weights:
            bag_preds, global_pred, attns, global_weights = self.forward_features(
                x, return_weights=True
            )
            all_preds = torch.cat([global_pred, bag_preds], dim=0)
            return all_preds, attns, global_weights
        else:
            bag_preds, global_pred = self.forward_features(x)
            all_preds = torch.cat([global_pred, bag_preds], dim=0)
            return all_preds
    
    def get_config(self):
        """Get model configuration as a dictionary."""
        config = {
            "input_embed_dim": self.input_embed_dim,
            "output_embed_dim": self.output_embed_dim,
            "use_phi": self.use_phi,
            "drop_path_rate": self.drop_path_rate,
            "drop_rate": self.drop_rate,
            "num_classes": self.num_classes,
            "max_dim": self.max_dim,
            "depth": self.depth,
            "global_depth": self.global_depth,
            "encoding_method": self.encoding_method,
            "num_heads": self.num_heads,
            "use_attn_mask": self.use_attn_mask,
            "mlp_ratio": self.mlp_ratio,
            "use_class_token": self.use_class_token,
            "global_k": self.global_k,
            "phi_dim": self.phi_dim,
            "downscale_depth": self.downscale_depth,
            "downscale_multiplier": self.downscale_multiplier,
            "downscale_stride_q": self.downscale_stride_q,
            "downscale_stride_k": self.downscale_stride_k,
            "noise_aug": self.noise_aug,
            "attnpool_mode": self.attnpool_mode,
            "name": self.name
        }
        return config
        

class RiskFormerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for RiskFormer model.
    
    This module wraps the RiskFormer_ViT model and provides the training, validation,
    and test steps for PyTorch Lightning.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        class_loss_map: Dict[str, Dict[int, torch.nn.Module]],
        task_weights: Optional[Dict[str, float]] = None,
        regional_coeff: float = 0.0,
    ):
        """
        Initialize the RiskFormer Lightning Module.
        
        Args:
            model_config: Configuration for the RiskFormer_ViT model
            optimizer_config: Configuration for the optimizer
            class_loss_map: Dictionary mapping task names to loss functions for each class
            task_weights: Optional dictionary mapping task names to weights for loss calculation
            regional_coeff: Coefficient for weighting local vs global loss
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store all configurations as instance attributes
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.class_loss_map = class_loss_map
        self.regional_coeff = regional_coeff
        
        # Create the model
        self.model = RiskFormer_ViT(**model_config)
        
        # Set task weights (default to 1.0 if not provided)
        self.task_weights = task_weights or {task: 1.0 for task in class_loss_map.keys()}
        
        # Define tasks and their types
        self.tasks = list(class_loss_map.keys())
        self.task_types = {}
        for task, loss_map in class_loss_map.items():
            # Determine if binary, multiclass, or regression
            first_loss = next(iter(loss_map.values()))
            if isinstance(first_loss, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                self.task_types[task] = "binary"
            elif isinstance(first_loss, nn.CrossEntropyLoss):
                self.task_types[task] = "multiclass" 
            elif len(loss_map) > 1:
                self.task_types[task] = "multiclass"
            else:
                self.task_types[task] = "regression"
        
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
    
    def _calculate_task_loss(self, predictions, labels, task, stage):
        """
        Calculate loss for a specific task.
        
        Args:
            predictions: Model predictions
            labels: Dictionary of labels or a tensor for a specific task
            task: Task name
            stage: 'train', 'val', or 'test'
            
        Returns:
            Loss value for the task
        """
        # If labels is a dictionary, extract the task-specific labels
        if isinstance(labels, dict):
            if task not in labels:
                # Skip tasks without labels
                return None
            task_labels = labels[task]
        else:
            # If labels is already a tensor, use it directly
            task_labels = labels
        
        # Check if task exists in class_loss_map
        if task not in self.class_loss_map or task not in self.task_types:
            # Skip non-existent tasks
            return None
            
        task_loss_map = self.class_loss_map[task]
        task_type = self.task_types[task]
        
        # Ensure labels have the right shape
        if isinstance(task_labels, torch.Tensor):
            if len(task_labels.shape) == 0:
                task_labels = task_labels.unsqueeze(0)
            elif len(task_labels.shape) == 2 and task_labels.shape[0] == 1:
                task_labels = task_labels.squeeze(0)
        
        # Calculate loss using slide_level_loss
        loss = slide_level_loss(
            predictions, 
            task_labels, 
            task_loss_map, 
            regional_coeff=self.regional_coeff
        )
        
        # Log task-specific loss
        self.log(f'{stage}_{task}_loss', loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=(task == self.tasks[0]))
        
        # Get predictions and targets
        if len(predictions.shape) > 1:
            # If we have instance-level predictions, use the global prediction
            preds = predictions[0].unsqueeze(0)  # Select global prediction and add batch dimension
        else:
            preds = predictions.unsqueeze(0)  # Add batch dimension
            
        # Threshold predictions for binary classification
        # For binary tasks, we need logits for metrics
        task_type = self.task_types[task]
        
        # Use the task_labels from earlier in the method, not the original labels
        # task_labels = labels
        
        # Log metrics based on task type
        if task_type == "binary":
            # Binary classification metrics
            acc = self.metrics[task][f"{stage}_acc"](preds, task_labels)
            self.log(f'{stage}_{task}_acc', acc, on_step=False, on_epoch=True, prog_bar=False)
            
            # AUROC for binary classification
            auroc_preds = torch.sigmoid(preds) if preds.shape[-1] == 1 else preds
            try:
                auroc = self.metrics[task][f"{stage}_auc"](auroc_preds, task_labels)
                self.log(f'{stage}_{task}_auc', auroc, on_step=False, on_epoch=True, prog_bar=False)
            except Exception as e:
                # AUROC can fail if all labels are the same
                logger.warning(f"Failed to compute {stage}_{task}_auc: {e}")
                
        elif task_type == "multiclass":
            # Multiclass classification metrics
            acc = self.metrics[task][f"{stage}_acc"](preds, task_labels)
            self.log(f'{stage}_{task}_acc', acc, on_step=False, on_epoch=True, prog_bar=False)
            
            # F1 Score
            try:
                f1 = self.metrics[task][f"{stage}_f1"](preds, task_labels)
                self.log(f'{stage}_{task}_f1', f1, on_step=False, on_epoch=True, prog_bar=False)
            except Exception as e:
                logger.warning(f"Failed to compute {stage}_{task}_f1: {e}")
            
            # AUROC
            try:
                auc = self.metrics[task][f"{stage}_auc"](preds, task_labels)
                self.log(f'{stage}_{task}_auc', auc, on_step=False, on_epoch=True, prog_bar=False)
            except Exception as e:
                logger.warning(f"Failed to compute {stage}_{task}_auc: {e}")
                
        elif task_type == "regression":
            # Regression metrics
            mse = self.metrics[task][f"{stage}_mse"](preds, task_labels)
            mae = self.metrics[task][f"{stage}_mae"](preds, task_labels)
            
            self.log(f'{stage}_{task}_mse', mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'{stage}_{task}_mae', mae, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        x, metadata = batch
        predictions = self(x)
        
        # Get labels for all tasks
        if 'labels' in metadata:
            labels = metadata['labels']
        else:
            # For backward compatibility
            labels = {task: metadata.get(task, metadata.get('label', None)) for task in self.tasks}
        
        # Calculate loss for each task
        total_loss = 0.0
        task_losses = {}
        
        for task in self.tasks:
            task_loss = self._calculate_task_loss(predictions, labels, task, 'train')
            if task_loss is not None:
                task_weight = self.task_weights.get(task, 1.0)
                weighted_loss = task_loss * task_weight
                task_losses[task] = weighted_loss
                total_loss += weighted_loss
        
        # Log total loss
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning."""
        x, metadata = batch
        predictions = self(x)
        
        # Get labels for all tasks
        if 'labels' in metadata:
            labels = metadata['labels']
        else:
            # For backward compatibility
            labels = {task: metadata.get(task, metadata.get('label', None)) for task in self.tasks}
        
        # Calculate loss for each task
        total_loss = 0.0
        task_losses = {}
        
        for task in self.tasks:
            task_loss = self._calculate_task_loss(predictions, labels, task, 'val')
            if task_loss is not None:
                task_weight = self.task_weights.get(task, 1.0)
                weighted_loss = task_loss * task_weight
                task_losses[task] = weighted_loss
                total_loss += weighted_loss
        
        # Log total loss
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step for Lightning."""
        x, metadata = batch
        predictions = self(x)
        
        # Get labels for all tasks
        if 'labels' in metadata:
            labels = metadata['labels']
        else:
            # For backward compatibility
            labels = {task: metadata.get(task, metadata.get('label', None)) for task in self.tasks}
        
        # Calculate loss for each task
        total_loss = 0.0
        task_losses = {}
        
        for task in self.tasks:
            task_loss = self._calculate_task_loss(predictions, labels, task, 'test')
            if task_loss is not None:
                task_weight = self.task_weights.get(task, 1.0)
                weighted_loss = task_loss * task_weight
                task_losses[task] = weighted_loss
                total_loss += weighted_loss
        
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

