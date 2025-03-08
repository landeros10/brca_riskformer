'''
layers.py

Vision Transformer layers for Whole Slide Image processing.
Author: landeros10
Created: 2025-02-05
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

def attention_pool(x, pool, hw_shape, has_cls_embed=True):
    """
    Apply pooling to attention features.
    Args:
        x: Input tensor.
        pool: Pooling layer.
        hw_shape: Height and width shape tuple.
        has_cls_embed: Whether the input has class embedding.
    Returns:
        Pooled tensor and new hw_shape.
    """
    if has_cls_embed:
        cls_token, x = torch.tensor_split(x, [1], dim=1)
    
    B, N, C = x.shape
    H, W = hw_shape
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
    x = pool(x)
    
    H, W = x.shape[-2:]
    x = x.flatten(2).transpose(1, 2)
    
    if has_cls_embed:
        x = torch.cat((cls_token, x), dim=1)
        
    return x, (H, W)

def calc_rel_pos_spatial(
    attn: torch.Tensor,
    q: torch.Tensor,
    has_cls_embed: bool,
    q_shape: Tuple[int, int],
    k_shape: Tuple[int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
):
    """
    Calculate relative positional embeddings.
    Args:
        attn: Attention map.
        q: Query q.
        has_cls_embed: Whether the input has class token.
        q_shape: Shape of q token map.
        k_shape: Shape of k token map.
        rel_pos_h: Relative position embeddings (height).
        rel_pos_w: Relative position embeddings (width).
    Returns:
        attn: Attention map with relative positional embeddings.
    """
    q_h, q_w = q_shape
    k_h, k_w = k_shape
    
    # Interpolate rel pos if shapes don't match.
    if q_h != k_h or q_w != k_w:
        rel_pos_h = F.interpolate(
            rel_pos_h.reshape(1, rel_pos_h.shape[0], -1).permute(0, 2, 1),
            size=q_h,
            mode="linear",
        ).permute(0, 2, 1).reshape(-1, q_h)
        
        rel_pos_w = F.interpolate(
            rel_pos_w.reshape(1, rel_pos_w.shape[0], -1).permute(0, 2, 1),
            size=q_w,
            mode="linear",
        ).permute(0, 2, 1).reshape(-1, q_w)
    
    # Get relative positional embeddings for height and width.
    B, nH, _, _ = attn.shape
    
    # Apply class token offset if exists.
    q_t = q
    if has_cls_embed:
        q_t = q[:, 1:]
        
    q_t = q_t.reshape(B, nH, q_h * q_w, -1)
    rel_h = torch.einsum("bhnm,hm->bhnm", q_t, rel_pos_h)
    rel_w = torch.einsum("bhnm,hm->bhnm", q_t, rel_pos_w)
    
    # Shift and scale if needed.
    attn_map = attn.reshape(B, nH, q_h * q_w, k_h * k_w)
    attn_map = attn_map + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    
    return attn_map.reshape(B, nH, q_h * q_w, k_h * k_w)

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training=False):
        return drop_path(x, self.drop_prob, training)

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    Basic attention module.
    """
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0.,
        residual=False, 
        residual_conv_kernel=3
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # For residual connection with convolution
        self.residual = residual
        if self.residual:
            padding = (residual_conv_kernel - 1) // 2
            self.res_conv = nn.Conv2d(
                dim, 
                dim, 
                kernel_size=residual_conv_kernel,
                padding=padding,
                groups=dim,
            )
            
    def res_conv_fn(self, v, height=None, width=None):
        B, nH, L, d = v.shape
        assert height is not None and width is not None
        
        v = v.transpose(1, 2).reshape(B, L, nH * d)
        cls_token, v = torch.tensor_split(v, [1], dim=1)
        
        # Reshape to 2D
        v = v.reshape(B, height, width, nH * d).permute(0, 3, 1, 2)
        
        # Apply residual convolution
        v = self.res_conv(v)
        
        # Reshape back
        v = v.permute(0, 2, 3, 1).reshape(B, height * width, nH * d)
        v = torch.cat([cls_token, v], dim=1)
        
        # Back to multi-head format
        v = v.reshape(B, L, nH, d).transpose(1, 2)
        
        return v
            
    def forward(self, x, attention_mask=None, height=None, width=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v)
        
        if self.residual:
            v_res = self.res_conv_fn(v, height, width)
            x = x + v_res
            
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

class MultiScaleAttention(nn.Module):
    """
    Multiscale Multi-head Attention from MVITv2.
    """
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        mode="conv",
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.has_cls_embed = has_cls_embed
        self.pool_first = pool_first
        self.norm_layer = norm_layer
        self.mode = mode
        self.residual_pooling = residual_pooling
        
        # Get input/output shapes for Q, K, V
        self.dim_in = dim
        self.input_size = input_size
        
        # Pooling configurations for q, k, v
        self.kernel_q = kernel_q
        self.kernel_kv = kernel_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        
        # Calculate pool kernel/stride/padding for q, k, v
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]
        
        # Q, K, V projections
        self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Linear(dim_out, dim_out)
        
        # Dropouts
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Setup for pooling
        if mode == "conv":
            self.pool_q = nn.Conv2d(
                dim, 
                dim,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=dim,
            ) if kernel_q[0] > 1 else nn.Identity()
            
            self.pool_k = nn.Conv2d(
                dim,
                dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim,
            ) if kernel_kv[0] > 1 else nn.Identity()
            
            self.pool_v = nn.Conv2d(
                dim,
                dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim,
            ) if kernel_kv[0] > 1 else nn.Identity()
        elif mode == "avg":
            self.pool_q = nn.AvgPool2d(
                kernel_q,
                stride=stride_q,
                padding=padding_q,
            ) if kernel_q[0] > 1 else nn.Identity()
            
            self.pool_k = nn.AvgPool2d(
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
            ) if kernel_kv[0] > 1 else nn.Identity()
            
            self.pool_v = nn.AvgPool2d(
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
            ) if kernel_kv[0] > 1 else nn.Identity()
        else:
            raise NotImplementedError(f"Pooling mode {mode} not supported")
            
        # Relative positional embedding
        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_zero_init = rel_pos_zero_init
        
        if self.rel_pos_spatial:
            # Initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(self.num_heads, input_size[0], input_size[0]))
            self.rel_pos_w = nn.Parameter(torch.zeros(self.num_heads, input_size[1], input_size[1]))
            
            if not self.rel_pos_zero_init:
                self._init_rel_pos()
    
    def _init_rel_pos(self):
        # Initialize relative positional embeddings to follow log-linear space
        pos_seq_h = torch.arange(self.input_size[0], device=self.rel_pos_h.device)
        pos_seq_w = torch.arange(self.input_size[1], device=self.rel_pos_w.device)
        
        # Get grid of coordinates
        grid_h = pos_seq_h.unsqueeze(0) - pos_seq_h.unsqueeze(1)
        grid_w = pos_seq_w.unsqueeze(0) - pos_seq_w.unsqueeze(1)
        
        # Convert to normalized distances
        grid_h = grid_h.float() / self.input_size[0]
        grid_w = grid_w.float() / self.input_size[1]
        
        # Initialize with small random values
        self.rel_pos_h.data = grid_h.unsqueeze(0).repeat(self.num_heads, 1, 1) * 0.01
        self.rel_pos_w.data = grid_w.unsqueeze(0).repeat(self.num_heads, 1, 1) * 0.01
            
    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        
        # Separate class tokens if present
        if self.has_cls_embed:
            cls_token, x = torch.tensor_split(x, [1], dim=1)
            
        if self.pool_first:
            # B, N, C -> B, C, H, W
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # Apply pooling
            x_q = self.pool_q(x).flatten(2).transpose(1, 2)
            x_k = self.pool_k(x).flatten(2).transpose(1, 2)
            x_v = self.pool_v(x).flatten(2).transpose(1, 2)
            
            # Add back class token
            if self.has_cls_embed:
                x_q = torch.cat([cls_token, x_q], dim=1)
                x_k = torch.cat([cls_token, x_k], dim=1)
                x_v = torch.cat([cls_token, x_v], dim=1)
                
            # Project q, k, v
            q = self.q(x_q).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k(x_k).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x_v).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            # Project q, k, v first
            q = self.q(x).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k(x).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # For q, k, v add back class token, reshape to images, apply pooling
            if self.has_cls_embed:
                q_cls, q = torch.tensor_split(q, [1], dim=2)
                k_cls, k = torch.tensor_split(k, [1], dim=2)
                v_cls, v = torch.tensor_split(v, [1], dim=2)
                
            # Reshape to image-like format for pooling: B, num_heads, HW, head_dim -> B*num_heads, head_dim, H, W
            q = q.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
            k = k.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
            v = v.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
            
            # Apply pooling
            q = self.pool_q(q).reshape(B, self.num_heads, -1, self.head_dim)
            k = self.pool_k(k).reshape(B, self.num_heads, -1, self.head_dim)
            v = self.pool_v(v).reshape(B, self.num_heads, -1, self.head_dim)
            
            # Add class tokens back
            if self.has_cls_embed:
                q = torch.cat([q_cls, q], dim=2)
                k = torch.cat([k_cls, k], dim=2)
                v = torch.cat([v_cls, v], dim=2)
                
        # Calculate new hw shapes after pooling
        q_hw = (H + 2 * (self.kernel_q[0] // 2) - self.kernel_q[0]) // self.stride_q[0] + 1, \
               (W + 2 * (self.kernel_q[1] // 2) - self.kernel_q[1]) // self.stride_q[1] + 1
        k_hw = (H + 2 * (self.kernel_kv[0] // 2) - self.kernel_kv[0]) // self.stride_kv[0] + 1, \
               (W + 2 * (self.kernel_kv[1] // 2) - self.kernel_kv[1]) // self.stride_kv[1] + 1
               
        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative positional embeddings if needed
        if self.rel_pos_spatial:
            attn = calc_rel_pos_spatial(
                attn, q, self.has_cls_embed, q_hw, k_hw, self.rel_pos_h, self.rel_pos_w
            )
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Update hw shape for return
        if self.has_cls_embed:
            hw_shape = q_hw
        else:
            hw_shape = q_hw
            
        return x, hw_shape

class Block(nn.Module):
    """
    Basic transformer block.
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        residual=False,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        self.has_cls_embed = has_cls_embed
        self.residual = residual
        
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            residual=residual,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        
    def forward(self, x, attention_mask=None, height=None, width=None):
        y, attn = self.attn(self.norm1(x), attention_mask=attention_mask, height=height, width=width)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn, (height, width), attention_mask

class MultiScaleBlock(nn.Module):
    """
    Multiscale Transformer Block from MVITv2.
    """
    def __init__(
        self,
        dim_out,
        input_size,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        mode="conv",
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
        dim_mul_in_att=False,
        use_mlp=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.dim_mul_in_att = dim_mul_in_att
        self.num_heads = num_heads
        self.has_cls_embed = has_cls_embed
        self.use_mlp = use_mlp
        
        # First normalization layer
        self.norm1 = norm_layer(dim)
        
        # Attention module
        attn_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim=dim,
            dim_out=attn_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Handle dimension change - projection after attention if needed
        if dim != dim_out and not dim_mul_in_att:
            self.proj = nn.Linear(dim, dim_out)
        else:
            self.proj = nn.Identity()
            
        # Second normalization and MLP
        self.norm2 = norm_layer(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        
        if use_mlp:
            self.mlp = Mlp(
                in_features=dim_out,
                hidden_features=mlp_hidden_dim,
                out_features=dim_out,
                act_layer=act_layer,
                drop=drop,
            )
        else:
            self.mlp = nn.Identity()
            
    def forward(self, x, hw_shape):
        # Apply attention
        x_norm = self.norm1(x)
        x_attn, hw_shape_new = self.attn(x_norm, hw_shape)
        
        # Handle shortcut/residual connection
        if self.dim != self.dim_out and not self.dim_mul_in_att:
            x = self.proj(x)
            
        x = x + self.drop_path(x_attn)
        
        # Apply MLP if needed
        if self.use_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            
        return x, hw_shape_new

class GlobalMaxPoolLayer(nn.Module):
    """
    Global max pooling layer that can handle class tokens.
    Equivalent to the TensorFlow GlobalMaxPoolLayer.
    """
    def __init__(self, use_class_token=False):
        super().__init__()
        self.use_class_token = use_class_token
        
    def forward(self, x, attention_mask=None, training=False, h=0, w=0):
        # If we have class tokens, separate them
        if self.use_class_token:
            cls_token, x = torch.tensor_split(x, [1], dim=1)
        else:
            cls_token = None
        
        # Apply max pooling along the sequence dimension
        if attention_mask is not None:
            # If we have an attention mask, use it to create a very negative value
            # for masked positions so they don't contribute to the max
            mask_value = -10000.0  # Large negative value
            mask_expanded = attention_mask.unsqueeze(-1)
            x_masked = x * mask_expanded + (1.0 - mask_expanded) * mask_value
            x = torch.max(x_masked, dim=1, keepdim=True)[0]
        else:
            x = torch.max(x, dim=1, keepdim=True)[0]
        
        # Add back class token if needed
        if self.use_class_token:
            x = torch.cat([cls_token, x], dim=1)
            
        # Return with placeholders for compatibility with Block output format
        return x, None, (h, w), attention_mask

class SinusoidalPositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding.
    Based on the TensorFlow implementation but for PyTorch.
    """
    def __init__(self, channels, height=16, width=16):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [B, H*W, C] or [B, H*W+1, C] if class token is present
            
        Returns:
            Position encoded tensor of same shape
        """
        # Handle class token if present - don't add positional encoding to it
        if inputs.shape[1] > self.height * self.width:
            cls_token = inputs[:, 0:1, :]
            x = inputs[:, 1:, :]
        else:
            cls_token = None
            x = inputs
            
        batch_size, seq_len, channels = x.shape
        
        # Reshape to [B, H, W, C]
        x = x.view(batch_size, self.height, self.width, channels)
        
        # Create position indices
        y_pos = torch.arange(self.height, device=x.device).float()
        x_pos = torch.arange(self.width, device=x.device).float()
        
        # Calculate frequency bands
        div_term = torch.exp(torch.arange(0, channels // 4, 2, device=x.device).float() * (-math.log(10000.0) / (channels // 4)))
        
        # Calculate sin and cos for both dimensions
        pos_y_sin = torch.sin(y_pos.unsqueeze(-1) * div_term).unsqueeze(-1).expand(-1, -1, channels // 4)
        pos_y_cos = torch.cos(y_pos.unsqueeze(-1) * div_term).unsqueeze(-1).expand(-1, -1, channels // 4)
        pos_x_sin = torch.sin(x_pos.unsqueeze(-1) * div_term).unsqueeze(-1).expand(-1, -1, channels // 4)
        pos_x_cos = torch.cos(x_pos.unsqueeze(-1) * div_term).unsqueeze(-1).expand(-1, -1, channels // 4)
        
        # Combine the encodings
        pos_enc = torch.cat([pos_y_sin, pos_y_cos, pos_x_sin, pos_x_cos], dim=-1)
        
        # Add the positional encoding to the input
        x = x + pos_enc.unsqueeze(0)
        
        # Reshape back to sequence
        x = x.view(batch_size, -1, channels)
        
        # Add class token back if it was present
        if cls_token is not None:
            x = torch.cat([cls_token, x], dim=1)
            
        return x
