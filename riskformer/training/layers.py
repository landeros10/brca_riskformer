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
import numpy as np

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
    Spatial Relative Positional Embeddings.
    This version exactly mimics the TensorFlow implementation.
    
    Args:
        attn: Attention map of shape (B, num_heads, q_N, k_N)
        q: Query tensor of shape (B, num_heads, q_N, head_dim)
        has_cls_embed: Whether there is a class token
        q_shape: Spatial shape of query (H, W)
        k_shape: Spatial shape of key (H, W)
        rel_pos_h: Relative position embedding for height dimension (rel_sp_dim, head_dim)
        rel_pos_w: Relative position embedding for width dimension (rel_sp_dim, head_dim)
    
    Returns:
        attn: Attention tensor with spatial positional bias added
    """
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape
    
    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = float(max(k_h / q_h, 1.0))
    k_h_ratio = float(max(q_h / k_h, 1.0))
    dist_h = (
        torch.arange(q_h, device=q.device).float()[:, None] * q_h_ratio -
        torch.arange(k_h, device=q.device).float()[None, :] * k_h_ratio
    )
    dist_h += float(k_h - 1) * k_h_ratio

    q_w_ratio = float(max(k_w / q_w, 1.0))
    k_w_ratio = float(max(q_w / k_w, 1.0))
    dist_w = (
        torch.arange(q_w, device=q.device).float()[:, None] * q_w_ratio -
        torch.arange(k_w, device=q.device).float()[None, :] * k_w_ratio
    )
    dist_w += float(k_w - 1) * k_w_ratio

    # Gather the relative positions
    Rh = torch.index_select(rel_pos_h, 0, dist_h.long().flatten()).reshape(q_h, k_h, -1)
    Rw = torch.index_select(rel_pos_w, 0, dist_w.long().flatten()).reshape(q_w, k_w, -1)

    B, n_head, q_N, dim = q.shape

    # Extract the spatial (non-class token) part of q
    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    
    # Apply einsum for efficient computation
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    # Extract the spatial part of attention (non-class tokens)
    attn_slice = attn[:, :, sp_idx:, sp_idx:]
    
    # Reshape to 6D for adding positional embeddings
    attn_slice = attn_slice.reshape(B, n_head, q_h, q_w, k_h, k_w)
    
    # Add relative positional embeddings
    attn_slice = attn_slice + rel_h[:, :, :, :, :, None] + rel_w[:, :, :, :, None, :]
    
    # Reshape back to 4D
    attn_slice = attn_slice.reshape(B, n_head, q_h * q_w, k_h * k_w)
    
    # Combine with the class token part of attention if present
    if sp_idx > 0:
        # Concatenate back the class token attention
        attn_with_cls_q = torch.cat([attn[:, :, :sp_idx, sp_idx:], attn_slice], dim=2)
        attn = torch.cat([attn[:, :, :, :sp_idx], attn_with_cls_q], dim=3)
        return attn
    else:
        return attn_slice

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

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

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
        
        # Q, K, V projections - use QKV combined projection if not pool_first
        if self.pool_first:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.qkv = None
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
            self.q = self.k = self.v = None
        
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
            
            self.norm_q = norm_layer(dim) if kernel_q[0] > 1 else None
            
            self.pool_k = nn.Conv2d(
                dim,
                dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim,
            ) if kernel_kv[0] > 1 else nn.Identity()
            
            self.norm_k = norm_layer(dim) if kernel_kv[0] > 1 else None
            
            self.pool_v = nn.Conv2d(
                dim,
                dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim,
            ) if kernel_kv[0] > 1 else nn.Identity()
            
            self.norm_v = norm_layer(dim) if kernel_kv[0] > 1 else None
            
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
            
            self.norm_q = self.norm_k = self.norm_v = None
            
        elif mode == "max":
            self.pool_q = nn.MaxPool2d(
                kernel_q,
                stride=stride_q,
                padding=padding_q,
            ) if kernel_q[0] > 1 else nn.Identity()
            
            self.pool_k = nn.MaxPool2d(
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
            ) if kernel_kv[0] > 1 else nn.Identity()
            
            self.pool_v = nn.MaxPool2d(
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
            ) if kernel_kv[0] > 1 else nn.Identity()
            
            self.norm_q = self.norm_k = self.norm_v = None
            
        else:
            raise NotImplementedError(f"Pooling mode {mode} not supported")
            
        # Relative positional embedding
        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_zero_init = rel_pos_zero_init
        
        if self.rel_pos_spatial:
            # Adjust shape to match TensorFlow implementation
            size = input_size[0]  # Assuming square input
            q_size = size // stride_q[1] if stride_q[1] > 1 else size
            kv_size = size // stride_kv[1] if stride_kv[1] > 1 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1
            
            # Initialize with the same shape as TensorFlow version
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            
            # Initialize weights properly
            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
    
    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        
        if self.pool_first:
            if self.has_cls_embed:
                cls_token, x = torch.tensor_split(x, [1], dim=1)

            # B, N, C -> B, C, H, W (reshape to image-like format for pooling)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # Apply pooling
            x_q = self.pool_q(x)
            x_k = self.pool_k(x)
            x_v = self.pool_v(x)
            
            # Get new shapes after pooling
            q_h, q_w = x_q.shape[2], x_q.shape[3]
            k_h, k_w = x_k.shape[2], x_k.shape[3]
            v_h, v_w = x_v.shape[2], x_v.shape[3]
            
            # Flatten spatial dimensions to tokens (B, C, H, W -> B, N, C)
            x_q = x_q.flatten(2).transpose(1, 2)
            x_k = x_k.flatten(2).transpose(1, 2)
            x_v = x_v.flatten(2).transpose(1, 2)
            
            # Apply normalization if present
            if self.norm_q is not None:
                x_q = self.norm_q(x_q)
            if self.norm_k is not None:
                x_k = self.norm_k(x_k)
            if self.norm_v is not None:
                x_v = self.norm_v(x_v)
            
            # Add back class token if present
            if self.has_cls_embed:
                x_q = torch.cat([cls_token, x_q], dim=1)
                x_k = torch.cat([cls_token, x_k], dim=1)
                x_v = torch.cat([cls_token, x_v], dim=1)
                
            # Project q, k, v through linear layers
            q = self.q(x_q).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k(x_k).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x_v).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # Output shape for hw tracking
            q_shape = (q_h, q_w)
            k_shape = (k_h, k_w)
            
        else:
            # Use combined QKV projection (more efficient)
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, head_dim
            q, k, v = qkv  # split into q, k, v
            
            # Extract class token for special handling during pooling
            if self.has_cls_embed:
                q_cls, q_spatial = torch.tensor_split(q, [1], dim=2)
                k_cls, k_spatial = torch.tensor_split(k, [1], dim=2)
                v_cls, v_spatial = torch.tensor_split(v, [1], dim=2)
                
                # Reshape spatial tokens to image format for pooling
                q_spatial = q_spatial.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
                k_spatial = k_spatial.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
                v_spatial = v_spatial.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
                
                # Apply pooling to spatial tokens
                q_spatial = self.pool_q(q_spatial)
                k_spatial = self.pool_k(k_spatial)
                v_spatial = self.pool_v(v_spatial)
                
                # Get pooled dimensions
                q_h, q_w = q_spatial.shape[2], q_spatial.shape[3]
                k_h, k_w = k_spatial.shape[2], k_spatial.shape[3]
                
                # Reshape back to sequence format
                q_spatial = q_spatial.reshape(B, self.num_heads, -1, self.head_dim)
                k_spatial = k_spatial.reshape(B, self.num_heads, -1, self.head_dim)
                v_spatial = v_spatial.reshape(B, self.num_heads, -1, self.head_dim)
                
                # Apply normalization if needed
                if self.norm_q is not None:
                    q_spatial = q_spatial.transpose(1, 2)
                    q_spatial = self.norm_q(q_spatial)
                    q_spatial = q_spatial.transpose(1, 2)
                
                if self.norm_k is not None:
                    k_spatial = k_spatial.transpose(1, 2)
                    k_spatial = self.norm_k(k_spatial)
                    k_spatial = k_spatial.transpose(1, 2)
                
                if self.norm_v is not None:
                    v_spatial = v_spatial.transpose(1, 2)
                    v_spatial = self.norm_v(v_spatial)
                    v_spatial = v_spatial.transpose(1, 2)
                
                # Recombine with class tokens
                q = torch.cat([q_cls, q_spatial], dim=2)
                k = torch.cat([k_cls, k_spatial], dim=2)
                v = torch.cat([v_cls, v_spatial], dim=2)
                
                # Record shape for positional encoding
                q_shape = (q_h, q_w)
                k_shape = (k_h, k_w)
                
            else:
                # No class token case - reshape to image format
                q = q.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
                k = k.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
                v = v.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
                
                # Apply pooling
                q = self.pool_q(q)
                k = self.pool_k(k)
                v = self.pool_v(v)
                
                # Get pooled dimensions
                q_h, q_w = q.shape[2], q.shape[3]
                k_h, k_w = k.shape[2], k.shape[3]
                
                # Reshape back to attention format
                q = q.reshape(B, self.num_heads, -1, self.head_dim)
                k = k.reshape(B, self.num_heads, -1, self.head_dim)
                v = v.reshape(B, self.num_heads, -1, self.head_dim)
                
                # Apply normalization if needed
                if self.norm_q is not None:
                    q = q.transpose(1, 2)
                    q = self.norm_q(q)
                    q = q.transpose(1, 2)
                
                if self.norm_k is not None:
                    k = k.transpose(1, 2)
                    k = self.norm_k(k)
                    k = k.transpose(1, 2)
                
                if self.norm_v is not None:
                    v = v.transpose(1, 2)
                    v = self.norm_v(v)
                    v = v.transpose(1, 2)
                
                # Record shape for positional encoding
                q_shape = (q_h, q_w)
                k_shape = (k_h, k_w)
                
        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative positional embeddings if needed
        if self.rel_pos_spatial:
            attn = calc_rel_pos_spatial(
                attn, q, self.has_cls_embed, q_shape, k_shape, self.rel_pos_h, self.rel_pos_w
            )
            
        # Apply softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dim_out)
        
        # Apply residual pooling if enabled
        if self.residual_pooling:
            if self.has_cls_embed:
                # Only apply residual to non-class tokens
                q_reshape = q.transpose(1, 2).reshape(B, -1, self.dim_out)
                cls_token, q_spatial = torch.tensor_split(q_reshape, [1], dim=1)
                x_cls, x_spatial = torch.tensor_split(x, [1], dim=1)
                x_spatial = x_spatial + q_spatial
                x = torch.cat([x_cls, x_spatial], dim=1)
            else:
                # Apply residual to all tokens
                x = x + q.transpose(1, 2).reshape(B, -1, self.dim_out)
        
        # Apply output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn, q_shape

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
        
        # Create pool_skip layer for residual path, matching attention's pooling
        self.pool_skip = None
        if np.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(k // 2) for k in kernel_skip]
            
            if mode == "conv":
                self.pool_skip = nn.Conv2d(
                    dim,
                    dim,
                    kernel_skip,
                    stride=stride_skip,
                    padding=padding_skip,
                    groups=dim,
                )
            elif mode == "avg":
                self.pool_skip = nn.AvgPool2d(
                    kernel_skip,
                    stride=stride_skip,
                    padding=padding_skip,
                )
            elif mode == "max":
                self.pool_skip = nn.MaxPool2d(
                    kernel_skip,
                    stride=stride_skip,
                    padding=padding_skip,
                )
            
        # Handle dimension change - projection after attention if needed
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        else:
            self.proj = nn.Identity()
            
        # Second normalization and MLP
        self.norm2 = norm_layer(dim_out if self.dim_mul_in_att else dim)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        
        if use_mlp:
            self.mlp = Mlp(
                in_features=dim_out if self.dim_mul_in_att else dim,
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
        
        # Apply attention with new hw_shape
        x_attn, attn_weights, hw_shape_new = self.attn(x_norm, hw_shape)
        
        # Handle dimension change for dim_mul_in_att case
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
            
        # Apply pooling to x for residual path
        if self.pool_skip is not None:
            x_res, _ = attention_pool(x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed)
        else:
            x_res = x
            
        # Add residual connection
        x = x_res + self.drop_path(x_attn)
        
        # Apply MLP if needed
        x_norm = self.norm2(x)
        
        if self.use_mlp:
            mlp_out = self.mlp(x_norm)
            if not self.dim_mul_in_att and self.dim != self.dim_out:
                x = self.proj(x_norm)
            x = x + self.drop_path(mlp_out)
            
        return x, attn_weights, hw_shape_new

class GlobalMaxPoolLayer(nn.Module):
    """
    Global max pooling layer that can handle class tokens.
    Equivalent to the TensorFlow GlobalMaxPoolLayer.
    """
    def __init__(self, use_class_token=False):
        super().__init__()
        self.use_class_token = use_class_token
        
    def forward(self, x, mask=None):        
        x_max_pooled = torch.max(x, dim=1)[0] # (bs, D)
        if mask is not None:
                x_avg_pooled = torch.mean(x[:, mask, :], dim=1) # (bs, D)
        else:
            x_avg_pooled = torch.mean(x, dim=1) # (bs, D)
        x_pooled = x_max_pooled + x_avg_pooled
        return x_pooled

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
        
        # Pre-compute the inverse frequencies (similar to TF implementation)
        emb_channels = int(2 * math.ceil(channels / 4))  # Ensure even division for sin/cos pairs
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, emb_channels, 2).float() / emb_channels))
        
    def get_emb(self, sin_inp):
        """
        PyTorch equivalent of the TensorFlow get_emb function.
        Gets a base embedding for one dimension with sin and cos intertwined.
        
        Args:
            sin_inp: Tensor of shape [pos_len, freq_len]
            
        Returns:
            Tensor of shape [pos_len, freq_len*2] with sin and cos values interleaved
        """
        # Stack sin and cos values along a new dimension
        emb = torch.stack([torch.sin(sin_inp), torch.cos(sin_inp)], dim=-1)  # [pos_len, freq_len, 2]
        
        # Reshape to flatten the last two dimensions to interleave sin and cos values
        emb = emb.reshape(*emb.shape[:-2], -1)  # [pos_len, freq_len*2]
        
        return emb
    
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
        
        # Get position indices
        pos_x = torch.arange(self.width, device=x.device).float()
        pos_y = torch.arange(self.height, device=x.device).float()
        
        # Move inv_freq to the same device as input
        inv_freq = self.inv_freq.to(x.device)
        
        # Create sine inputs by multiplying positions with frequencies
        # Equivalent to einsum in TF implementation
        sin_inp_x = pos_x.unsqueeze(1) * inv_freq.unsqueeze(0)  # [W, channels//4]
        sin_inp_y = pos_y.unsqueeze(1) * inv_freq.unsqueeze(0)  # [H, channels//4]
        
        # Get embeddings with interleaved sin and cos values (matching TF implementation)
        emb_x = self.get_emb(sin_inp_x)  # [W, channels//2]
        emb_y = self.get_emb(sin_inp_y)  # [H, channels//2]
        
        # Expand dimensions to create 2D grid
        emb_x = emb_x.unsqueeze(0).expand(self.height, -1, -1)  # [H, W, channels//2]
        emb_y = emb_y.unsqueeze(1).expand(-1, self.width, -1)  # [H, W, channels//2]
        
        # Concatenate the x and y embeddings
        emb = torch.cat([emb_x, emb_y], dim=2)  # [H, W, channels]
        
        # Ensure the output has the same channel dimension as the input
        emb = emb[:, :, :channels]
        
        # Add batch dimension and add to input
        # Add positional encoding to input
        x = x + emb.unsqueeze(0)
        
        # Reshape back to sequence form
        x = x.view(batch_size, -1, channels)
        
        # Add class token back if it was present
        if cls_token is not None:
            x = torch.cat([cls_token, x], dim=1)
            
        return x

        
        
