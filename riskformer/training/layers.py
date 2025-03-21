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


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        Args:
            x: Tensor of shape [B, L, D]
        Returns:
            Tensor of shape [B, L, D]
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


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

    def calc_rel_pos_spatial(
        self,
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
            attn = self.calc_rel_pos_spatial(
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
                
            # Create a parallel pool specifically for the attention mask
            self.pool_attn = nn.MaxPool2d(
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

    def attention_pool(self, x, pool, hw_shape, has_cls_embed=True):
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
            
    def forward(self, x, hw_shape, attn_mask=None):
        # Apply attention
        x_norm = self.norm1(x)
        
        # Apply attention with new hw_shape
        x_attn, attn_weights, hw_shape_new = self.attn(x_norm, hw_shape)
        
        # Handle dimension change for dim_mul_in_att case
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
            
        # Apply pooling to x for residual path
        if self.pool_skip is not None:
            x_res, _ = self.attention_pool(x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed)
            
            # Also apply pooling to attention mask if provided
            if attn_mask is not None:
                # Use the same attention_pool function for the mask
                mask_dtype = attn_mask.dtype
                attn_mask, _ = self.attention_pool(attn_mask.to(torch.float32), self.pool_attn, hw_shape, has_cls_embed=self.has_cls_embed)
                attn_mask = attn_mask.to(mask_dtype)
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
                
        return x, attn_weights, hw_shape_new, attn_mask


class GlobalPoolLayer(nn.Module):
    """
    Global max pooling layer that can handle class tokens.
    Equivalent to the TensorFlow GlobalMaxPoolLayer.
    """
    def __init__(self, pool_method=None):
        super().__init__()
        # validate pool_method
        if pool_method is not None and pool_method not in ["combined", "max", "avg"]:
            raise ValueError("Invalid pool method. Must be one of: 'combined', 'max', 'avg'")
        
        self.pool_method = "combined" if pool_method is None else pool_method
        

    def _pool_max(self, x, mask=None):
        x_max_pooled = torch.max(x, dim=1)[0] # [B, D]
        return x_max_pooled
    
    def _pool_avg(self, x, mask=None):
        if mask is not None:
            mask_float = mask.to(x.dtype)  # [B, L, 1]
            x_sum = torch.sum(x * mask_float, dim=1)  # [B, D]
            mask_sum = torch.sum(mask_float, dim=1).clamp(min=1e-5)  # [B, 1]
            x_avg_pooled = x_sum / mask_sum  # [B, D]
        else:
            x_avg_pooled = torch.mean(x, dim=1) # [B, D]
        return x_avg_pooled

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape [B, L, D]
            mask: Tensor of shape [B, L, 1]
        Returns:
            Tensor of shape [B, D]
        """
        if self.pool_method == "combined":
            x_max_pooled = self._pool_max(x, mask)
            x_avg_pooled = self._pool_avg(x, mask)
            x_pooled = x_max_pooled + x_avg_pooled  
        elif self.pool_method == "max":
            x_pooled = self._pool_max(x, mask)
        elif self.pool_method == "avg":
            x_pooled = self._pool_avg(x, mask)

        return x_pooled # [B, D] representing B region-level features


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding.
    Based on the TensorFlow implementation but for PyTorch.
    """
    def __init__(
        self,
        channels: int,
        height: int = 16,
        width: int = 16,
        use_cls_token: bool = False,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be greater than 0, got {channels}")
        if height <= 0:
            raise ValueError(f"height must be greater than 0, got {height}")
        if width <= 0:
            raise ValueError(f"width must be greater than 0, got {width}")

        self.channels = channels
        self.height = height
        self.width = width
        self.use_cls_token = use_cls_token

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
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape [B, L, C] or [B, L + 1, C] if class token is present
                where the sequence length L is equal to height * width of the token array
            
        Returns:
            Position encoded tensor of same shape as inputs
        """
        # Handle class token if present - don't add positional encoding to it
        if self.use_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=1)
        else:
            cls_token = None
            
        batch_size, _, channels = x.shape

        if channels != self.channels:
            raise ValueError(f"channels must be equal to {self.channels}, got {channels}")
        
        # Reshape to [B, H, W, C]
        try:
            x = x.view(batch_size, self.height, self.width, channels)
        except Exception as e:
            raise ValueError(f"Failed to reshape input tensor to [B, H, W, C]") from e
        
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

        
        
