'''
layers.py

Vision Transformer layers for Whole Slide Image processing.
Author: landeros10
Created: 2025-02-05
'''
import torch
import torch.nn as nn

class VisionTransformerWSI(nn.Module):
    """Vision Transformer for Whole Slide Image processing."""
    
    def __init__(
        self,
        input_embed_dim,
        output_embed_dim,
        num_patches,
        max_dim,
        drop_path_rate,
        drop_rate,
        num_classes,
        depth,
        num_heads,
        name=None,
        **kwargs
    ):
        super().__init__()
        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim
        self.num_patches = num_patches
        self.max_dim = max_dim
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.depth = depth
        self.num_heads = num_heads

    def forward(self, x):
        pass

    def get_config(self):
        return {
            'input_embed_dim': self.input_embed_dim,
            'output_embed_dim': self.output_embed_dim,
            'num_patches': self.num_patches,
            'max_dim': self.max_dim,
            'drop_path_rate': self.drop_path_rate,
            'drop_rate': self.drop_rate,
            'num_classes': self.num_classes,
            'depth': self.depth,
            'num_heads': self.num_heads,
        }


class VisionTransformerWSI_256(VisionTransformerWSI):
    """Vision Transformer for Whole Slide Image processing with 256x256 patches."""
    
    def __init__(
        self,
        input_embed_dim,
        phi_dim,
        use_phi,
        output_embed_dim,
        drop_path_rate,
        drop_rate,
        num_classes,
        max_dim,
        depth,
        global_depth,
        encoding_method,
        mask_num,
        mask_preglobal,
        num_heads,
        use_attn_mask,
        mlp_ratio,
        use_class_token,
        use_nystrom,
        num_landmarks,
        global_k,
        downscale_depth=1,
        downscale_multiplier=1.25,
        noise_aug=0.1,
        data_dir=None,
        attnpool_mode="conv",
        name=None,
        **kwargs
    ):
        super().__init__(
            input_embed_dim=input_embed_dim,
            output_embed_dim=output_embed_dim,
            num_patches=256,  # Fixed for 256x256 patches
            max_dim=max_dim,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            num_classes=num_classes,
            depth=depth,
            num_heads=num_heads,
            name=name,
            **kwargs
        )
        self.phi_dim = phi_dim
        self.use_phi = use_phi
        self.global_depth = global_depth
        self.encoding_method = encoding_method
        self.mask_num = mask_num
        self.mask_preglobal = mask_preglobal
        self.use_attn_mask = use_attn_mask
        self.mlp_ratio = mlp_ratio
        self.use_class_token = use_class_token
        self.use_nystrom = use_nystrom
        self.num_landmarks = num_landmarks
        self.global_k = global_k
        self.downscale_depth = downscale_depth
        self.downscale_multiplier = downscale_multiplier
        self.noise_aug = noise_aug
        self.data_dir = data_dir
        self.attnpool_mode = attnpool_mode

    def forward(self, x):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({
            'phi_dim': self.phi_dim,
            'use_phi': self.use_phi,
            'global_depth': self.global_depth,
            'encoding_method': self.encoding_method,
            'mask_num': self.mask_num,
            'mask_preglobal': self.mask_preglobal,
            'use_attn_mask': self.use_attn_mask,
            'mlp_ratio': self.mlp_ratio,
            'use_class_token': self.use_class_token,
            'use_nystrom': self.use_nystrom,
            'num_landmarks': self.num_landmarks,
            'global_k': self.global_k,
            'downscale_depth': self.downscale_depth,
            'downscale_multiplier': self.downscale_multiplier,
            'noise_aug': self.noise_aug,
            'data_dir': self.data_dir,
            'attnpool_mode': self.attnpool_mode,
        })
        return config 