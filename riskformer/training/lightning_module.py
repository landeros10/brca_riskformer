import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Union, List
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.optim import Adam, SGD, AdamW

from riskformer.training.riskformer_vit import RiskFormer_ViT
from riskformer.utils.training_utils import slide_level_loss


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
        class_loss_map: Dict[int, torch.nn.Module],
        regional_coeff: float = 0.0,
    ):
        """
        Initialize the RiskFormer Lightning Module.
        
        Args:
            model_config: Configuration for the RiskFormer_ViT model
            optimizer_config: Configuration for the optimizer
            class_loss_map: Dictionary mapping class indices to loss functions
            regional_coeff: Coefficient for weighting local vs global loss
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the model
        self.model = RiskFormer_ViT(**model_config)
        
        # Store configurations
        self.optimizer_config = optimizer_config
        self.class_loss_map = class_loss_map
        self.regional_coeff = regional_coeff
        
        # Metrics
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        
    def forward(self, x, training=False, return_weights=False, return_gradcam=False):
        """Forward pass through the model."""
        return self.model(x, training, return_weights, return_gradcam)
    
    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        x, metadata = batch
        predictions = self(x, training=True)
        labels = metadata['label']
        
        loss = slide_level_loss(
            predictions, 
            labels, 
            self.class_loss_map, 
            regional_coeff=self.regional_coeff
        )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy for binary classification
        if predictions.shape[1] == 1:
            preds = (predictions > 0.5).float()
            acc = self.train_acc(preds, labels)
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning."""
        x, metadata = batch
        predictions = self(x, training=False)
        labels = metadata['label']
        
        loss = slide_level_loss(
            predictions, 
            labels, 
            self.class_loss_map, 
            regional_coeff=self.regional_coeff
        )
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy for binary classification
        if predictions.shape[1] == 1:
            preds = (predictions > 0.5).float()
            acc = self.val_acc(preds, labels)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for Lightning."""
        x, metadata = batch
        predictions = self(x, training=False)
        labels = metadata['label']
        
        loss = slide_level_loss(
            predictions, 
            labels, 
            self.class_loss_map, 
            regional_coeff=self.regional_coeff
        )
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Calculate accuracy for binary classification
        if predictions.shape[1] == 1:
            preds = (predictions > 0.5).float()
            acc = self.test_acc(preds, labels)
            self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
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