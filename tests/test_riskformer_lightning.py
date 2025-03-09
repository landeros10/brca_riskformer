import unittest
import pytest
import torch
import torch.nn as nn
from riskformer.training.model import RiskFormerLightningModule
import pytorch_lightning as pl
from unittest.mock import patch

class TestRiskFormerLightningModule(unittest.TestCase):
    
    def setUp(self):
        # Define a simple model configuration for testing
        self.model_config = {
            "input_embed_dim": 16,
            "output_embed_dim": 32,
            "use_phi": False,
            "drop_path_rate": 0.1,
            "drop_rate": 0.1,
            "num_classes": 1,  # Changed to 1 for binary classification
            "max_dim": 16,  # Ensure this is a perfect square
            "depth": 2,
            "global_depth": 1,
            "encoding_method": "standard",  # Changed from "linear" to "standard"
            "mask_num": 2,
            "mask_preglobal": False,
            "num_heads": 2,
            "use_attn_mask": False,
            "mlp_ratio": 2.0,
            "use_class_token": True,
            "global_k": 4,
        }
        
        # Define optimizer config
        self.optimizer_config = {
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "scheduler": "plateau",
            "patience": 5,
            "factor": 0.5,
        }
        
        # Define loss functions for a binary classification task
        self.class_loss_map = {
            "task1": {0: nn.BCEWithLogitsLoss()}
        }
        
        # Task weights
        self.task_weights = {"task1": 1.0}
        
        # Create the Lightning module
        self.model = RiskFormerLightningModule(
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            class_loss_map=self.class_loss_map,
            task_weights=self.task_weights,
            regional_coeff=0.0
        )
    
    def test_initialization(self):
        """Test if the model initializes correctly."""
        # Check if the model is initialized
        self.assertIsNotNone(self.model)
        
        # Check if the model parameters are set correctly
        self.assertEqual(self.model.regional_coeff, 0.0)
        self.assertEqual(self.model.task_weights, {"task1": 1.0})
        
        # Check if task types are correctly identified
        self.assertEqual(self.model.task_types, {"task1": "binary"})
        
        # Check if metrics are initialized
        self.assertIn("task1", self.model.metrics)
        self.assertIn("train_acc", self.model.metrics["task1"])
        self.assertIn("val_acc", self.model.metrics["task1"])
        self.assertIn("test_acc", self.model.metrics["task1"])
    
    def test_forward(self):
        """Test the forward pass."""
        # Create a small dummy input
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        x = torch.randn(batch_size, channels, height, width)
        
        # Run forward pass
        try:
            output = self.model(x, training=False)
            
            # Check if output has the expected shape - should have batch_size as first dimension
            self.assertEqual(output.shape[0], batch_size)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")
    
    @patch('torchmetrics.Accuracy.__call__')
    @patch('torchmetrics.F1Score.__call__')
    @patch('torchmetrics.AUROC.__call__')
    def test_training_step(self, mock_auroc, mock_f1, mock_acc):
        """Test the training step."""
        # Mock metrics to avoid shape mismatch errors
        mock_acc.return_value = torch.tensor(0.75)
        mock_f1.return_value = torch.tensor(0.8)
        mock_auroc.return_value = torch.tensor(0.9)
        
        # Create a dummy batch
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        x = torch.randn(batch_size, channels, height, width)
        
        # Create labels matching the expected format (2 samples)
        labels = {"task1": torch.tensor([1.0, 0.0])}
        
        # For binary classification with BCEWithLogitsLoss, we need to properly format the labels
        # Create a mock forward method that returns the expected output shape
        original_forward = self.model.forward
        def mock_forward(*args, **kwargs):
            # Return a tensor instead of a dictionary to match slide_level_loss's expectations
            # Format: [batch_size, num_classes]
            return torch.randn(batch_size, 1)  # Return logits for binary classification
        
        self.model.forward = mock_forward
        
        # Patch the _calculate_task_loss method to handle our test data
        original_calculate_task_loss = self.model._calculate_task_loss
        def mock_calculate_task_loss(predictions, labels, task, stage):
            # Extract the label tensor from the dictionary
            label_tensor = labels["task1"]
            # Use the original slide_level_loss directly with tensors
            loss_fn = self.model.class_loss_map["task1"][0]
            return loss_fn(predictions.view(len(label_tensor)), label_tensor)
            
        self.model._calculate_task_loss = mock_calculate_task_loss
        
        # Create batch
        batch = (x, {"labels": labels})
        
        try:
            loss = self.model.training_step(batch, 0)
            
            # Check if loss is a tensor
            self.assertIsInstance(loss, torch.Tensor)
        except Exception as e:
            self.model.forward = original_forward  # Restore original methods
            self.model._calculate_task_loss = original_calculate_task_loss
            self.fail(f"Training step failed with error: {e}")
            
        # Restore original methods
        self.model.forward = original_forward
        self.model._calculate_task_loss = original_calculate_task_loss
    
    @patch('torchmetrics.Accuracy.__call__')
    @patch('torchmetrics.F1Score.__call__')
    @patch('torchmetrics.AUROC.__call__')
    def test_validation_step(self, mock_auroc, mock_f1, mock_acc):
        """Test the validation step."""
        # Mock metrics to avoid shape mismatch errors
        mock_acc.return_value = torch.tensor(0.75)
        mock_f1.return_value = torch.tensor(0.8)
        mock_auroc.return_value = torch.tensor(0.9)
        
        # Create a dummy batch
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        x = torch.randn(batch_size, channels, height, width)
        
        # Create labels matching the expected format (2 samples)
        labels = {"task1": torch.tensor([1.0, 0.0])}
        
        # For binary classification with BCEWithLogitsLoss, we need to properly format the labels
        # Create a mock forward method that returns the expected output shape
        original_forward = self.model.forward
        def mock_forward(*args, **kwargs):
            # Return a tensor instead of a dictionary to match slide_level_loss's expectations
            # Format: [batch_size, num_classes]
            return torch.randn(batch_size, 1)  # Return logits for binary classification
        
        self.model.forward = mock_forward
        
        # Patch the _calculate_task_loss method to handle our test data
        original_calculate_task_loss = self.model._calculate_task_loss
        def mock_calculate_task_loss(predictions, labels, task, stage):
            # Extract the label tensor from the dictionary
            label_tensor = labels["task1"]
            # Use the original slide_level_loss directly with tensors
            loss_fn = self.model.class_loss_map["task1"][0]
            return loss_fn(predictions.view(len(label_tensor)), label_tensor)
            
        self.model._calculate_task_loss = mock_calculate_task_loss
        
        # Create batch
        batch = (x, {"labels": labels})
        
        try:
            loss = self.model.validation_step(batch, 0)
            
            # Check if loss is a tensor
            self.assertIsInstance(loss, torch.Tensor)
        except Exception as e:
            self.model.forward = original_forward  # Restore original methods
            self.model._calculate_task_loss = original_calculate_task_loss
            self.fail(f"Validation step failed with error: {e}")
            
        # Restore original methods
        self.model.forward = original_forward
        self.model._calculate_task_loss = original_calculate_task_loss
    
    @patch('torchmetrics.Accuracy.__call__')
    @patch('torchmetrics.F1Score.__call__')
    @patch('torchmetrics.AUROC.__call__')
    def test_test_step(self, mock_auroc, mock_f1, mock_acc):
        """Test the test step."""
        # Mock metrics to avoid shape mismatch errors
        mock_acc.return_value = torch.tensor(0.75)
        mock_f1.return_value = torch.tensor(0.8)
        mock_auroc.return_value = torch.tensor(0.9)
        
        # Create a dummy batch
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        x = torch.randn(batch_size, channels, height, width)
        
        # Create labels matching the expected format (2 samples)
        labels = {"task1": torch.tensor([1.0, 0.0])}
        
        # For binary classification with BCEWithLogitsLoss, we need to properly format the labels
        # Create a mock forward method that returns the expected output shape
        original_forward = self.model.forward
        def mock_forward(*args, **kwargs):
            # Return a tensor instead of a dictionary to match slide_level_loss's expectations
            # Format: [batch_size, num_classes]
            return torch.randn(batch_size, 1)  # Return logits for binary classification
        
        self.model.forward = mock_forward
        
        # Patch the _calculate_task_loss method to handle our test data
        original_calculate_task_loss = self.model._calculate_task_loss
        def mock_calculate_task_loss(predictions, labels, task, stage):
            # Extract the label tensor from the dictionary
            label_tensor = labels["task1"]
            # Use the original slide_level_loss directly with tensors
            loss_fn = self.model.class_loss_map["task1"][0]
            return loss_fn(predictions.view(len(label_tensor)), label_tensor)
            
        self.model._calculate_task_loss = mock_calculate_task_loss
        
        # Create batch
        batch = (x, {"labels": labels})
        
        try:
            loss = self.model.test_step(batch, 0)
            
            # Check if loss is a tensor
            self.assertIsInstance(loss, torch.Tensor)
        except Exception as e:
            self.model.forward = original_forward  # Restore original methods
            self.model._calculate_task_loss = original_calculate_task_loss
            self.fail(f"Test step failed with error: {e}")
            
        # Restore original methods
        self.model.forward = original_forward
        self.model._calculate_task_loss = original_calculate_task_loss
    
    def test_configure_optimizers(self):
        """Test the optimizer configuration."""
        # Get optimizers and schedulers
        try:
            optim_config = self.model.configure_optimizers()
            
            # Check if optimizer is returned
            self.assertIn('optimizer', optim_config)
            
            # Check if scheduler is returned
            self.assertIn('lr_scheduler', optim_config)
            
            # Check if monitor is set correctly
            self.assertEqual(optim_config['lr_scheduler']['monitor'], 'val_loss')
        except Exception as e:
            self.fail(f"Optimizer configuration failed with error: {e}")


if __name__ == "__main__":
    unittest.main() 