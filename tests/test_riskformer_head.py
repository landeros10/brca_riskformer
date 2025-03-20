import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from riskformer.training.model import RiskFormer_Head

class TestRiskFormerHead:
    """Tests for the RiskFormer_Head class."""
    
    @pytest.fixture
    def embed_dim(self):
        """Return a standard embedding dimension for testing."""
        return 64
    
    @pytest.fixture
    def batch_size(self):
        """Return a standard batch size for testing."""
        return 4
    
    @pytest.fixture
    def tasks_config(self):
        """Return a dictionary of task configurations for testing."""
        return {
            "binary_task": {
                "type": "binary",
                "num_classes": 1,
                "activation": "sigmoid"
            },
            "regression_task": {
                "type": "regression",
                "num_classes": 1,
                "activation": None
            },
            "multiclass_task": {
                "type": "multiclass",
                "num_classes": 3,
                "activation": "softmax"
            }
        }
    
    @pytest.fixture
    def head_instance(self, tasks_config, embed_dim):
        """Create a RiskFormer_Head instance for testing."""
        return RiskFormer_Head(tasks_config, embed_dim, drop_rate=0.0)
    
    def test_initialization(self, head_instance, tasks_config, embed_dim):
        """Test that the head initializes correctly with task-specific layers."""
        # Check that task heads are created
        assert hasattr(head_instance, 'heads')
        
        # Check that we have the right number of task heads
        assert len(head_instance.heads) == len(tasks_config)
        
        # Check task_indices are created correctly
        assert hasattr(head_instance, 'task_indices')
        assert len(head_instance.task_indices) == len(tasks_config)
        
        # Verify that the total number of output dimensions is correct
        total_classes = sum(config['num_classes'] for config in tasks_config.values())
        
        # Create a dummy input
        x = torch.rand(1, embed_dim)
        outputs = head_instance(x)
        assert outputs.shape[1] == total_classes
    
    def test_forward(self, head_instance, tasks_config, batch_size, embed_dim):
        """Test the forward method of RiskFormer_Head."""
        # Create input tensor
        x = torch.rand(batch_size, embed_dim)
        
        # Forward pass
        outputs = head_instance(x)
        
        # Get total output dimensions
        total_classes = sum(config['num_classes'] for config in tasks_config.values())
        
        # Check output shapes
        assert outputs.shape == (batch_size, total_classes)
        
        # Test output range for binary task (sigmoid activation)
        binary_start, binary_end = head_instance.task_indices["binary_task"]
        binary_outputs = outputs[:, binary_start:binary_end]
        assert torch.all((binary_outputs >= 0) & (binary_outputs <= 1))
        
        # Test multiclass outputs
        multi_start, multi_end = head_instance.task_indices["multiclass_task"]
        multi_outputs = outputs[:, multi_start:multi_end]
        # Check softmax outputs sum to approximately 1
        assert torch.allclose(torch.sum(multi_outputs, dim=1), 
                             torch.ones(batch_size), 
                             atol=1e-6)
    
    def test_get_task_output(self, head_instance, tasks_config, batch_size, embed_dim):
        """Test the get_task_output method for each task type."""
        # Create input tensor
        x = torch.rand(batch_size, embed_dim)
        
        # Test binary task
        binary_out = head_instance.get_task_output(x, "binary_task")
        assert binary_out.shape == (batch_size, 1)
        assert torch.all((binary_out >= 0) & (binary_out <= 1))  # Should be sigmoid output
        
        # Test regression task
        regression_out = head_instance.get_task_output(x, "regression_task")
        assert regression_out.shape == (batch_size, 1)
        
        # Test multiclass task
        multiclass_out = head_instance.get_task_output(x, "multiclass_task")
        assert multiclass_out.shape == (batch_size, 3)
        assert torch.allclose(torch.sum(multiclass_out, dim=1), 
                             torch.ones(batch_size), 
                             atol=1e-6)  # Should sum to 1 (softmax)
    
    def test_head_activation(self, head_instance):
        """Test the _head_activation method with different activation types."""
        # Test None activation
        none_activation = head_instance._head_activation(None)
        assert isinstance(none_activation, nn.Identity)
        
        # Test "None" string activation
        none_str_activation = head_instance._head_activation("None")
        assert isinstance(none_str_activation, nn.Identity)
        
        # Test "tanh" activation
        tanh_activation = head_instance._head_activation("tanh")
        assert isinstance(tanh_activation, nn.Tanh)
        
        # Test "sigmoid" activation
        sigmoid_activation = head_instance._head_activation("sigmoid")
        assert isinstance(sigmoid_activation, nn.Sigmoid)
        
        # Test "softmax" activation
        softmax_activation = head_instance._head_activation("softmax")
        assert isinstance(softmax_activation, nn.Softmax)
        
        # Test invalid activation raises error
        with pytest.raises(ValueError):
            head_instance._head_activation("invalid_activation")
    
    def test_error_handling(self, head_instance, batch_size, embed_dim):
        """Test error handling for invalid task names and types."""
        # Create input tensor
        x = torch.rand(batch_size, embed_dim)
        
        # Test with invalid task name
        with pytest.raises(KeyError):
            head_instance.get_task_output(x, "invalid_task")
        
        # Test with task that has invalid type
        # Create a new head with an invalid task type
        invalid_tasks = {
            "invalid_type_task": {
                "type": "invalid_type",
                "num_classes": 1,
                "activation": None
            }
        }
        
        # Creation should succeed, but forward pass should fail
        head_invalid = RiskFormer_Head(invalid_tasks, embed_dim, drop_rate=0.0)
        
        # No explicit ValueError in the implementation for invalid types 