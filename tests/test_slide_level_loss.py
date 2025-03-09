import pytest
import torch
import torch.nn as nn
from riskformer.utils.training_utils import slide_level_loss

class TestSlideLevelLoss:
    """
    Tests for the slide_level_loss function, focusing on multi-task learning
    and handling of different task types.
    """
    
    @pytest.fixture
    def binary_pred_single_instance(self):
        """Create a single binary prediction."""
        # Shape: [1, 1] (batch_size=1, num_classes=1)
        return torch.tensor([[0.7]], dtype=torch.float32)
    
    @pytest.fixture
    def binary_pred_multi_instance(self):
        """Create binary predictions with multiple instance predictions."""
        # Shape: [5, 1] (5 instances, num_classes=1)
        # First prediction is global, rest are instance-level
        return torch.tensor([
            [0.7],  # Global prediction
            [0.8],  # Instance 1 (strong positive)
            [0.6],  # Instance 2 (moderate positive)
            [0.3],  # Instance 3 (moderate negative)
            [0.2],  # Instance 4 (strong negative)
        ], dtype=torch.float32)
    
    @pytest.fixture
    def multiclass_pred_single_instance(self):
        """Create a single multiclass prediction."""
        # Shape: [1, 3] (batch_size=1, num_classes=3)
        return torch.tensor([[0.2, 0.7, 0.1]], dtype=torch.float32)
    
    @pytest.fixture
    def multiclass_pred_multi_instance(self):
        """Create multiclass predictions with multiple instance predictions."""
        # Shape: [5, 3] (5 instances, num_classes=3)
        return torch.tensor([
            [0.2, 0.7, 0.1],  # Global prediction
            [0.1, 0.8, 0.1],  # Instance 1
            [0.3, 0.6, 0.1],  # Instance 2
            [0.7, 0.2, 0.1],  # Instance 3
            [0.6, 0.3, 0.1],  # Instance 4
        ], dtype=torch.float32)
    
    @pytest.fixture
    def multitask_pred_single_instance(self):
        """Create a single prediction for multiple tasks."""
        # Shape: [1, 3] (batch_size=1, 3 task outputs)
        # First output for binary task, second for regression, third for multiclass
        return torch.tensor([[0.7, 42.5, 0.8]], dtype=torch.float32)
    
    @pytest.fixture
    def multitask_pred_multi_instance(self):
        """Create multi-task predictions with multiple instance predictions."""
        # Shape: [5, 3] (5 instances, 3 task outputs)
        return torch.tensor([
            [0.7, 42.5, 0.8],  # Global prediction
            [0.8, 45.0, 0.7],  # Instance 1
            [0.6, 40.0, 0.6],  # Instance 2
            [0.3, 35.0, 0.3],  # Instance 3
            [0.2, 30.0, 0.2],  # Instance 4
        ], dtype=torch.float32)
    
    def test_binary_classification_single_instance(self, binary_pred_single_instance):
        """Test binary classification loss with a single instance."""
        # Binary label (positive class)
        label = torch.tensor([1.0], dtype=torch.float32)
        
        # Loss function for binary classification
        loss_fn = nn.BCEWithLogitsLoss()
        class_loss_map = {0: loss_fn}
        
        # Calculate loss
        loss = slide_level_loss(binary_pred_single_instance, label, class_loss_map)
        
        # Expected loss
        expected_loss = loss_fn(binary_pred_single_instance[0], label)
        
        # Check that loss matches expected value
        assert torch.allclose(loss, expected_loss), "Binary classification loss doesn't match expected value"
    
    def test_binary_classification_multi_instance(self, binary_pred_multi_instance):
        """Test binary classification loss with multiple instances and regional coefficient."""
        # Binary label (positive class)
        label = torch.tensor([1.0], dtype=torch.float32)
        
        # Loss function for binary classification
        loss_fn = nn.BCEWithLogitsLoss()
        class_loss_map = {0: loss_fn}
        
        # Regional coefficient (50% weight to local loss)
        regional_coeff = 0.5
        
        # Calculate loss
        loss = slide_level_loss(binary_pred_multi_instance, label, class_loss_map, regional_coeff=regional_coeff)
        
        # Expected global loss
        global_loss = loss_fn(binary_pred_multi_instance[0], label) * (1 - regional_coeff)
        
        # Expected local loss (top-k = 1 since total_instances // 10 = 0, but min is 1)
        # The top prediction is at index 1 with value 0.8
        top_k_pred = binary_pred_multi_instance[1].unsqueeze(0)  # Shape: [1, 1]
        local_loss = loss_fn(top_k_pred.squeeze(1), label.expand(1)) * regional_coeff
        
        # Total expected loss
        expected_loss = global_loss + local_loss
        
        # Check that loss matches expected value
        assert torch.allclose(loss, expected_loss), "Binary classification with regional loss doesn't match expected value"
    
    def test_multiclass_classification(self, multiclass_pred_single_instance):
        """Test multiclass classification loss."""
        # Create a prediction with proper shape [batch_size, num_classes]
        # We'll use the existing fixture but ensure it has the right shape for CrossEntropyLoss
        pred = multiclass_pred_single_instance  # Shape is already [1, 3]
        
        # Multiclass label - index 1 (second class), which is within range for a 3-class problem
        label = torch.tensor([1], dtype=torch.long)
        
        # Loss function for multiclass classification
        loss_fn = nn.CrossEntropyLoss()
        
        # Create a loss map with a single entry
        class_loss_map = {0: loss_fn}
        
        # Create a dictionary-based prediction for slide_level_loss
        dict_predictions = {'multiclass': pred}
        dict_labels = {'multiclass': label}
        
        # Define task types
        task_types = {'multiclass': 'multiclass'}
        
        # Calculate loss using dictionary approach
        loss = slide_level_loss(dict_predictions, dict_labels, 
                               {'multiclass': {0: loss_fn}}, 
                               task_types=task_types)
        
        # Expected loss - direct calculation with properly shaped tensors
        expected_loss = loss_fn(pred, label)
        
        # Check that loss matches expected value
        assert abs(loss.item() - expected_loss.item()) < 1e-6, "Loss does not match expected value"
    
    def test_multitask_learning(self, multitask_pred_single_instance):
        """Test multi-task learning with different task types."""
        # Labels for different tasks
        binary_label = torch.tensor([1.0], dtype=torch.float32)
        regression_label = torch.tensor([45.0], dtype=torch.float32)
        # For multiclass, need to make sure indices are within range (only 0 or 1 for binary)
        multiclass_label = torch.tensor([0], dtype=torch.long)
        
        # Create a dictionary for all labels
        labels = {
            'binary': binary_label,
            'regression': regression_label,
            'multiclass': multiclass_label
        }
        
        # Loss functions for different tasks
        binary_loss_fn = nn.BCEWithLogitsLoss()
        regression_loss_fn = nn.MSELoss()
        multiclass_loss_fn = nn.CrossEntropyLoss()
        
        # Create a dictionary mapping task names to loss functions
        class_loss_map = {
            'binary': {0: binary_loss_fn},
            'regression': {0: regression_loss_fn},
            'multiclass': {0: multiclass_loss_fn}
        }
        
        # Task types
        task_types = {
            'binary': 'binary',
            'regression': 'regression',
            'multiclass': 'multiclass'
        }
        
        # We'll use dictionary-based predictions to match the expected format
        # Extract predictions for each task from the multitask_pred tensor
        dict_predictions = {
            # BCEWithLogitsLoss needs shape compatibility with label
            'binary': torch.tensor([multitask_pred_single_instance[0, 0]]),  # Shape: [1]
            'regression': torch.tensor([multitask_pred_single_instance[0, 1]]),  # Shape: [1]
            'multiclass': torch.tensor([[1.0 - multitask_pred_single_instance[0, 2].item(), 
                                         multitask_pred_single_instance[0, 2].item()]])  # Shape: [1, 2] for binary classification
        }
        
        # Calculate loss using our dictionary-based format
        loss = slide_level_loss(dict_predictions, labels, class_loss_map, task_types=task_types)
        
        # Calculate expected losses for each task directly with loss functions
        binary_expected_loss = binary_loss_fn(dict_predictions['binary'], binary_label)
        regression_expected_loss = regression_loss_fn(dict_predictions['regression'], regression_label)
        multiclass_expected_loss = multiclass_loss_fn(dict_predictions['multiclass'], multiclass_label)
        
        # Total expected loss
        expected_loss = binary_expected_loss + regression_expected_loss + multiclass_expected_loss
        
        # Check that loss is approximately equal with a reasonable tolerance
        # The tolerance needs to be high enough to account for any numerical differences
        # in how losses are calculated inside slide_level_loss
        assert abs(loss.item() - expected_loss.item()) < 1e-5, "Multi-task learning loss is too different from expected value"
    
    def test_regional_coefficient_effect(self, binary_pred_multi_instance):
        """Test that regional coefficient properly balances global vs local loss."""
        # Binary label (positive class)
        label = torch.tensor([1.0], dtype=torch.float32)
        
        # Loss function for binary classification
        loss_fn = nn.BCEWithLogitsLoss()
        class_loss_map = {0: loss_fn}
        
        # Calculate loss with different regional coefficients
        loss_coeff_0 = slide_level_loss(binary_pred_multi_instance, label, class_loss_map, regional_coeff=0.0)
        loss_coeff_05 = slide_level_loss(binary_pred_multi_instance, label, class_loss_map, regional_coeff=0.5)
        loss_coeff_1 = slide_level_loss(binary_pred_multi_instance, label, class_loss_map, regional_coeff=1.0)
        
        # Expected losses
        global_loss = loss_fn(binary_pred_multi_instance[0], label)
        
        # With regional_coeff=0.0, we should only have the global loss
        assert torch.allclose(loss_coeff_0, global_loss), "With regional_coeff=0, loss should equal global loss"
        
        # With regional_coeff=1.0, we should only have the local loss
        assert not torch.allclose(loss_coeff_1, global_loss), "With regional_coeff=1, loss should not equal global loss"
        
        # With regional_coeff=0.5, we should be between global and local loss
        assert loss_coeff_0 != loss_coeff_05 != loss_coeff_1, "Loss should change with different regional coefficients"
    
    def test_multitask_regional_effect(self, multitask_pred_multi_instance):
        """Test multi-task learning with regional coefficient."""
        # Labels for different tasks - we need to properly format for use with regional coefficient
        num_instances = multitask_pred_multi_instance.shape[0]
        
        # For binary loss with regional coefficient, we need to expand the target to match all instances
        labels = {
            'binary': torch.tensor([1.0]).expand(num_instances),  # Shape: [num_instances]
            'regression': torch.tensor([45.0]).expand(num_instances),  # Shape: [num_instances]
            'multiclass': torch.tensor([0]).expand(num_instances)  # Shape: [num_instances]
        }
        
        # Loss functions for different tasks
        class_loss_map = {
            'binary': {0: nn.BCEWithLogitsLoss(reduction='none')},  # Use 'none' to apply per-instance
            'regression': {0: nn.MSELoss(reduction='none')},  # Use 'none' to apply per-instance
            'multiclass': {0: nn.CrossEntropyLoss(reduction='none')}  # Use 'none' to apply per-instance
        }
        
        # Task types
        task_types = {
            'binary': 'binary',
            'regression': 'regression',
            'multiclass': 'multiclass'
        }
        
        # Create dictionary-based predictions for slide_level_loss
        # For each task, we need a tensor that includes all instances
        binary_values = multitask_pred_multi_instance[:, 0]  # Shape: [num_instances]
        regression_values = multitask_pred_multi_instance[:, 1]  # Shape: [num_instances]
        
        # For multiclass, we need to create a tensor with shape [num_instances, num_classes]
        multiclass_values = torch.zeros(num_instances, 2)  # 2 classes for binary classification
        for i in range(num_instances):
            multiclass_values[i, 0] = 1.0 - multitask_pred_multi_instance[i, 2]  # Probability of class 0
            multiclass_values[i, 1] = multitask_pred_multi_instance[i, 2]  # Probability of class 1
        
        # Create the prediction dictionary
        dict_predictions = {
            'binary': binary_values,  # Shape: [num_instances]
            'regression': regression_values,  # Shape: [num_instances]
            'multiclass': multiclass_values  # Shape: [num_instances, 2]
        }
        
        # Calculate loss with different regional coefficients
        # We're only testing the relative difference here, not absolute values
        loss_coeff_0 = slide_level_loss(dict_predictions, labels, class_loss_map,
                                        regional_coeff=0.0, task_types=task_types)
        loss_coeff_05 = slide_level_loss(dict_predictions, labels, class_loss_map,
                                         regional_coeff=0.5, task_types=task_types)
        loss_coeff_1 = slide_level_loss(dict_predictions, labels, class_loss_map,
                                        regional_coeff=1.0, task_types=task_types)
        
        # Check that regional coefficient has the expected effect
        # With regional_coeff=0, only the global prediction (first instance) is used
        # With regional_coeff=1, only the regional predictions (other instances) are used
        # With regional_coeff=0.5, both global and regional predictions contribute equally
        
        # Verify that middle value falls between the extremes
        assert loss_coeff_0 != loss_coeff_1, "Regional coefficient has no effect"
        assert min(loss_coeff_0.item(), loss_coeff_1.item()) <= loss_coeff_05.item() <= max(loss_coeff_0.item(), loss_coeff_1.item()), \
            "Regional coefficient doesn't have the expected effect on loss values" 