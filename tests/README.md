# RiskFormer Tests

This directory contains unit and integration tests for the RiskFormer model implementation.

## Test Organization

The tests are organized into two main directories:

### Unit Tests (`tests/unit/`)
Contains tests for individual components:
- `test_riskformer_vit.py`: Tests for model components
- `test_dataset.py`: Tests for dataset class
- `test_datamodule.py`: Tests for data module
- `test_train.py`: Tests for training functions

### Integration Tests (`tests/integration/`)
Contains end-to-end tests that verify complete workflows:
- `test_riskformer_vit.py`: Model integration tests
- `test_dataset.py`: Dataset integration tests
- `test_datamodule.py`: Data module integration tests
- `test_train.py`: Training workflow tests

## Requirements

- pytest
- torch
- numpy

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test types
pytest tests/unit/
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_riskformer_vit.py

# Run with coverage
pytest --cov=riskformer tests/
```

## Test Guidelines

1. **Unit Tests**: Test individual components in `tests/unit/`
2. **Integration Tests**: Test complete workflows in `tests/integration/`
3. **Fixtures**: Use pytest fixtures for common setup
4. **Documentation**: Include clear docstrings and comments

## Debugging

For detailed test output:
```bash
pytest tests/ -v --no-header --no-summary -s
``` 