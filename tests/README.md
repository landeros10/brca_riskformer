# RiskFormer Tests

This directory contains unit and integration tests for the RiskFormer model implementation.

## Test Files

- `test_riskformer_vit.py`: Unit tests for individual components of the RiskFormer_ViT model
- `test_riskformer_integration.py`: Integration tests for the RiskFormer_ViT model with more realistic inputs
- `test_layers.py`: Unit tests specifically for the PyTorch layer implementations in layers.py

## Requirements

To run these tests, you need:
- pytest
- torch
- numpy

## Running the Tests

### Running all tests

```bash
pytest tests/
```

### Running specific test files

```bash
# Run unit tests for RiskFormer_ViT
pytest tests/test_riskformer_vit.py

# Run integration tests
pytest tests/test_riskformer_integration.py

# Run layer tests
pytest tests/test_layers.py
```

### Running specific test cases

```bash
# Run a specific test
pytest tests/test_riskformer_vit.py::test_forward_pass

# Run tests with a specific pattern in their name
pytest -k "augmentation"
```

### Running with verbose output

```bash
pytest -v tests/
```

### Running with test coverage

```bash
pytest --cov=riskformer tests/
```

## Adding New Tests

When adding new tests, follow these principles:

1. **Unit Tests**: Place tests for individual components in `test_riskformer_vit.py` or `test_layers.py`
2. **Integration Tests**: Place end-to-end or integration tests in `test_riskformer_integration.py`
3. **Fixtures**: Use pytest fixtures for common setup and teardown
4. **Parametrization**: Use `@pytest.mark.parametrize` for testing multiple variations
5. **Assertions**: Use specific assertions to clearly indicate test expectations

## Debugging Failed Tests

If a test fails, you can get more detailed output by using:

```bash
pytest tests/ -v --no-header --no-summary -s
```

This disables output capturing (`-s`), adds verbosity (`-v`), and removes headers and summaries for cleaner output.

## Test Structure Principles

- **Independence**: Each test should be independent and not rely on the state of previous tests
- **Clarity**: Test names should clearly indicate what's being tested
- **Coverage**: Aim to test all components and edge cases
- **Consistency**: Maintain consistent structure and style across test files 