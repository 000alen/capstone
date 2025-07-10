# Contributing to Secure Transformer

Thank you for your interest in contributing to the Secure Transformer project! This document outlines the development workflow and testing requirements.

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Local Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/capstone.git
   cd capstone
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Verify setup by running tests:
   ```bash
   uv run pytest
   ```

## Testing Requirements

All contributions must pass the comprehensive test suite, which includes:

### Core Test Suites

1. **Equivariance Tests** (`secure_transformer/tests/equivariance.py`)
   - **Critical for security**: These tests verify SO(N)-equivariance properties
   - Test individual layers (LieResidual, IGatedNonlinear, ETokenAttention)
   - Test composed layers (EBlock, ServerCore)
   - Test end-to-end round-trip equivariance
   - Must pass with high precision (errors < 1e-4 for small matrices, < 2e-4 for large)

2. **Cryptographic Tests** (`secure_transformer/tests/crypto.py`)
   - **Critical for security**: Verify IND-CPA security guarantees
   - Monte Carlo tests for cryptographic advantage
   - Must demonstrate security ≈ 0.5 (random guessing)

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test suites
uv run pytest secure_transformer/tests/equivariance.py -v
uv run pytest secure_transformer/tests/crypto.py -v

# Run tests with coverage (if configured)
uv run pytest --cov=secure_transformer

# Run type checking
uv run mypy secure_transformer/
```

## Contribution Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow existing code style and patterns
- Add appropriate type hints
- Update docstrings for new functions/classes

### 3. Add Tests

**Required for all changes:**

- **New model components**: Must include equivariance tests
- **Security-related changes**: Must include cryptographic tests
- **Bug fixes**: Must include regression tests
- **New features**: Must include comprehensive unit tests

### 4. Verify Tests Pass

```bash
# Run the full test suite
uv run pytest

# Verify no type errors
uv run mypy secure_transformer/
```

### 5. Submit Pull Request

- Ensure all tests pass locally
- Write a clear description of changes
- Reference any related issues
- The CI/CD pipeline will automatically run tests

## Code Guidelines

### Security-Critical Code

When modifying security-critical components, extra care is required:

- **ClientFront/ClientBack**: Changes affect encryption/decryption
- **Random number generation**: Must maintain cryptographic security
- **Matrix operations**: Must preserve equivariance properties

### Testing Security Properties

- **Equivariance**: `f(K @ x) ≈ K @ f(x)` for rotation matrices K
- **Round-trip**: `decrypt(encrypt(x)) ≈ x` for all inputs
- **IND-CPA**: Adversary advantage should be ≈ 0.5 (random guessing)

### Common Pitfalls

1. **Noise vs Signal confusion**: Always decrypt before analyzing signal/noise components
2. **Matrix dimension errors**: Ensure consistent tensor shapes throughout pipeline
3. **Numerical precision**: Use appropriate tolerances for floating-point comparisons
4. **Random seed management**: Use fixed seeds in tests for reproducibility

## CI/CD Pipeline

All pull requests trigger automated testing:

### GitHub Actions Workflow

```yaml
# Triggers: pushes to main, PRs to main
# Platform: Ubuntu Latest
# Python: 3.13
# Package Manager: uv
```

### Test Steps

1. **Environment Setup**
   - Install system dependencies
   - Set up Python 3.13
   - Install uv and sync dependencies

2. **Type Checking**
   - Run MyPy on `secure_transformer/`
   - Non-blocking (continues on error)

3. **Core Test Suites**
   - Run all tests with pytest
   - Run equivariance tests (security-critical)
   - Run crypto tests (security-critical)

4. **Import Verification**
   - Test basic module imports
   - Verify package integrity

### Test Failure Investigation

If tests fail in CI:

1. **Check the logs**: GitHub Actions provides detailed output
2. **Reproduce locally**: Use the same commands as CI
3. **Common issues**:
   - Missing dependencies
   - Platform-specific differences
   - Numerical precision on different hardware

## Getting Help

- **General questions**: Open a GitHub issue
- **Security concerns**: Contact maintainers directly
- **Test failures**: Include full error output and system info

## Review Process

1. **Automated checks**: Must pass all CI tests
2. **Security review**: Required for changes to cryptographic components
3. **Code review**: At least one maintainer approval required
4. **Documentation**: Updates must include relevant documentation changes

Thank you for contributing to secure machine learning! 