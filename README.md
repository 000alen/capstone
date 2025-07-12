# Secure Transformer

[![Tests](https://github.com/000alen/capstone/actions/workflows/tests.yml/badge.svg)](https://github.com/000alen/capstone/actions/workflows/tests.yml)

An IND-CPA-secure SO(N)-equivariant transformer implementation with cryptographic privacy guarantees.

## Features

- **IND-CPA Security**: Information-theoretic security guarantees up to $2^λ$ advantage
- **SO(N) Equivariance**: Mathematically proven equivariance properties maintained throughout
- **Split Computation**: Client-server architecture with encrypted processing
- **WikiText-103 Training**: Complete training pipeline on WikiText-103 dataset

## Quick Start

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run tests
uv run pytest
```

### Training

```bash
# Basic training
uv run python -m secure_transformer.train

# With custom parameters
uv run python -m secure_transformer.train --d_signal 256 --m_noise 128 --batch_size 16
```

### Evaluation

```bash
# Evaluate a trained model
uv run python -m secure_transformer.evaluate --checkpoint ./checkpoints/best_checkpoint.pt

# Generate text samples
uv run python -m secure_transformer.evaluate --checkpoint ./checkpoints/best_checkpoint.pt --generate_samples

# Analyze security properties
uv run python -m secure_transformer.evaluate --checkpoint ./checkpoints/best_checkpoint.pt --analyze_security
```

## Architecture

The secure transformer consists of three main components:

1. **ClientFront**: Embeds tokens, adds noise, and applies rotation
2. **ServerCore**: SO(N)-equivariant processing with attention and Lie algebra layers
3. **ClientBack**: Decrypts and produces final logits
