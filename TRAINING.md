# Secure Transformer Training Guide

This guide explains how to train and evaluate the IND-CPA-secure SO(N)-equivariant transformer model on WikiText-103 dataset.

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Login to Weights & Biases

```bash
wandb login
```

### 3. Start Training

```bash
# Quick experiment with small model
python train_launcher.py --config configs/small_config.yaml --experiment_name "my-first-experiment"

# Production training with base configuration
python train_launcher.py --config configs/base_config.yaml --experiment_name "production-run"

# Large scale training
python train_launcher.py --config configs/large_config.yaml --experiment_name "large-scale-experiment"
```

## Configuration Files

### Model Architecture Parameters

- **d_signal**: Dimension of real signal coordinates
- **m_noise**: Dimension of noise coordinates (for security)
- **k_vec**: Vector dimension per token
- **layers**: Number of transformer layers
- **heads**: Number of attention heads
- **rank**: Lie algebra rank for residual connections
- **sigma**: Noise scale for security

## Training Scripts

### Option 1: Using the Launcher (Recommended)

```bash
python train_launcher.py --config configs/base_config.yaml [OPTIONS]
```

**Options:**

- `--experiment_name NAME`: Custom experiment name
- `--resume PATH`: Resume from checkpoint
- `--batch_size N`: Override batch size
- `--learning_rate LR`: Override learning rate
- `--max_epochs N`: Override max epochs
- `--checkpoint_dir DIR`: Override checkpoint directory

**Examples:**

```bash
# Basic training
python train_launcher.py --config configs/base_config.yaml

# Custom experiment name
python train_launcher.py --config configs/base_config.yaml --experiment_name "secure-transformer-v1"

# Resume training
python train_launcher.py --config configs/base_config.yaml --resume checkpoints/latest_checkpoint.pt

# Override hyperparameters
python train_launcher.py --config configs/base_config.yaml --batch_size 64 --learning_rate 1e-4
```

### Option 2: Direct Training Script

```bash
python -m secure_transformer.train [OPTIONS]
```

**Examples:**

```bash
# Basic training with default parameters
python -m secure_transformer.train

# Custom model architecture
python -m secure_transformer.train --d_signal 256 --m_noise 128 --layers 8

# Custom training parameters
python -m secure_transformer.train --batch_size 64 --learning_rate 1e-4 --max_epochs 100
```

## Evaluation

### Basic Evaluation

```bash
python -m secure_transformer.evaluate --checkpoint checkpoints/best_checkpoint.pt
```

### Comprehensive Evaluation

```bash
python -m secure_transformer.evaluate \
    --checkpoint checkpoints/best_checkpoint.pt \
    --eval_split test \
    --generate_samples \
    --num_samples 10 \
    --analyze_security \
    --output_file evaluation_results.json
```
