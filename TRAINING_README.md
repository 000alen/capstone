# Secure Transformer Training Guide

This guide explains how to train and evaluate the IND-CPA-secure SO(N)-equivariant transformer model on WikiText-103 dataset.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
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

## 📁 Project Structure

```
capstone/
├── secure_transformer/
│   ├── model.py          # Core model implementation
│   ├── train.py          # Training script with full features
│   ├── evaluate.py       # Evaluation and analysis script
│   └── utils.py          # Utility functions
├── configs/
│   ├── base_config.yaml  # Standard configuration
│   ├── small_config.yaml # Quick experimentation
│   └── large_config.yaml # Production scale
├── train_launcher.py     # Convenient training launcher
└── requirements.txt      # Dependencies
```

## 🔧 Configuration Files

### Model Architecture Parameters

- **d_signal**: Dimension of real signal coordinates
- **m_noise**: Dimension of noise coordinates (for security)
- **k_vec**: Vector dimension per token
- **layers**: Number of transformer layers
- **heads**: Number of attention heads
- **rank**: Lie algebra rank for residual connections
- **sigma**: Noise scale for security

### Available Configs

| Config | Use Case | Model Size | Training Time |
|--------|----------|------------|---------------|
| `small_config.yaml` | Testing/Development | ~2M params | ~2 hours |
| `base_config.yaml` | Standard Training | ~15M params | ~12 hours |
| `large_config.yaml` | Production Scale | ~100M params | ~48 hours |

## 🚂 Training Scripts

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

## 📊 Monitoring Training

### Weights & Biases Dashboard

Training automatically logs to W&B with the following metrics:

**Training Metrics:**
- `train_loss`: Cross-entropy loss
- `train_perplexity`: Training perplexity  
- `grad_norm`: Gradient norm (for monitoring stability)
- `learning_rate`: Current learning rate

**Validation Metrics:**
- `val_loss`: Validation cross-entropy loss
- `val_perplexity`: Validation perplexity

**Model Metrics:**
- Parameter histograms and gradients
- Layer-wise statistics
- Security-related metrics

### Local Monitoring

Check training progress locally:

```bash
# View latest logs
tail -f checkpoints/training.log

# Monitor GPU usage
nvidia-smi -l 1
```

## 💾 Checkpointing

### Automatic Checkpointing

The training script automatically saves:

- **Latest checkpoint**: `checkpoints/latest_checkpoint.pt` (every epoch)
- **Best checkpoint**: `checkpoints/best_checkpoint.pt` (lowest validation loss)
- **Periodic checkpoints**: `checkpoints/checkpoint_step_N.pt` (every `save_interval` steps)

### Checkpoint Contents

Each checkpoint contains:
- Model state dictionary
- Optimizer state dictionary  
- Learning rate scheduler state
- Training configuration
- Training step and epoch
- Best validation loss

### Resuming Training

```bash
# Resume from latest checkpoint
python train_launcher.py --config configs/base_config.yaml --resume checkpoints/latest_checkpoint.pt

# Resume from specific checkpoint
python train_launcher.py --config configs/base_config.yaml --resume checkpoints/checkpoint_step_5000.pt

# Resume from best checkpoint
python train_launcher.py --config configs/base_config.yaml --resume checkpoints/best_checkpoint.pt
```

## 🔍 Evaluation

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

### Evaluation Options

- `--eval_split`: Dataset split to evaluate (`test` or `validation`)
- `--generate_samples`: Generate text samples
- `--num_samples`: Number of text samples to generate
- `--analyze_security`: Analyze security properties
- `--output_file`: Save detailed results to JSON file

### Evaluation Outputs

**Dataset Metrics:**
- Perplexity on test/validation set
- Cross-entropy loss
- Token-level accuracy

**Text Generation:**
- Sample generations from various prompts
- Quality assessment of generated text

**Security Analysis:**
- Noise variance statistics
- Signal-to-noise ratio analysis
- Rotation matrix properties (determinant, orthogonality)
- Information leakage analysis

**Model Analysis:**
- Layer-wise parameter statistics
- Embedding analysis
- Attention pattern analysis

## 🔐 Security Properties

### What Makes This Model Secure?

1. **Noise Injection**: Each token gets `m_noise` additional random coordinates
2. **Random Rotation**: Fresh SO(N) rotation matrix for each prompt
3. **Equivariant Processing**: Server cannot distinguish signal from noise
4. **Information-Theoretic Security**: Provides IND-CPA security up to 2^λ queries

### Monitoring Security

The evaluation script provides security metrics:

```bash
python -m secure_transformer.evaluate --checkpoint model.pt --analyze_security
```

**Key Security Metrics:**
- **Noise Variance**: Should be consistent with σ²
- **Rotation Determinant**: Should be exactly 1.0 (SO(N) property)  
- **Orthogonality Error**: Should be near 0 (rotation quality)
- **Signal-to-Noise Ratio**: Controlled by d_signal/m_noise ratio

## 🎯 Hyperparameter Tuning

### Key Hyperparameters

**Model Architecture:**
- `d_signal/m_noise`: Controls security vs. efficiency trade-off
- `k_vec`: Vector dimension (affects model capacity)
- `layers/heads`: Standard transformer scaling

**Training:**
- `learning_rate`: Start with 3e-4, reduce for large models
- `batch_size`: Scale with GPU memory
- `warmup_steps`: ~2000 for stable training

**Security:**
- `sigma`: Noise scale (typically 1.0)
- Higher `m_noise` = more security but slower training

### Tuning Strategies

1. **Start Small**: Use `small_config.yaml` for rapid iteration
2. **Security vs. Performance**: Balance `d_signal` and `m_noise`
3. **Scale Gradually**: Increase model size once baseline works
4. **Monitor Convergence**: Watch validation perplexity and gradients

## 🚨 Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Reduce batch size
python train_launcher.py --config configs/base_config.yaml --batch_size 16

# Use smaller model
python train_launcher.py --config configs/small_config.yaml
```

**Training Instability:**
```bash
# Reduce learning rate
python train_launcher.py --config configs/base_config.yaml --learning_rate 1e-4

# Check gradient clipping in config
```

**Slow Training:**
```bash
# Reduce sequence length and increase batch size
# Reduce num_workers if CPU bottleneck
# Check GPU utilization with nvidia-smi
```

**Poor Convergence:**
```bash
# Increase warmup steps
# Check data loading and tokenization
# Verify model equivariance with tests
```

### Getting Help

1. **Check Logs**: Training logs contain detailed error information
2. **Run Tests**: Use `pytest secure_transformer/tests/` to verify model correctness
3. **Monitor W&B**: Check for gradient/loss anomalies
4. **Check Equivariance**: Run equivariance tests to verify model correctness

## 📈 Performance Expectations

### Training Speed

| Model Size | GPU | Batch Size | Tokens/sec | Time to Convergence |
|------------|-----|------------|------------|-------------------|
| Small (2M) | V100 | 16 | ~8K | 2 hours |
| Base (15M) | V100 | 32 | ~4K | 12 hours |
| Large (100M) | A100 | 64 | ~8K | 48 hours |

### Perplexity Targets

| Model | WikiText-103 PPL | Training Loss |
|-------|-----------------|---------------|
| Small | ~80-100 | ~4.4 |
| Base | ~50-70 | ~4.0 |
| Large | ~30-50 | ~3.5 |

*Note: Secure models typically have 10-20% higher perplexity than standard transformers due to noise injection.*

## 🎉 Next Steps

1. **Start Training**: Begin with `small_config.yaml` for testing
2. **Monitor Progress**: Use W&B dashboard to track training
3. **Evaluate Model**: Run comprehensive evaluation on trained model
4. **Analyze Security**: Verify security properties are maintained
5. **Scale Up**: Move to larger configurations for production use

For more details on the model architecture and security properties, see the main model documentation and the equivariance tests. 