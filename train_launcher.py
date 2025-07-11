#!/usr/bin/env python3
"""
Convenient launcher script for training the secure transformer.
Uses YAML configuration files for experiment management.

Usage:
    python train_launcher.py --config configs/base_config.yaml
    python train_launcher.py --config configs/base_config.yaml --experiment_name my_experiment
    python train_launcher.py --config configs/base_config.yaml --resume checkpoints/latest_checkpoint.pt
    python train_launcher.py --config configs/base_config.yaml --count_params
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add the secure_transformer module to the path
sys.path.append(str(Path(__file__).parent))

from secure_transformer.train import Trainer, TrainingConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def flatten_config(config: dict, prefix: str = '') -> dict:
    """Flatten nested config dictionary"""
    flattened = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flattened.update(flatten_config(value, f"{prefix}{key}_" if prefix else f"{key}_"))
        else:
            flattened[f"{prefix}{key}" if prefix else key] = value
    return flattened


def create_training_config(config_dict: dict) -> TrainingConfig:
    """Create TrainingConfig from flattened config dictionary"""
    # Map YAML config keys to TrainingConfig fields
    config_mapping = {
        # Model parameters
        'model_d_signal': 'd_signal',
        'model_m_noise': 'm_noise', 
        'model_k_vec': 'k_vec',
        'model_layers': 'layers',
        'model_heads': 'heads',
        'model_rank': 'rank',
        'model_sigma': 'sigma',
        
        # Training parameters
        'training_batch_size': 'batch_size',
        'training_sequence_length': 'sequence_length',
        'training_learning_rate': 'learning_rate',
        'training_weight_decay': 'weight_decay',
        'training_beta1': 'beta1',
        'training_beta2': 'beta2',
        'training_grad_clip_norm': 'grad_clip_norm',
        'training_max_epochs': 'max_epochs',
        'training_warmup_steps': 'warmup_steps',
        'training_eval_interval': 'eval_interval',
        'training_save_interval': 'save_interval',
        
        # Data parameters
        'data_dataset_name': 'dataset_name',
        'data_dataset_config': 'dataset_config',
        'data_tokenizer_name': 'tokenizer_name',
        'data_num_workers': 'num_workers',
        'data_pin_memory': 'pin_memory',
        'data_max_dataset_tokens': 'max_dataset_tokens',
        
        # Experiment parameters
        'experiment_project_name': 'project_name',
        'experiment_experiment_name': 'experiment_name',
        'experiment_checkpoint_dir': 'checkpoint_dir',
        'experiment_log_level': 'log_level',
        
        # Hardware parameters
        'hardware_device': 'device',
    }
    
    # Create kwargs for TrainingConfig
    training_kwargs = {}
    for yaml_key, config_key in config_mapping.items():
        if yaml_key in config_dict and config_dict[yaml_key] is not None:
            training_kwargs[config_key] = config_dict[yaml_key]
    
    # Handle device auto-detection
    if training_kwargs.get('device') == 'auto':
        import torch
        training_kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return TrainingConfig(**training_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Launch secure transformer training with YAML config")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--count_params",
        action="store_true",
        help="Display parameter count breakdown and exit"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Override experiment name from config"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override learning rate from config"
    )
    
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Override max epochs from config"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Override checkpoint directory from config"
    )
    
    args = parser.parse_args()
    
    # Load and flatten config
    config = load_config(args.config)
    flattened_config = flatten_config(config)
    
    # Override config with command line arguments
    if args.experiment_name:
        flattened_config['experiment_experiment_name'] = args.experiment_name
    if args.batch_size:
        flattened_config['training_batch_size'] = args.batch_size
    if args.learning_rate:
        flattened_config['training_learning_rate'] = args.learning_rate
    if args.max_epochs:
        flattened_config['training_max_epochs'] = args.max_epochs
    if args.checkpoint_dir:
        flattened_config['experiment_checkpoint_dir'] = args.checkpoint_dir
    
    # Create training configuration
    training_config = create_training_config(flattened_config)
    
    # If only counting parameters, do that and exit
    if args.count_params:
        # Import here to avoid unnecessary imports when just counting params
        from secure_transformer.model import SecureTransformer
        
        model = SecureTransformer(training_config)
        
        # Count parameters for each component
        client_front_params = sum(p.numel() for p in model.client_front.parameters())
        server_core_params = sum(p.numel() for p in model.server_core.parameters())
        client_back_params = sum(p.numel() for p in model.client_back.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        
        # Print breakdown
        print("Config: ", config["model"])

        print("Parameter Distribution:")
        print(f"  ClientFront:  {client_front_params:>12,} ({client_front_params/total_params*100:.1f}%)")
        print(f"  ServerCore:   {server_core_params:>12,} ({server_core_params/total_params*100:.1f}%)")
        print(f"  ClientBack:   {client_back_params:>12,} ({client_back_params/total_params*100:.1f}%)")
        print(f"  Total:        {total_params:>12,}")
        return
    
    # Print configuration summary
    print("=" * 60)
    print("SECURE TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Experiment: {training_config.experiment_name}")
    print(f"Device: {training_config.device}")
    print(f"Model size: d={training_config.d_signal}, m={training_config.m_noise}, k={training_config.k_vec}")
    print(f"Architecture: {training_config.layers} layers, {training_config.heads} heads")
    print(f"Training: {training_config.max_epochs} epochs, lr={training_config.learning_rate}")
    print(f"Batch size: {training_config.batch_size}, seq_len={training_config.sequence_length}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = Trainer(training_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        if not trainer.load_checkpoint(args.resume):
            print("Failed to load checkpoint. Starting from scratch.")
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved. Training stopped.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Saving emergency checkpoint...")
        trainer.save_checkpoint()
        raise


if __name__ == "__main__":
    main() 