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

import logging
import argparse
import yaml
import colorlog
from secure_transformer.train import Trainer, TrainingConfig

# Configure colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

logger = logging.getLogger("train_launcher")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def flatten_config(config: dict, prefix: str = "") -> dict:
    """Flatten nested config dictionary"""

    flattened = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flattened.update(
                flatten_config(value, f"{prefix}{key}_" if prefix else f"{key}_")
            )
        else:
            flattened[f"{prefix}{key}" if prefix else key] = value
    return flattened


def create_training_config(config_dict: dict) -> TrainingConfig:
    """Create TrainingConfig from flattened config dictionary"""

    # Map YAML config keys to TrainingConfig fields
    config_mapping = {
        # Model parameters
        "model_d_signal": "d_signal",
        "model_m_noise": "m_noise",
        "model_k_vec": "k_vec",
        "model_layers": "layers",
        "model_heads": "heads",
        "model_rank": "rank",
        "model_sigma": "sigma",
        # Training parameters
        "training_batch_size": "batch_size",
        "training_sequence_length": "sequence_length",
        "training_learning_rate": "learning_rate",
        "training_weight_decay": "weight_decay",
        "training_beta1": "beta1",
        "training_beta2": "beta2",
        "training_grad_clip_norm": "grad_clip_norm",
        "training_max_epochs": "max_epochs",
        "training_warmup_steps": "warmup_steps",
        "training_eval_interval": "eval_interval",
        "training_save_interval": "save_interval",
        # Data parameters
        "data_dataset_name": "dataset_name",
        "data_dataset_config": "dataset_config",
        "data_tokenizer_name": "tokenizer_name",
        "data_num_workers": "num_workers",
        "data_pin_memory": "pin_memory",
        "data_max_dataset_tokens": "max_dataset_tokens",
        # Experiment parameters
        "experiment_project_name": "project_name",
        "experiment_experiment_name": "experiment_name",
        "experiment_checkpoint_dir": "checkpoint_dir",
        "experiment_log_level": "log_level",
        # Hardware parameters
        "hardware_device": "device",
    }

    # Create kwargs for TrainingConfig
    training_kwargs = {}
    for yaml_key, config_key in config_mapping.items():
        if yaml_key in config_dict and config_dict[yaml_key] is not None:
            training_kwargs[config_key] = config_dict[yaml_key]

    # Handle device auto-detection
    if training_kwargs.get("device") == "auto":
        import torch

        training_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainingConfig(**training_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Launch secure transformer training with YAML config"
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--count_params",
        action="store_true",
        help="Display parameter count breakdown and exit",
    )

    parser.add_argument(
        "--experiment_name", type=str, help="Override experiment name from config"
    )

    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")

    parser.add_argument(
        "--batch_size", type=int, help="Override batch size from config"
    )

    parser.add_argument(
        "--learning_rate", type=float, help="Override learning rate from config"
    )

    parser.add_argument(
        "--max_epochs", type=int, help="Override max epochs from config"
    )

    parser.add_argument(
        "--checkpoint_dir", type=str, help="Override checkpoint directory from config"
    )

    args = parser.parse_args()

    config = load_config(args.config)
    flattened_config = flatten_config(config)

    # Override config with command line arguments
    if args.experiment_name:
        flattened_config["experiment_experiment_name"] = args.experiment_name
    if args.batch_size:
        flattened_config["training_batch_size"] = args.batch_size
    if args.learning_rate:
        flattened_config["training_learning_rate"] = args.learning_rate
    if args.max_epochs:
        flattened_config["training_max_epochs"] = args.max_epochs
    if args.checkpoint_dir:
        flattened_config["experiment_checkpoint_dir"] = args.checkpoint_dir

    training_config = create_training_config(flattened_config)

    if args.count_params:
        import json

        from secure_transformer.model import SecureTransformer

        model = SecureTransformer(training_config)

        # Count parameters for each component
        client_front_params = sum(p.numel() for p in model.client_front.parameters())
        server_core_params = sum(p.numel() for p in model.server_core.parameters())
        client_back_params = sum(p.numel() for p in model.client_back.parameters())
        total_params = sum(p.numel() for p in model.parameters())

        # Print breakdown
        logging.info("config: %s", json.dumps(config["model"], indent=2))

        logging.info("parameter distribution:")

        logging.info(
            f"ClientFront: {client_front_params:,} ({client_front_params/total_params*100:.1f}%)"
        )

        logging.info(
            f"ServerCore: {server_core_params:,} ({server_core_params/total_params*100:.1f}%)"
        )

        logging.info(
            f"ClientBack: {client_back_params:,} ({client_back_params/total_params*100:.1f}%)"
        )

        logging.info(f"total: {total_params:,}")

        return

    logging.info(f"config: {args.config}")
    logging.info(f"experiment: {training_config.experiment_name}")
    logging.info(f"device: {training_config.device}")
    logging.info(
        f"Model size: d={training_config.d_signal}, m={training_config.m_noise}, k={training_config.k_vec}"
    )
    logging.info(
        f"architecture: {training_config.layers} layers, {training_config.heads} heads"
    )
    logging.info(
        f"Training: {training_config.max_epochs} epochs, lr={training_config.learning_rate}"
    )
    logging.info(
        f"batch size: {training_config.batch_size}, seq_len={training_config.sequence_length}"
    )

    trainer = Trainer(training_config)

    if args.resume:
        logging.info(f"resuming from checkpoint: {args.resume}")
        if not trainer.load_checkpoint(args.resume):
            logging.info("failed to load checkpoint. starting from scratch.")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.info("\ntraining interrupted by user. saving checkpoint...")
        trainer.save_checkpoint()
        logging.info("checkpoint saved. training stopped.")
    except Exception as e:
        logging.info(f"\ntraining failed with error: {e}")
        logging.info("saving emergency checkpoint...")
        trainer.save_checkpoint()
        raise


if __name__ == "__main__":
    main()
