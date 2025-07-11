from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainingConfig:
    """Training configuration parameters"""

    # Model architecture
    vocab_size: int = 50257
    d_signal: int = 128
    m_noise: int = 64
    k_vec: int = 16
    layers: int = 6
    heads: int = 8
    rank: int = 8
    sigma: float = 1.0

    # Training hyperparameters
    batch_size: int = 32
    sequence_length: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # Training schedule
    max_epochs: int = 50
    warmup_steps: int = 2000
    eval_interval: int = 500
    save_interval: int = 1000

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    tokenizer_name: str = "gpt2"
    max_dataset_tokens: Optional[int] = (
        None  # Limit dataset size for testing (None = full dataset)
    )

    # Logging and checkpointing
    project_name: str = "secure-transformer"
    experiment_name: Optional[str] = None
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
