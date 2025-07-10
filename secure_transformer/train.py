"""
Complete training script for the IND-CPA-secure SO(N)-equivariant transformer.

Features:
- WikiText-103 dataset loading and preprocessing
- Wandb integration for experiment tracking
- Checkpointing and resuming
- Validation and perplexity evaluation
- Learning rate scheduling
- Gradient clipping and monitoring
"""

import os
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import wandb
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from .model import ClientFront, ServerCore, ClientBack
from .utils import random_orthogonal


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

    # Logging and checkpointing
    project_name: str = "secure-transformer"
    experiment_name: Optional[str] = None
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True


class WikiTextDataset(Dataset):
    """WikiText-103 dataset with tokenization and chunking"""

    def __init__(self, split: str, tokenizer, sequence_length: int):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        # Concatenate all text
        text = "\n".join(dataset["text"])

        # Tokenize
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        # Calculate number of sequences
        self.num_sequences = len(self.tokens) // sequence_length

        logging.info(
            f"Loaded {split} split: {len(self.tokens)} tokens, {self.num_sequences} sequences"
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = start + self.sequence_length

        # Input and target (shifted by 1)
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1 : end + 1]

        return input_ids, target_ids


class SecureTransformer(nn.Module):
    """Complete secure transformer model combining all components"""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        self.client_front = ClientFront(
            vocab=config.vocab_size,
            d_signal=config.d_signal,
            m_noise=config.m_noise,
            k_vec=config.k_vec,
            sigma=config.sigma,
        )

        self.server_core = ServerCore(
            N=config.d_signal + config.m_noise,
            k_vec=config.k_vec,
            layers=config.layers,
            heads=config.heads,
        )

        self.client_back = ClientBack(
            vocab=config.vocab_size,
            d_signal=config.d_signal,
            m_noise=config.m_noise,
            k_vec=config.k_vec,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through the secure transformer"""
        # Client front: embed, add noise, rotate
        encrypted, rotation = self.client_front(tokens)

        # Server processing: equivariant computation
        processed = self.server_core(encrypted)

        # Client back: decrypt and get logits
        logits = self.client_back(processed, rotation)

        return logits


class Trainer:
    """Main training class"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = SecureTransformer(config).to(self.device)
        self.logger.info(
            f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Load data
        self.train_loader, self.val_loader = self._load_data()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize wandb
        if config.experiment_name is None:
            config.experiment_name = f"secure-transformer-{int(time.time())}"

        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            config=config.__dict__,
        )

        # Log model architecture
        wandb.watch(self.model, log_freq=100)

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine annealing"""
        total_steps = len(self.train_loader) * self.config.max_epochs

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.01,
        )

        # Sequential scheduler
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )

    def _load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare WikiText-103 dataset"""
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Update vocab size to match tokenizer
        self.config.vocab_size = len(tokenizer)

        # Create datasets
        train_dataset = WikiTextDataset("train", tokenizer, self.config.sequence_length)
        val_dataset = WikiTextDataset(
            "validation", tokenizer, self.config.sequence_length
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        return train_loader, val_loader

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss"""
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def _compute_perplexity(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute perplexity from loss"""
        return torch.exp(loss)

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass
        logits = self.model(input_ids)
        loss = self._compute_loss(logits, target_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        # Compute metrics
        perplexity = self._compute_perplexity(loss)

        return {
            "train_loss": loss.item(),
            "train_perplexity": perplexity.item(),
            "grad_norm": grad_norm.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            logits = self.model(input_ids)
            loss = self._compute_loss(logits, target_ids)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)

        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }

        # Save regular checkpoint
        checkpoint_path = (
            Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at step {self.step}")

        # Save latest checkpoint
        latest_path = Path(self.config.checkpoint_dir) / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        self.logger.info(f"Saved checkpoint at step {self.step}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]

            self.logger.info(f"Loaded checkpoint from step {self.step}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")

        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # Training loop
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)

                # Log training metrics
                if self.step % 50 == 0:
                    wandb.log(metrics, step=self.step)

                # Validation
                if self.step % self.config.eval_interval == 0:
                    val_metrics = self.validate()
                    wandb.log(val_metrics, step=self.step)

                    # Check if best model
                    is_best = val_metrics["val_loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics["val_loss"]

                    self.logger.info(
                        f"Step {self.step}: "
                        f"train_loss={metrics['train_loss']:.4f}, "
                        f"val_loss={val_metrics['val_loss']:.4f}, "
                        f"val_ppl={val_metrics['val_perplexity']:.2f}"
                    )

                    # Save checkpoint
                    if self.step % self.config.save_interval == 0:
                        self.save_checkpoint(is_best)

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

            # Save checkpoint at end of epoch
            self.save_checkpoint()

        self.logger.info("Training completed!")
        wandb.finish()


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Train secure transformer on WikiText-103"
    )

    # Model parameters
    parser.add_argument("--d_signal", type=int, default=128, help="Signal dimension")
    parser.add_argument("--m_noise", type=int, default=64, help="Noise dimension")
    parser.add_argument("--k_vec", type=int, default=16, help="Vector dimension")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--rank", type=int, default=8, help="Lie algebra rank")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--sequence_length", type=int, default=512, help="Sequence length"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")

    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(**vars(args))

    # Initialize trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
