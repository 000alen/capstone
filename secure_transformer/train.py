import os
import math
import time
import argparse
import logging
import typing
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import wandb
from transformers import AutoTokenizer

from .config import TrainingConfig
from .model import SecureTransformer
from .dataset import WikiTextDataset


class Trainer:
    """Main training class"""

    step_times: typing.List[float]

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing Secure Transformer Training...")

        # Create checkpoint directory
        self.logger.info("Setting up checkpoint directory...")
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # Load data (and tokenizer) FIRST so vocab_size is correct
        # ------------------------------------------------------------
        self.logger.info("Loading datasets and tokenizer...")
        self.train_loader, self.val_loader = self._load_data()

        # Calculate total steps for time estimation (needs train_loader)
        self.total_steps = len(self.train_loader) * self.config.max_epochs

        # ------------------------------------------------------------
        # Initialize model (after vocab_size potentially updated)
        # ------------------------------------------------------------
        self.logger.info("Initializing model...")
        self.model = SecureTransformer(config).to(self.device)

        # Optionally load pretrained embeddings
        if self.config.load_pretrained_embeddings:
            self._load_pretrained_embeddings()

        # Optionally freeze embeddings
        if self.config.freeze_embeddings:
            self.logger.info("Freezing token embedding matrix (no gradient updates)...")
            self.model.client_front.embed.requires_grad_(False)

        self.logger.info(
            f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

        # ------------------------------------------------------------
        # Initialize optimizer (only trainable params)
        # ------------------------------------------------------------
        self.logger.info("Setting up optimizer...")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Time tracking for estimates
        self.training_start_time = None
        self.step_times = []  # Track recent step times for moving average
        self.max_step_times = 100  # Keep last 100 step times for averaging

        # Initialize scheduler
        self.logger.info("Setting up learning rate scheduler...")
        self.scheduler = self._create_scheduler()

        # Initialize wandb
        self.logger.info("Initializing Weights & Biases...")
        if config.experiment_name is None:
            config.experiment_name = f"secure-transformer-{int(time.time())}"

        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            config=config.__dict__,
        )

        # Log model architecture
        wandb.watch(self.model, log_freq=100)

        self.logger.info("Initialization complete! Ready to train.")
        self.logger.info(f"Experiment: {config.experiment_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Total training steps: {self.total_steps:,}")

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

    # ------------------------------------------------------------------
    # New helper: load pretrained embeddings into ClientFront
    # ------------------------------------------------------------------
    def _load_pretrained_embeddings(self):
        """Load token embeddings from a pretrained HuggingFace model

        Requires that the tokenizer has already set self.config.vocab_size.
        Checks dimensional consistency with (k_vec * d_signal).
        """
        from transformers import AutoModel

        model_name = self.config.pretrained_model_name
        self.logger.info(f"Loading pretrained embeddings from '{model_name}' ...")

        hf_model = AutoModel.from_pretrained(model_name)
        emb_weight = hf_model.get_input_embeddings().weight.detach()

        vocab_size, embed_dim = emb_weight.shape
        if vocab_size != self.config.vocab_size:
            raise ValueError(
                f"Vocab size mismatch: tokenizer={self.config.vocab_size}, model={vocab_size}"
            )

        expected_dim = self.config.k_vec * self.config.d_signal
        if embed_dim != expected_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {expected_dim} (k_vec*d_signal), got {embed_dim}."
            )

        # Reshape and copy into ClientFront embedding parameter
        reshaped = emb_weight.view(vocab_size, self.config.k_vec, self.config.d_signal)
        target_param = self.model.client_front.embed
        self.model.client_front.embed.data.copy_(
            reshaped.to(device=target_param.device, dtype=target_param.dtype)
        )

        self.logger.info("Pretrained embeddings loaded successfully.")

    def _load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare WikiText-103 dataset"""
        self.logger.info("Loading data and initializing datasets...")

        # Initialize tokenizer
        self.logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Update vocab size to match tokenizer
        self.config.vocab_size = len(tokenizer)
        self.logger.info(f"Tokenizer loaded: vocab_size={self.config.vocab_size}")

        # Create datasets with progress logging
        self.logger.info("Creating training dataset...")
        train_dataset = WikiTextDataset(
            "train",
            tokenizer,
            self.config.sequence_length,
            self.config.max_dataset_tokens,
        )

        self.logger.info("Creating validation dataset...")
        val_dataset = WikiTextDataset(
            "validation",
            tokenizer,
            self.config.sequence_length,
            self.config.max_dataset_tokens,
        )

        # Create data loaders
        self.logger.info("Creating data loaders...")
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

        self.logger.info(f"Data loading complete:")
        self.logger.info(f"Training batches: {len(train_loader):,}")
        self.logger.info(f"Validation batches: {len(val_loader):,}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Sequence length: {self.config.sequence_length}")

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
        step_start_time = time.time()

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

        # Track step time
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)

        # Keep only recent step times for moving average
        if len(self.step_times) > self.max_step_times:
            self.step_times.pop(0)

        # Compute metrics
        perplexity = self._compute_perplexity(loss)

        return {
            "train_loss": loss.item(),
            "train_perplexity": perplexity.item(),
            "grad_norm": grad_norm.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "step_time": step_time,
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
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            # Handle extremely large loss values that would overflow
            self.logger.warning(
                f"Perplexity overflow for avg_loss={avg_loss:.4f}; setting to inf."
            )
            perplexity = float("inf")

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
            "training_start_time": self.training_start_time,
            "step_times": self.step_times,
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

            # Load timing information if available (for backwards compatibility)
            if "training_start_time" in checkpoint:
                self.training_start_time = checkpoint["training_start_time"]
            if "step_times" in checkpoint:
                self.step_times = checkpoint["step_times"]

            self.logger.info(f"Loaded checkpoint from step {self.step}")

            # Log time information if available
            if self.training_start_time:
                time_estimates = self._get_time_estimates()
                self.logger.info(
                    f"Resumed training - Elapsed: {time_estimates['elapsed']}, ETA: {time_estimates['total']}"
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")

        # Set training start time
        self.training_start_time = time.time()

        # Log training overview
        self.logger.info(f"Total steps: {self.total_steps:,}")
        self.logger.info(f"Steps per epoch: {len(self.train_loader):,}")
        self.logger.info(f"Total epochs: {self.config.max_epochs}")

        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # Training loop
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)

                # Log training metrics
                if self.step % 50 == 0:
                    # Add time estimates to metrics
                    time_estimates = self._get_time_estimates()
                    metrics.update(time_estimates)
                    wandb.log(metrics, step=self.step)

                # Validation
                if self.step % self.config.eval_interval == 0:
                    val_metrics = self.validate()

                    # Get time estimates and progress info
                    time_estimates = self._get_time_estimates()
                    progress_percent = (self.step / self.total_steps) * 100

                    wandb.log(
                        {
                            **val_metrics,
                            **time_estimates,
                            "progress_percent": progress_percent,
                        },
                        step=self.step,
                    )

                    # Check if best model
                    is_best = val_metrics["val_loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics["val_loss"]

                    self.logger.info(
                        f"Step {self.step}/{self.total_steps} ({progress_percent:.1f}%): "
                        f"train_loss={metrics['train_loss']:.4f}, "
                        f"val_loss={val_metrics['val_loss']:.4f}, "
                        f"val_ppl={val_metrics['val_perplexity']:.2f} | "
                        f"Elapsed: {time_estimates['elapsed']}, "
                        f"Remaining: {time_estimates['remaining']}, "
                        f"ETA: {time_estimates['total']}"
                    )

                    # Save checkpoint
                    if self.step % self.config.save_interval == 0:
                        self.save_checkpoint(is_best)

                # Log cache statistics periodically (every 1000 steps)
                if self.step % 1000 == 0 and self.step > 0:
                    try:
                        train_cache_stats = self.train_loader.dataset.get_cache_stats()
                        self.logger.info(f"Dataset cache stats: {train_cache_stats}")
                    except Exception:
                        pass  # Ignore if cache stats not available

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            time_estimates = self._get_time_estimates()
            epochs_remaining = self.config.max_epochs - (epoch + 1)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.max_epochs} completed in {epoch_time:.2f}s | "
                f"Epochs remaining: {epochs_remaining} | "
                f"Total elapsed: {time_estimates['elapsed']}, "
                f"ETA: {time_estimates['total']}"
            )

            # Save checkpoint at end of epoch
            self.save_checkpoint()

        self.logger.info("Training completed!")
        final_time = self._get_time_estimates()
        self.logger.info(f"Total training time: {final_time['elapsed']}")

        # Final cache statistics
        try:
            train_cache_stats = self.train_loader.dataset.get_cache_stats()
            self.logger.info(f"Final dataset cache stats: {train_cache_stats}")
        except Exception:
            pass

        wandb.finish()

    def _format_time(self, seconds: float) -> str:
        """Format time in a human-readable way"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"

    def _get_time_estimates(self) -> Dict[str, str]:
        """Calculate time estimates based on current progress"""
        if self.training_start_time is None or self.step == 0:
            return {
                "elapsed": "0s",
                "remaining": "calculating...",
                "total": "calculating...",
                "avg_step_time": "calculating...",
            }

        # Calculate elapsed time
        elapsed = time.time() - self.training_start_time

        # Calculate average step time from recent steps
        if len(self.step_times) > 0:
            avg_step_time = sum(self.step_times) / len(self.step_times)
        else:
            avg_step_time = elapsed / max(self.step, 1)

        # Calculate remaining steps and estimated time
        remaining_steps = max(0, self.total_steps - self.step)
        estimated_remaining = remaining_steps * avg_step_time
        estimated_total = elapsed + estimated_remaining

        return {
            "elapsed": self._format_time(elapsed),
            "remaining": self._format_time(estimated_remaining),
            "total": self._format_time(estimated_total),
            "avg_step_time": f"{avg_step_time:.2f}s",
        }


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Train secure transformer on WikiText-103"
    )

    parser.add_argument("--d_signal", type=int, default=128, help="Signal dimension")
    parser.add_argument("--m_noise", type=int, default=64, help="Noise dimension")
    parser.add_argument("--k_vec", type=int, default=16, help="Vector dimension")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--rank", type=int, default=8, help="Lie algebra rank")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--sequence_length", type=int, default=512, help="Sequence length"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")

    parser.add_argument(
        "--max_dataset_tokens",
        type=int,
        help="Limit dataset size for testing (default: full dataset)",
    )

    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    config = TrainingConfig(**vars(args))

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
