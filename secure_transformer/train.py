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
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import wandb
from datasets import load_dataset  # type: ignore
from transformers import AutoTokenizer

from .config import TrainingConfig
from .model import SecureTransformer


class WikiTextDataset(Dataset):
    """Memory-efficient WikiText-103 dataset with lazy loading"""

    def __init__(
        self,
        split: str,
        tokenizer,
        sequence_length: int,
        max_length: Optional[int] = None,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.max_length = max_length  # Optional limit for testing

        # Load dataset in streaming mode to avoid loading all into memory
        self.dataset = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split=split, streaming=True
        )

        # Quick initialization without processing entire dataset
        self._initialize_lazy()

    def _initialize_lazy(self):
        """Fast initialization with lazy loading - estimate size without full processing"""
        logging.info(f"Initializing {self.__class__.__name__} with lazy loading...")
        
        # For WikiText-103, we know approximate sizes to avoid full scan
        # These are rough estimates to get started quickly
        if self.max_length:
            estimated_tokens = min(self.max_length, 100_000_000)  # Cap at 100M tokens
        else:
            # Rough estimates for WikiText-103 splits
            estimates = {
                "train": 100_000_000,      # ~100M tokens
                "validation": 200_000,     # ~200K tokens  
                "test": 240_000           # ~240K tokens
            }
            estimated_tokens = estimates.get(self.split, estimates["train"])
        
        # Estimate number of sequences
        self.estimated_num_sequences = max(1, (estimated_tokens - 1) // self.sequence_length)
        
        # Cache for processed articles to avoid re-tokenization
        self._article_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logging.info(
            f"Lazy initialization complete for '{self.split}' split: "
            f"estimated ~{estimated_tokens:,} tokens, "
            f"~{self.estimated_num_sequences:,} sequences"
        )

    def __len__(self):
        return self.estimated_num_sequences

    def __getitem__(self, idx):
        if idx >= self.estimated_num_sequences:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.estimated_num_sequences}"
            )

        # Calculate which articles and positions we need for this sequence
        target_start_pos = idx * self.sequence_length
        target_end_pos = target_start_pos + self.sequence_length + 1  # +1 for target
        
        # Get tokens for this range with lazy loading
        tokens = self._get_tokens_lazy(target_start_pos, target_end_pos)

        # Ensure we have enough tokens
        if len(tokens) < self.sequence_length + 1:
            # Pad with eos token if necessary
            pad_length = (self.sequence_length + 1) - len(tokens)
            tokens.extend([self.tokenizer.eos_token_id] * pad_length)

        # Convert to tensor and create input/target pair
        tokens = torch.tensor(tokens[: self.sequence_length + 1], dtype=torch.long)
        input_ids = tokens[:-1]  # First sequence_length tokens
        target_ids = tokens[1:]  # Shifted by 1 for next token prediction

        return input_ids, target_ids

    def _get_tokens_lazy(self, start_pos: int, end_pos: int) -> list:
        """Efficiently get tokens for range with caching and lazy processing"""
        tokens = []
        current_pos = 0
        articles_processed = 0
        
        # Iterator through dataset
        for article_idx, example in enumerate(self.dataset):
            text = example["text"].strip()
            if not text:  # Skip empty lines
                continue
                
            # Check cache first
            if article_idx in self._article_cache:
                article_tokens = self._article_cache[article_idx]
                self._cache_hits += 1
            else:
                # Tokenize and cache
                article_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                # Only cache if article is reasonably sized (avoid memory bloat)
                if len(article_tokens) < 10000:  # Cache articles < 10K tokens
                    self._article_cache[article_idx] = article_tokens
                self._cache_misses += 1
            
            if not article_tokens:
                continue
                
            article_start = current_pos
            article_end = current_pos + len(article_tokens)
            
            # Check if this article overlaps with our desired range
            if article_end <= start_pos:
                # Article is completely before our range
                current_pos = article_end
                continue
            elif article_start >= end_pos:
                # Article is completely after our range - we can stop
                break
                
            # This article overlaps with our range
            article_relative_start = max(0, start_pos - article_start)
            article_relative_end = min(len(article_tokens), end_pos - article_start)

            if article_relative_end > article_relative_start:
                needed_tokens = article_tokens[article_relative_start:article_relative_end]
                tokens.extend(needed_tokens)

                # If we have enough tokens, we can stop
                if len(tokens) >= (end_pos - start_pos):
                    break
                    
            current_pos = article_end
            articles_processed += 1
            
            # Optional: limit processing for testing
            if self.max_length and current_pos >= self.max_length:
                break
                
            # Early stopping if we've found enough tokens
            if len(tokens) >= (end_pos - start_pos):
                break

        return tokens
    
    def get_cache_stats(self):
        """Get caching statistics for debugging"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total_requests, 1) * 100
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses, 
            "hit_rate": f"{hit_rate:.1f}%",
            "cached_articles": len(self._article_cache)
        }


class Trainer:
    """Main training class"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Initializing Secure Transformer Training...")

        # Create checkpoint directory
        self.logger.info("📁 Setting up checkpoint directory...")
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.logger.info("🧠 Initializing model...")
        self.model = SecureTransformer(config).to(self.device)
        self.logger.info(
            f"✅ Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

        # Initialize optimizer
        self.logger.info("⚡ Setting up optimizer...")
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

        # Time tracking for estimates
        self.training_start_time = None
        self.step_times = []  # Track recent step times for moving average
        self.max_step_times = 100  # Keep last 100 step times for averaging

        # Load data
        self.train_loader, self.val_loader = self._load_data()

        # Calculate total steps for time estimation
        self.total_steps = len(self.train_loader) * self.config.max_epochs

        # Initialize scheduler
        self.logger.info("📈 Setting up learning rate scheduler...")
        self.scheduler = self._create_scheduler()

        # Initialize wandb
        self.logger.info("📊 Initializing Weights & Biases...")
        if config.experiment_name is None:
            config.experiment_name = f"secure-transformer-{int(time.time())}"

        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            config=config.__dict__,
        )

        # Log model architecture
        wandb.watch(self.model, log_freq=100)
        
        self.logger.info("✅ Initialization complete! Ready to train.")
        self.logger.info(f"🎯 Experiment: {config.experiment_name}")
        self.logger.info(f"🔧 Device: {self.device}")
        self.logger.info(f"📏 Total training steps: {self.total_steps:,}")

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
        self.logger.info("🔄 Loading data and initializing datasets...")
        
        # Initialize tokenizer
        self.logger.info("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Update vocab size to match tokenizer
        self.config.vocab_size = len(tokenizer)
        self.logger.info(f"✅ Tokenizer loaded: vocab_size={self.config.vocab_size}")

        # Create datasets with progress logging
        self.logger.info("📚 Creating training dataset...")
        train_dataset = WikiTextDataset(
            "train",
            tokenizer,
            self.config.sequence_length,
            self.config.max_dataset_tokens,
        )
        
        self.logger.info("📖 Creating validation dataset...")
        val_dataset = WikiTextDataset(
            "validation",
            tokenizer,
            self.config.sequence_length,
            self.config.max_dataset_tokens,
        )

        # Create data loaders
        self.logger.info("⚙️  Creating data loaders...")
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
        
        self.logger.info(f"✅ Data loading complete:")
        self.logger.info(f"   📊 Training batches: {len(train_loader):,}")
        self.logger.info(f"   📊 Validation batches: {len(val_loader):,}")
        self.logger.info(f"   📊 Batch size: {self.config.batch_size}")
        self.logger.info(f"   📊 Sequence length: {self.config.sequence_length}")

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
                        self.logger.info(f"📊 Dataset cache stats: {train_cache_stats}")
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
            self.logger.info(f"📊 Final dataset cache stats: {train_cache_stats}")
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

    # Data parameters
    parser.add_argument(
        "--max_dataset_tokens",
        type=int,
        help="Limit dataset size for testing (default: full dataset)",
    )

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
