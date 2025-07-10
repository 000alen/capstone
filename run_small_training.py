#!/usr/bin/env python3
"""
Script to run training with the small config and memory-efficient dataset loading.
This should avoid OOM issues on systems with limited RAM.
"""

import subprocess
import sys
from pathlib import Path

def run_training():
    """Run training with small config and dataset size limit"""
    
    # Training command with memory-efficient settings
    cmd = [
        sys.executable, "-m", "secure_transformer.train",
        "--d_signal", "64",           # Small model
        "--m_noise", "32", 
        "--k_vec", "8",
        "--layers", "3",
        "--heads", "4",
        "--batch_size", "8",          # Small batch size to save memory
        "--sequence_length", "256",   # Shorter sequences
        "--max_dataset_tokens", "500000",  # Limit to 500K tokens for testing
        "--learning_rate", "0.0005",
        "--max_epochs", "5",          # Just a few epochs for testing
        "--warmup_steps", "200",
        "--eval_interval", "50",
        "--save_interval", "100",
        "--experiment_name", "secure-transformer-test",
        "--checkpoint_dir", "./checkpoints_test"
    ]
    
    print("🚀 Starting training with memory-efficient settings...")
    print("Command:", " ".join(cmd))
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    run_training() 