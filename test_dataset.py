#!/usr/bin/env python3
"""
Simple test script to verify the memory-efficient WikiText dataset loading
"""

import torch
from transformers import AutoTokenizer
from secure_transformer.train import WikiTextDataset
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_dataset_loading():
    """Test dataset loading with memory monitoring"""
    print("Testing memory-efficient WikiText dataset loading...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Test with small dataset first
    print("\n1. Testing with small dataset (100K tokens)...")
    train_dataset = WikiTextDataset(
        split="train", 
        tokenizer=tokenizer, 
        sequence_length=256, 
        max_length=100000  # Small test
    )
    
    print(f"Memory after small dataset: {get_memory_usage():.1f} MB")
    print(f"Dataset size: {len(train_dataset)} sequences")
    
    # Test accessing some samples
    print("\n2. Testing data access...")
    for i in range(min(3, len(train_dataset))):
        input_ids, target_ids = train_dataset[i]
        print(f"Sample {i}: input shape {input_ids.shape}, target shape {target_ids.shape}")
    
    print(f"Memory after accessing samples: {get_memory_usage():.1f} MB")
    
    # Test with larger dataset 
    print("\n3. Testing with larger dataset (1M tokens)...")
    train_dataset_large = WikiTextDataset(
        split="train", 
        tokenizer=tokenizer, 
        sequence_length=256, 
        max_length=1000000  # 1M tokens
    )
    
    print(f"Memory after large dataset: {get_memory_usage():.1f} MB")
    print(f"Large dataset size: {len(train_dataset_large)} sequences")
    
    # Test batch creation
    print("\n4. Testing batch creation...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(train_dataset_large, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    input_ids, target_ids = batch
    
    print(f"Batch input shape: {input_ids.shape}")
    print(f"Batch target shape: {target_ids.shape}")
    print(f"Final memory usage: {get_memory_usage():.1f} MB")
    
    print("\n✅ Dataset loading test completed successfully!")

if __name__ == "__main__":
    test_dataset_loading() 