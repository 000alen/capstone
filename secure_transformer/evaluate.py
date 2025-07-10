"""
Evaluation script for the secure transformer model.

Features:
- Load trained checkpoints
- Evaluate on WikiText-103 test set
- Compute perplexity and other metrics
- Generate sample text
- Analyze security properties
"""

import os
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np

from .train import WikiTextDataset, SecureTransformer, TrainingConfig
from .utils import random_orthogonal


class Evaluator:
    """Model evaluation class"""
    
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint["config"]
        
        # Initialize model
        self.model = SecureTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info(f"Loaded model from {checkpoint_path}")
        logging.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def evaluate_dataset(self, split: str = "test") -> Dict[str, float]:
        """Evaluate model on WikiText-103 dataset"""
        # Create dataset
        dataset = WikiTextDataset(split, self.tokenizer, self.config.sequence_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    target_ids.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += target_ids.numel()
                num_batches += 1
                
                if num_batches % 100 == 0:
                    current_ppl = math.exp(total_loss / total_tokens)
                    logging.info(f"Batch {num_batches}, current perplexity: {current_ppl:.2f}")
        
        # Compute final metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_batches": num_batches
        }
    
    def generate_text(self, 
                     prompt: str, 
                     max_length: int = 100, 
                     temperature: float = 1.0,
                     top_k: int = 50) -> str:
        """Generate text given a prompt"""
        # Tokenize prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt), 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0)
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # Take last sequence_length tokens if needed
                if generated.size(1) > self.config.sequence_length:
                    current_input = generated[:, -self.config.sequence_length:]
                else:
                    current_input = generated
                
                # Forward pass
                logits = self.model(current_input)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text
    
    def analyze_security_properties(self, num_samples: int = 100) -> Dict[str, float]:
        """Analyze security properties of the model"""
        results = {
            "noise_variance": [],
            "signal_to_noise_ratio": [],
            "rotation_determinant": [],
            "orthogonality_error": []
        }
        
        # Generate random tokens for analysis
        vocab_size = len(self.tokenizer)
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Random tokens
                tokens = torch.randint(0, vocab_size, (1, 10), device=self.device)
                
                # Get encrypted representation
                encrypted, rotation = self.model.client_front(tokens)
                
                # Decrypt to get the original signal and noise components
                decrypted = torch.einsum("ij,btkj->btki", rotation.t(), encrypted)
                signal_part = decrypted[..., :self.config.d_signal]
                noise_part = decrypted[..., self.config.d_signal:]
                
                # Compute statistics
                noise_var = noise_part.var().item()
                signal_norm = signal_part.norm().item()
                noise_norm = noise_part.norm().item()
                snr = signal_norm / (noise_norm + 1e-8)
                
                # Check rotation properties
                det = torch.det(rotation).item()
                orthogonality_error = (rotation @ rotation.t() - torch.eye(
                    rotation.size(0), device=rotation.device
                )).norm().item()
                
                results["noise_variance"].append(noise_var)
                results["signal_to_noise_ratio"].append(snr)
                results["rotation_determinant"].append(det)
                results["orthogonality_error"].append(orthogonality_error)
        
        # Compute summary statistics
        summary = {}
        for key, values in results.items():
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_min"] = np.min(values)
            summary[f"{key}_max"] = np.max(values)
        
        return summary
    
    def compute_layer_statistics(self) -> Dict[str, Any]:
        """Compute statistics about model layers"""
        stats = {}
        
        # Analyze embeddings
        embed_weights = self.model.client_front.embed.data
        stats["embedding_norm_mean"] = embed_weights.norm(dim=-1).mean().item()
        stats["embedding_norm_std"] = embed_weights.norm(dim=-1).std().item()
        
        # Analyze server layers
        layer_stats = []
        for i, block in enumerate(self.model.server_core.blocks):
            layer_info = {
                "layer": i,
                "attention_qs_norm": block.attn.qs.norm().item(),
                "attention_ks_norm": block.attn.ks.norm().item(), 
                "attention_vs_norm": block.attn.vs.norm().item(),
                "lie_u_norm": block.lie.u.norm().item(),
                "lie_v_norm": block.lie.v.norm().item(),
                "lie_a_norm": block.lie.a.norm().item()
            }
            layer_stats.append(layer_info)
        
        stats["layers"] = layer_stats
        
        # Analyze output head
        head_weights = self.model.client_back.head.weight.data
        stats["head_norm_mean"] = head_weights.norm(dim=-1).mean().item()
        stats["head_norm_std"] = head_weights.norm(dim=-1).std().item()
        
        return stats


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate secure transformer model")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to evaluate on"
    )
    
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        help="Generate text samples"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of text samples to generate"
    )
    
    parser.add_argument(
        "--analyze_security",
        action="store_true",
        help="Analyze security properties"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Save results to file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator
    evaluator = Evaluator(args.checkpoint)
    
    results = {}
    
    # Evaluate on dataset
    print("Evaluating on dataset...")
    dataset_results = evaluator.evaluate_dataset(args.eval_split)
    results["dataset_evaluation"] = dataset_results
    
    print(f"Results on {args.eval_split} set:")
    print(f"  Loss: {dataset_results['loss']:.4f}")
    print(f"  Perplexity: {dataset_results['perplexity']:.2f}")
    print(f"  Total tokens: {dataset_results['total_tokens']:,}")
    
    # Generate text samples
    if args.generate_samples:
        print("\nGenerating text samples...")
        prompts = [
            "The quick brown fox",
            "In a distant galaxy",
            "The theory of relativity",
            "Machine learning is", 
            "Once upon a time"
        ]
        
        samples = []
        for i, prompt in enumerate(prompts[:args.num_samples]):
            generated = evaluator.generate_text(prompt, max_length=50)
            samples.append({"prompt": prompt, "generated": generated})
            print(f"\nSample {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
        
        results["text_samples"] = samples
    
    # Analyze security properties
    if args.analyze_security:
        print("\nAnalyzing security properties...")
        security_stats = evaluator.analyze_security_properties()
        results["security_analysis"] = security_stats
        
        print("Security Statistics:")
        for key, value in security_stats.items():
            if "mean" in key:
                print(f"  {key}: {value:.4f}")
    
    # Compute layer statistics
    print("\nComputing layer statistics...")
    layer_stats = evaluator.compute_layer_statistics()
    results["layer_statistics"] = layer_stats
    
    print("Model Statistics:")
    print(f"  Embedding norm (mean): {layer_stats['embedding_norm_mean']:.4f}")
    print(f"  Head norm (mean): {layer_stats['head_norm_mean']:.4f}")
    print(f"  Number of layers: {len(layer_stats['layers'])}")
    
    # Save results
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main() 