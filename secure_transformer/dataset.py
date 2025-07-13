import logging
from typing import Optional, List
import os
import pickle

import torch
from torch.utils.data import Dataset

from datasets import load_dataset  # type: ignore
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """Memory-efficient WikiText-103 dataset with tokenize-once approach"""

    split: str
    tokenizer: AutoTokenizer
    sequence_length: int
    max_length: Optional[int] = None
    all_tokens: List[int]
    num_sequences: int
    cache_dir: Optional[str] = None

    def __init__(
        self,
        split: str,
        tokenizer,
        sequence_length: int,
        max_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.max_length = max_length
        self.cache_dir = cache_dir

        self._initialize_tokenize_once()

    def _get_cache_path(self) -> Optional[str]:
        """Get the cache file path for pre-tokenized data"""
        if not self.cache_dir:
            return None

        os.makedirs(self.cache_dir, exist_ok=True)

        tokenizer_name = getattr(self.tokenizer, "name_or_path", "unknown").replace(
            "/", "_"
        )
        max_len_str = f"_max{self.max_length}" if self.max_length else ""
        cache_filename = f"wikitext103_{self.split}_{tokenizer_name}{max_len_str}.pkl"

        return os.path.join(self.cache_dir, cache_filename)

    def _load_from_cache(self) -> bool:
        """Try to load pre-tokenized data from cache"""
        cache_path = self._get_cache_path()
        if not cache_path or not os.path.exists(cache_path):
            return False

        try:
            logging.info(f"Loading tokenized data from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
                self.all_tokens = cache_data["tokens"]
                self.num_sequences = cache_data["num_sequences"]
            logging.info(
                f"Successfully loaded {len(self.all_tokens):,} tokens from cache"
            )
            return True
        except Exception as e:
            logging.warning(f"Failed to load from cache: {e}")
            return False

    def _save_to_cache(self):
        """Save pre-tokenized data to cache"""
        cache_path = self._get_cache_path()
        if not cache_path:
            return

        try:
            logging.info(f"Saving tokenized data to cache: {cache_path}")
            cache_data = {
                "tokens": self.all_tokens,
                "num_sequences": self.num_sequences,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            logging.info(f"Successfully saved {len(self.all_tokens):,} tokens to cache")
        except Exception as e:
            logging.warning(f"Failed to save to cache: {e}")

    def _initialize_tokenize_once(self):
        """Initialize by tokenizing the entire dataset once"""
        logging.info(
            f"Initializing {self.__class__.__name__} with tokenize-once approach..."
        )

        # Try to load from cache first
        if self._load_from_cache():
            return

        # Load the dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=self.split)

        # Optionally limit dataset size for testing
        if self.max_length:
            # Estimate how many examples we need based on average tokens per example
            # For WikiText-103, articles average ~2000 tokens
            estimated_examples_needed = self.max_length // 2000 + 1000  # Add buffer
            dataset = dataset.select(
                range(min(len(dataset), estimated_examples_needed))
            )

        logging.info(f"Processing {len(dataset):,} examples from {self.split} split...")

        # Define tokenization function
        def tokenize_function(examples):
            # Filter out empty texts
            texts = [text.strip() for text in examples["text"] if text.strip()]
            if not texts:
                return {"tokens": []}

            # Tokenize all texts in the batch
            tokenized = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            # Flatten all tokens into a single list
            # all_tokens = []
            # for token_ids in tokenized["input_ids"]:
            #     all_tokens.extend(token_ids)

            # return {"tokens": all_tokens}
            return {"tokens": tokenized["input_ids"]}

        # Use map to tokenize the dataset efficiently
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,  # Process in batches for efficiency
            remove_columns=dataset.column_names,  # Remove original columns
            desc="Tokenizing dataset",
        )

        # Collect all tokens from the mapped dataset
        self.all_tokens = []
        total_articles = 0

        for example in tokenized_dataset:
            if example["tokens"]:
                self.all_tokens.extend(example["tokens"])
                total_articles += 1

            # Optional: limit total tokens for testing
            if self.max_length and len(self.all_tokens) >= self.max_length:
                self.all_tokens = self.all_tokens[: self.max_length]
                break

        # Calculate exact number of sequences
        self.num_sequences = max(1, (len(self.all_tokens) - 1) // self.sequence_length)

        # Save to cache for future use
        self._save_to_cache()

        logging.info(
            f"Tokenization complete for '{self.split}' split: "
            f"processed {total_articles:,} articles, "
            f"{len(self.all_tokens):,} tokens, "
            f"{self.num_sequences:,} sequences"
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if idx >= self.num_sequences:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.num_sequences}"
            )

        # Calculate start position for this sequence
        start_pos = idx * self.sequence_length
        end_pos = start_pos + self.sequence_length + 1  # +1 for target

        # Extract tokens for this sequence
        if end_pos <= len(self.all_tokens):
            tokens = self.all_tokens[start_pos:end_pos]
        else:
            # Handle edge case where we don't have enough tokens
            tokens = self.all_tokens[start_pos:]
            # Pad with eos token if necessary
            pad_length = (self.sequence_length + 1) - len(tokens)
            if pad_length > 0:
                tokens.extend([self.tokenizer.eos_token_id] * pad_length)

        # Convert to tensor and create input/target pair
        tokens = torch.tensor(tokens, dtype=torch.long)
        input_ids = tokens[:-1]  # First sequence_length tokens
        target_ids = tokens[1:]  # Shifted by 1 for next token prediction

        return input_ids, target_ids

    def get_stats(self):
        """Get dataset statistics"""
        return {
            "split": self.split,
            "total_tokens": len(self.all_tokens),
            "num_sequences": self.num_sequences,
            "sequence_length": self.sequence_length,
            "max_length_limit": self.max_length,
        }

    def get_cache_stats(self):
        """Get caching statistics for debugging (kept for compatibility)"""
        return {
            "tokenize_once_approach": True,
            "total_tokens": len(self.all_tokens),
            "num_sequences": self.num_sequences,
            "cache_file": self._get_cache_path(),
        }
