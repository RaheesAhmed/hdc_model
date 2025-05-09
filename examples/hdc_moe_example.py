"""
Example usage of the HDC-MoE language model.

This script demonstrates how to use the HDC-MoE language model and
provides detailed memory usage and performance statistics.
"""

import sys
import os
import time
import psutil
import tracemalloc
import gc
from typing import List, Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc_model.models.hdc_moe import HDCMoELanguageModel


def get_memory_usage():
    """Get the current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def format_bytes(bytes: int) -> str:
    """Format bytes as a human-readable string."""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.2f} GB"


def main():
    # Sample corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a stitch in time saves nine",
        "all that glitters is not gold",
        "actions speak louder than words",
        "the early bird catches the worm",
        "practice makes perfect",
        "look before you leap",
        "the pen is mightier than the sword",
        "when in rome do as the romans do",
        "the cat sat on the mat",
        "the dog barked at the cat",
        "the bird flew over the tree",
        "the fish swam in the pond",
        "the sun shines bright in the sky",
        "the moon glows at night",
        "the stars twinkle in the dark",
        "the rain falls from the clouds",
        "the snow covers the ground",
        "the wind blows through the trees",
        "the river flows to the sea",
    ]
    
    # Parameters
    n = 3
    dimensions = 10000
    n_experts = 10
    sparsity = 0.9
    top_k_experts = 2
    
    print("HDC-MoE Language Model Example")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n = {n} (n-gram size)")
    print(f"  dimensions = {dimensions} (hypervector dimensionality)")
    print(f"  n_experts = {n_experts} (number of experts)")
    print(f"  sparsity = {sparsity} (proportion of zeros in hypervectors)")
    print(f"  top_k_experts = {top_k_experts} (number of experts used for prediction)")
    print(f"Corpus size: {len(corpus)} sentences, {sum(len(s.split()) for s in corpus)} words")
    print()
    
    # Measure memory usage before creating the model
    gc.collect()
    base_memory = get_memory_usage()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create and train the model
    print("Training HDC-MoE Language Model...")
    start_time = time.time()
    model = HDCMoELanguageModel(n=n, dimensions=dimensions, n_experts=n_experts, 
                               sparsity=sparsity, top_k_experts=top_k_experts)
    model.fit(corpus)
    train_time = time.time() - start_time
    
    # Measure memory usage
    memory_usage = get_memory_usage() - base_memory
    memory_stats = model.memory_usage()
    
    # Generate text
    seed = ["the", "quick"]
    print(f"\nGenerating text from seed: '{' '.join(seed)}'")
    start_time = time.time()
    generated = model.generate(seed, length=10)
    generation_time = time.time() - start_time
    
    print(f"Generated: '{' '.join(generated)}'")
    print(f"Generation time: {generation_time:.4f} seconds")
    
    # Try different seeds
    seeds = [
        ["the", "cat"],
        ["a", "stitch"],
        ["actions", "speak"],
        ["practice", "makes"]
    ]
    
    print("\nGenerating text from different seeds:")
    for seed in seeds:
        generated = model.generate(seed, length=10)
        print(f"  Seed: '{' '.join(seed)}' -> '{' '.join(generated)}'")
    
    # Get expert statistics
    expert_stats = []
    for i, expert in enumerate(model.experts):
        expert_mem = expert.memory_usage()
        expert_stats.append({
            'id': i,
            'context_vectors': len(expert.context_vectors),
            'memory_usage': expert_mem['total_bytes']
        })
    
    # Sort experts by number of context vectors
    expert_stats.sort(key=lambda x: x['context_vectors'], reverse=True)
    
    print("\nExpert Statistics:")
    print(f"{'ID':<4} {'Context Vectors':<15} {'Memory Usage':<15}")
    print("-" * 40)
    for stat in expert_stats:
        print(f"{stat['id']:<4} {stat['context_vectors']:<15} {format_bytes(stat['memory_usage']):<15}")
    
    # Print memory usage statistics
    print("\nMemory Usage Statistics:")
    print(f"  Total memory usage: {memory_usage:.2f} MB")
    print(f"  Token vectors: {format_bytes(memory_stats['token_vectors_bytes'])}")
    print(f"  Expert total: {format_bytes(memory_stats['expert_total_bytes'])}")
    print(f"  Tracemalloc current: {memory_stats['tracemalloc_current_mb']:.2f} MB")
    print(f"  Tracemalloc peak: {memory_stats['tracemalloc_peak_mb']:.2f} MB")
    
    # Stop memory tracking
    tracemalloc.stop()
    
    # Print vocabulary statistics
    print("\nVocabulary Statistics:")
    print(f"  Vocabulary size: {len(model.vocab)} tokens")
    print(f"  Average token vector size: {sum(v.memory_usage() for v in model.token_vectors) / len(model.token_vectors):.2f} bytes")
    
    # Print training statistics
    print("\nTraining Statistics:")
    print(f"  Training time: {train_time:.4f} seconds")
    
    # Calculate memory efficiency
    bytes_per_token = sum(v.memory_usage() for v in model.token_vectors) / len(model.token_vectors)
    theoretical_bytes = dimensions / 8  # Theoretical minimum for binary vectors (1 bit per dimension)
    efficiency_ratio = theoretical_bytes / bytes_per_token
    
    print("\nMemory Efficiency:")
    print(f"  Bytes per token: {bytes_per_token:.2f} bytes")
    print(f"  Theoretical minimum: {theoretical_bytes:.2f} bytes")
    print(f"  Efficiency ratio: {efficiency_ratio:.2f}x")
    
    # Print sparsity statistics
    avg_nonzeros = sum(len(v.indices) for v in model.token_vectors) / len(model.token_vectors)
    actual_sparsity = 1 - (avg_nonzeros / dimensions)
    
    print("\nSparsity Statistics:")
    print(f"  Target sparsity: {sparsity:.2f}")
    print(f"  Actual sparsity: {actual_sparsity:.2f}")
    print(f"  Average non-zeros per vector: {avg_nonzeros:.2f}")


if __name__ == "__main__":
    main()
