"""
Example usage of the optimized HDC language model.

This script demonstrates how to use the optimized HDC language model and
compares its memory usage and performance with the original model.
"""

import sys
import os
import time
import psutil

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc_model.models.language_model import NGramLanguageModel
from hdc_model.models.optimized_language_model import OptimizedNGramLanguageModel


def get_memory_usage():
    """Get the current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


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
    
    print("Comparing Original vs. Optimized HDC Language Models")
    print("=" * 60)
    print(f"Parameters: n={n}, dimensions={dimensions}")
    print(f"Corpus size: {len(corpus)} sentences, {sum(len(s.split()) for s in corpus)} words")
    print()
    
    # Measure memory usage before creating any models
    base_memory = get_memory_usage()
    
    # Train the original model
    print("Training Original N-Gram Language Model...")
    start_time = time.time()
    original_model = NGramLanguageModel(n=n, dimensions=dimensions, vector_type='binary')
    original_model.fit(corpus)
    original_train_time = time.time() - start_time
    
    # Measure memory usage
    original_memory = get_memory_usage() - base_memory
    
    # Generate text with the original model
    seed = ["the", "quick"]
    start_time = time.time()
    original_generated = original_model.generate(seed, length=10)
    original_generate_time = time.time() - start_time
    
    print(f"Original Model Seed: {' '.join(seed)}")
    print(f"Original Model Generated: {' '.join(original_generated)}")
    print(f"Original Model Training Time: {original_train_time:.4f} seconds")
    print(f"Original Model Generation Time: {original_generate_time:.4f} seconds")
    print(f"Original Model Memory Usage: {original_memory:.2f} MB")
    print()
    
    # Clean up to measure optimized model memory accurately
    del original_model
    
    # Reset memory baseline
    base_memory = get_memory_usage()
    
    # Train the optimized model
    print("Training Optimized N-Gram Language Model...")
    start_time = time.time()
    optimized_model = OptimizedNGramLanguageModel(n=n, dimensions=dimensions)
    optimized_model.fit(corpus)
    optimized_train_time = time.time() - start_time
    
    # Measure memory usage
    optimized_memory = get_memory_usage() - base_memory
    
    # Generate text with the optimized model
    start_time = time.time()
    optimized_generated = optimized_model.generate(seed, length=10)
    optimized_generate_time = time.time() - start_time
    
    print(f"Optimized Model Seed: {' '.join(seed)}")
    print(f"Optimized Model Generated: {' '.join(optimized_generated)}")
    print(f"Optimized Model Training Time: {optimized_train_time:.4f} seconds")
    print(f"Optimized Model Generation Time: {optimized_generate_time:.4f} seconds")
    print(f"Optimized Model Memory Usage: {optimized_memory:.2f} MB")
    print()
    
    # Calculate improvement ratios
    memory_ratio = original_memory / optimized_memory if optimized_memory > 0 else float('inf')
    train_speedup = original_train_time / optimized_train_time if optimized_train_time > 0 else float('inf')
    generate_speedup = original_generate_time / optimized_generate_time if optimized_generate_time > 0 else float('inf')
    
    print("Performance Comparison:")
    print(f"Memory Efficiency: {memory_ratio:.2f}x improvement")
    print(f"Training Speed: {train_speedup:.2f}x speedup")
    print(f"Generation Speed: {generate_speedup:.2f}x speedup")
    
    # Get detailed memory usage from the optimized model
    memory_stats = optimized_model.memory_usage()
    print("\nOptimized Model Memory Breakdown:")
    print(f"Token Vectors: {memory_stats['token_vectors_bytes'] / 1024:.2f} KB")
    print(f"Context Vectors: {memory_stats['context_vectors_bytes'] / 1024:.2f} KB")
    print(f"Target Vectors: {memory_stats['target_vectors_bytes'] / 1024:.2f} KB")
    print(f"Total: {memory_stats['total_mb']:.2f} MB")


if __name__ == "__main__":
    main()
