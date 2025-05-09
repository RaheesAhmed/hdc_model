"""
Memory usage benchmark for HDC operations.

This script compares the memory usage of the original and optimized HDC operations.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc_model.core import operations
from hdc_model.core import optimized_operations as opt_ops


def get_memory_usage():
    """Get the current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_memory_usage(dimensions_list, n_vectors=100):
    """
    Benchmark memory usage for different vector dimensions.
    
    Args:
        dimensions_list: List of dimensions to test.
        n_vectors: Number of vectors to create.
        
    Returns:
        Dictionary with memory usage results.
    """
    results = {
        'dimensions': dimensions_list,
        'original_memory': [],
        'optimized_memory': [],
        'memory_ratio': []
    }
    
    for dimensions in dimensions_list:
        print(f"Benchmarking {dimensions} dimensions...")
        
        # Measure memory usage for original operations
        gc.collect()
        base_memory = get_memory_usage()
        
        # Create vectors using original operations
        original_vectors = [operations.random_binary(dimensions) for _ in range(n_vectors)]
        
        # Measure memory usage
        original_memory = get_memory_usage() - base_memory
        results['original_memory'].append(original_memory)
        
        # Clean up
        del original_vectors
        gc.collect()
        
        # Measure memory usage for optimized operations
        base_memory = get_memory_usage()
        
        # Create vectors using optimized operations
        optimized_vectors = [opt_ops.BitPackedVector.random(dimensions) for _ in range(n_vectors)]
        
        # Measure memory usage
        optimized_memory = get_memory_usage() - base_memory
        results['optimized_memory'].append(optimized_memory)
        
        # Calculate memory ratio
        memory_ratio = original_memory / optimized_memory if optimized_memory > 0 else float('inf')
        results['memory_ratio'].append(memory_ratio)
        
        print(f"  Original: {original_memory:.2f} MB")
        print(f"  Optimized: {optimized_memory:.2f} MB")
        print(f"  Ratio: {memory_ratio:.2f}x")
        
        # Clean up
        del optimized_vectors
        gc.collect()
    
    return results


def plot_results(results):
    """
    Plot the benchmark results.
    
    Args:
        results: Dictionary with benchmark results.
    """
    plt.figure(figsize=(12, 10))
    
    # Plot memory usage
    plt.subplot(2, 1, 1)
    plt.plot(results['dimensions'], results['original_memory'], 'o-', label='Original')
    plt.plot(results['dimensions'], results['optimized_memory'], 'o-', label='Optimized')
    plt.xlabel('Dimensions')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot memory ratio
    plt.subplot(2, 1, 2)
    plt.plot(results['dimensions'], results['memory_ratio'], 'o-')
    plt.xlabel('Dimensions')
    plt.ylabel('Memory Ratio (Original / Optimized)')
    plt.title('Memory Efficiency Improvement')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('memory_benchmark.png')
    plt.close()


def main():
    """Run the memory usage benchmark."""
    # Dimensions to test
    dimensions_list = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    # Number of vectors to create
    n_vectors = 100
    
    # Run the benchmark
    results = benchmark_memory_usage(dimensions_list, n_vectors)
    
    # Plot the results
    plot_results(results)
    
    # Print summary
    print("\nSummary:")
    print(f"Average memory ratio: {np.mean(results['memory_ratio']):.2f}x")
    print(f"Maximum memory ratio: {np.max(results['memory_ratio']):.2f}x at {results['dimensions'][np.argmax(results['memory_ratio'])]} dimensions")


if __name__ == "__main__":
    main()
