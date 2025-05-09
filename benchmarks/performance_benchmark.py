"""
Performance benchmark for HDC operations.

This script compares the performance of the original and optimized HDC operations.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc_model.core import operations
from hdc_model.core import optimized_operations as opt_ops


def benchmark_operation(operation_name, dimensions_list, n_iterations=100):
    """
    Benchmark an operation for different vector dimensions.
    
    Args:
        operation_name: Name of the operation to benchmark.
        dimensions_list: List of dimensions to test.
        n_iterations: Number of iterations for each test.
        
    Returns:
        Dictionary with timing results.
    """
    results = {
        'dimensions': dimensions_list,
        'original_time': [],
        'optimized_time': [],
        'speedup': []
    }
    
    for dimensions in dimensions_list:
        print(f"Benchmarking {operation_name} with {dimensions} dimensions...")
        
        # Prepare data for the operation
        if operation_name == 'bind':
            # Original operations
            x_array = operations.random_binary(dimensions)
            y_array = operations.random_binary(dimensions)
            
            # Optimized operations
            x_packed = opt_ops.BitPackedVector.from_binary_array(x_array)
            y_packed = opt_ops.BitPackedVector.from_binary_array(y_array)
            
            # Functions to benchmark
            original_func = lambda: operations.bind_binary(x_array, y_array)
            optimized_func = lambda: opt_ops.bind_bit_packed(x_packed, y_packed)
            
        elif operation_name == 'bundle':
            # Number of vectors to bundle
            n_vectors = 10
            
            # Original operations
            arrays = [operations.random_binary(dimensions) for _ in range(n_vectors)]
            
            # Optimized operations
            packed_vectors = [opt_ops.BitPackedVector.from_binary_array(arr) for arr in arrays]
            
            # Functions to benchmark
            original_func = lambda: operations.bundle_binary(arrays)
            optimized_func = lambda: opt_ops.bundle_bit_packed(packed_vectors)
            
        elif operation_name == 'hamming':
            # Original operations
            x_array = operations.random_binary(dimensions)
            y_array = operations.random_binary(dimensions)
            
            # Optimized operations
            x_packed = opt_ops.BitPackedVector.from_binary_array(x_array)
            y_packed = opt_ops.BitPackedVector.from_binary_array(y_array)
            
            # Functions to benchmark
            original_func = lambda: operations.hamming_distance(x_array, y_array)
            optimized_func = lambda: opt_ops.hamming_distance_bit_packed(x_packed, y_packed)
            
        elif operation_name == 'permute':
            # Original operations
            x_array = operations.random_binary(dimensions)
            
            # Optimized operations
            x_packed = opt_ops.BitPackedVector.from_binary_array(x_array)
            
            # Functions to benchmark
            original_func = lambda: operations.permute(x_array)
            optimized_func = lambda: opt_ops.permute_bit_packed(x_packed)
            
        else:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        # Benchmark original operations
        start_time = time.time()
        for _ in range(n_iterations):
            original_func()
        original_time = (time.time() - start_time) / n_iterations
        results['original_time'].append(original_time)
        
        # Benchmark optimized operations
        start_time = time.time()
        for _ in range(n_iterations):
            optimized_func()
        optimized_time = (time.time() - start_time) / n_iterations
        results['optimized_time'].append(optimized_time)
        
        # Calculate speedup
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        results['speedup'].append(speedup)
        
        print(f"  Original: {original_time * 1000:.2f} ms")
        print(f"  Optimized: {optimized_time * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    return results


def plot_results(results_dict):
    """
    Plot the benchmark results.
    
    Args:
        results_dict: Dictionary with benchmark results for each operation.
    """
    operations = list(results_dict.keys())
    n_operations = len(operations)
    
    plt.figure(figsize=(15, 10))
    
    # Plot execution time
    for i, operation in enumerate(operations):
        results = results_dict[operation]
        
        plt.subplot(2, n_operations, i + 1)
        plt.plot(results['dimensions'], [t * 1000 for t in results['original_time']], 'o-', label='Original')
        plt.plot(results['dimensions'], [t * 1000 for t in results['optimized_time']], 'o-', label='Optimized')
        plt.xlabel('Dimensions')
        plt.ylabel('Time (ms)')
        plt.title(f'{operation.capitalize()} Operation')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, n_operations, i + 1 + n_operations)
        plt.plot(results['dimensions'], results['speedup'], 'o-')
        plt.xlabel('Dimensions')
        plt.ylabel('Speedup (Original / Optimized)')
        plt.title(f'{operation.capitalize()} Speedup')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png')
    plt.close()


def main():
    """Run the performance benchmark."""
    # Dimensions to test
    dimensions_list = [1000, 5000, 10000, 50000, 100000]
    
    # Operations to benchmark
    operations = ['bind', 'bundle', 'hamming', 'permute']
    
    # Number of iterations for each test
    n_iterations = 100
    
    # Run the benchmarks
    results_dict = {}
    for operation in operations:
        results_dict[operation] = benchmark_operation(operation, dimensions_list, n_iterations)
    
    # Plot the results
    plot_results(results_dict)
    
    # Print summary
    print("\nSummary:")
    for operation in operations:
        results = results_dict[operation]
        avg_speedup = np.mean(results['speedup'])
        max_speedup = np.max(results['speedup'])
        max_dim = results['dimensions'][np.argmax(results['speedup'])]
        print(f"{operation.capitalize()}:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Maximum speedup: {max_speedup:.2f}x at {max_dim} dimensions")


if __name__ == "__main__":
    main()
