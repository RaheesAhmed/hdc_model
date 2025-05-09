"""
Benchmark comparing HDC-MoE model with DistilBERT.

This script compares the performance, memory usage, and accuracy of the
HDC-MoE model with DistilBERT on a language modeling task.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc
import torch
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import tracemalloc
import argparse
import json
from typing import List, Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc_model.models.hdc_moe import HDCMoELanguageModel


def get_memory_usage():
    """Get the current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_wikitext_sample(file_path: str, max_lines: int = 1000) -> List[str]:
    """
    Load a sample from the WikiText dataset.
    
    Args:
        file_path: Path to the WikiText file.
        max_lines: Maximum number of lines to load.
        
    Returns:
        List of text strings.
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if line and not line.startswith('='):
                texts.append(line)
    return texts


def benchmark_hdc_moe(texts: List[str], n: int = 3, dimensions: int = 10000, 
                      n_experts: int = 10, sparsity: float = 0.9, 
                      top_k_experts: int = 2) -> Dict[str, Any]:
    """
    Benchmark the HDC-MoE model.
    
    Args:
        texts: List of text strings.
        n: The size of the n-grams.
        dimensions: The dimensionality of the hypervectors.
        n_experts: The number of experts in the mixture.
        sparsity: The sparsity of the hypervectors.
        top_k_experts: The number of top experts to use for prediction.
        
    Returns:
        Dictionary with benchmark results.
    """
    print(f"Benchmarking HDC-MoE (n={n}, dimensions={dimensions}, n_experts={n_experts}, sparsity={sparsity})")
    
    # Split texts into train and test
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    test_texts = texts[train_size:]
    
    # Measure memory usage before creating the model
    gc.collect()
    base_memory = get_memory_usage()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create and train the model
    start_time = time.time()
    model = HDCMoELanguageModel(n=n, dimensions=dimensions, n_experts=n_experts, 
                               sparsity=sparsity, top_k_experts=top_k_experts)
    model.fit(train_texts)
    train_time = time.time() - start_time
    
    # Measure memory usage
    memory_usage = get_memory_usage() - base_memory
    memory_stats = model.memory_usage()
    
    # Evaluate on test set
    correct = 0
    total = 0
    start_time = time.time()
    
    for text in test_texts:
        tokens = model.tokenizer.tokenize(text)
        if len(tokens) < n:
            continue
        
        for i in range(len(tokens) - n + 1):
            context = tokens[i:i+n-1]
            target = tokens[i+n-1]
            
            predictions = model.predict_next(context, top_k=1)
            if predictions and predictions[0][0] == target:
                correct += 1
            total += 1
    
    eval_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Generate some text
    seed = model.tokenizer.tokenize(test_texts[0])[:n-1]
    start_time = time.time()
    generated = model.generate(seed, length=20)
    generation_time = time.time() - start_time
    
    # Stop memory tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'model': 'HDC-MoE',
        'parameters': {
            'n': n,
            'dimensions': dimensions,
            'n_experts': n_experts,
            'sparsity': sparsity,
            'top_k_experts': top_k_experts
        },
        'performance': {
            'train_time': train_time,
            'eval_time': eval_time,
            'generation_time': generation_time,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        },
        'memory': {
            'usage_mb': memory_usage,
            'tracemalloc_current_mb': current / (1024 * 1024),
            'tracemalloc_peak_mb': peak / (1024 * 1024),
            'stats': memory_stats
        },
        'generation': {
            'seed': ' '.join(seed),
            'generated': ' '.join(generated)
        }
    }


def benchmark_distilbert(texts: List[str]) -> Dict[str, Any]:
    """
    Benchmark DistilBERT.
    
    Args:
        texts: List of text strings.
        
    Returns:
        Dictionary with benchmark results.
    """
    print("Benchmarking DistilBERT")
    
    # Split texts into train and test
    train_size = int(0.8 * len(texts))
    test_texts = texts[train_size:]
    
    # Measure memory usage before creating the model
    gc.collect()
    base_memory = get_memory_usage()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Load model and tokenizer
    start_time = time.time()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    load_time = time.time() - start_time
    
    # Measure memory usage
    memory_usage = get_memory_usage() - base_memory
    
    # Evaluate on test set
    correct = 0
    total = 0
    start_time = time.time()
    
    for text in test_texts:
        # Tokenize
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 3:
            continue
        
        for i in range(len(tokens) - 2):
            # Get context and target
            context = tokens[i:i+2]
            target = tokens[i+2]
            
            # Convert to input IDs
            input_ids = tokenizer.convert_tokens_to_ids(context + ['[MASK]'])
            input_ids = torch.tensor([input_ids])
            
            # Get predictions
            with torch.no_grad():
                outputs = model(input_ids)
                predictions = outputs.logits
            
            # Get the predicted token
            predicted_id = torch.argmax(predictions[0, -1]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_id])[0]
            
            if predicted_token == target:
                correct += 1
            total += 1
    
    eval_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Generate some text
    seed = tokenizer.tokenize(test_texts[0])[:2]
    start_time = time.time()
    
    # Simple generation
    generated = seed.copy()
    for _ in range(20):
        # Convert to input IDs
        input_ids = tokenizer.convert_tokens_to_ids(generated[-2:] + ['[MASK]'])
        input_ids = torch.tensor([input_ids])
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits
        
        # Get the predicted token
        predicted_id = torch.argmax(predictions[0, -1]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_id])[0]
        
        generated.append(predicted_token)
    
    generation_time = time.time() - start_time
    
    # Stop memory tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Get model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        'model': 'DistilBERT',
        'parameters': {
            'model_size_mb': model_size
        },
        'performance': {
            'load_time': load_time,
            'eval_time': eval_time,
            'generation_time': generation_time,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        },
        'memory': {
            'usage_mb': memory_usage,
            'tracemalloc_current_mb': current / (1024 * 1024),
            'tracemalloc_peak_mb': peak / (1024 * 1024)
        },
        'generation': {
            'seed': ' '.join(seed),
            'generated': ' '.join(generated)
        }
    }


def plot_results(hdc_results: Dict[str, Any], distilbert_results: Dict[str, Any]) -> None:
    """
    Plot the benchmark results.
    
    Args:
        hdc_results: Results from the HDC-MoE benchmark.
        distilbert_results: Results from the DistilBERT benchmark.
    """
    plt.figure(figsize=(15, 10))
    
    # Memory usage
    plt.subplot(2, 2, 1)
    models = ['HDC-MoE', 'DistilBERT']
    memory = [hdc_results['memory']['usage_mb'], distilbert_results['memory']['usage_mb']]
    plt.bar(models, memory)
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    
    # Evaluation time
    plt.subplot(2, 2, 2)
    eval_time = [hdc_results['performance']['eval_time'], distilbert_results['performance']['eval_time']]
    plt.bar(models, eval_time)
    plt.ylabel('Evaluation Time (s)')
    plt.title('Evaluation Time Comparison')
    
    # Generation time
    plt.subplot(2, 2, 3)
    gen_time = [hdc_results['performance']['generation_time'], distilbert_results['performance']['generation_time']]
    plt.bar(models, gen_time)
    plt.ylabel('Generation Time (s)')
    plt.title('Generation Time Comparison')
    
    # Accuracy
    plt.subplot(2, 2, 4)
    accuracy = [hdc_results['performance']['accuracy'], distilbert_results['performance']['accuracy']]
    plt.bar(models, accuracy)
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    
    plt.tight_layout()
    plt.savefig('distilbert_comparison.png')
    plt.close()


def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark HDC-MoE vs DistilBERT')
    parser.add_argument('--wikitext', type=str, default='wikitext-103-raw/wiki.train.raw',
                        help='Path to WikiText dataset')
    parser.add_argument('--max-lines', type=int, default=1000,
                        help='Maximum number of lines to load from WikiText')
    parser.add_argument('--dimensions', type=int, default=10000,
                        help='Dimensionality of hypervectors')
    parser.add_argument('--n-experts', type=int, default=10,
                        help='Number of experts in the mixture')
    parser.add_argument('--sparsity', type=float, default=0.9,
                        help='Sparsity of hypervectors')
    parser.add_argument('--top-k-experts', type=int, default=2,
                        help='Number of top experts to use for prediction')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for benchmark results')
    args = parser.parse_args()
    
    # Load WikiText sample
    try:
        texts = load_wikitext_sample(args.wikitext, args.max_lines)
    except FileNotFoundError:
        print(f"WikiText file not found: {args.wikitext}")
        print("Please download WikiText-103 from https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/")
        print("Using a small sample corpus instead.")
        
        # Use a small sample corpus
        texts = [
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
        ] * 50  # Repeat to get a larger corpus
    
    print(f"Loaded {len(texts)} texts")
    
    # Run benchmarks
    hdc_results = benchmark_hdc_moe(
        texts, 
        dimensions=args.dimensions,
        n_experts=args.n_experts,
        sparsity=args.sparsity,
        top_k_experts=args.top_k_experts
    )
    
    distilbert_results = benchmark_distilbert(texts)
    
    # Calculate improvement ratios
    memory_ratio = distilbert_results['memory']['usage_mb'] / hdc_results['memory']['usage_mb']
    eval_speedup = distilbert_results['performance']['eval_time'] / hdc_results['performance']['eval_time']
    gen_speedup = distilbert_results['performance']['generation_time'] / hdc_results['performance']['generation_time']
    accuracy_ratio = hdc_results['performance']['accuracy'] / distilbert_results['performance']['accuracy']
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Memory Usage: HDC-MoE = {hdc_results['memory']['usage_mb']:.2f} MB, DistilBERT = {distilbert_results['memory']['usage_mb']:.2f} MB")
    print(f"Memory Ratio: {memory_ratio:.2f}x less memory for HDC-MoE")
    print(f"Evaluation Time: HDC-MoE = {hdc_results['performance']['eval_time']:.2f}s, DistilBERT = {distilbert_results['performance']['eval_time']:.2f}s")
    print(f"Evaluation Speedup: {eval_speedup:.2f}x faster for HDC-MoE")
    print(f"Generation Time: HDC-MoE = {hdc_results['performance']['generation_time']:.2f}s, DistilBERT = {distilbert_results['performance']['generation_time']:.2f}s")
    print(f"Generation Speedup: {gen_speedup:.2f}x faster for HDC-MoE")
    print(f"Accuracy: HDC-MoE = {hdc_results['performance']['accuracy']:.4f}, DistilBERT = {distilbert_results['performance']['accuracy']:.4f}")
    print(f"Accuracy Ratio: {accuracy_ratio:.2f}x (target: >0.99x)")
    
    # Plot results
    plot_results(hdc_results, distilbert_results)
    
    # Save results
    results = {
        'hdc_moe': hdc_results,
        'distilbert': distilbert_results,
        'comparison': {
            'memory_ratio': memory_ratio,
            'eval_speedup': eval_speedup,
            'gen_speedup': gen_speedup,
            'accuracy_ratio': accuracy_ratio
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"Plot saved to distilbert_comparison.png")


if __name__ == "__main__":
    main()
