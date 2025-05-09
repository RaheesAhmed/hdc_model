"""
HDC-based Mixture of Experts (MoE) language model.

This module implements a novel language model that combines Hyperdimensional Computing
with the Mixture of Experts architecture for efficient language modeling.
"""

import numpy as np
import time
import tracemalloc
from typing import List, Dict, Tuple, Optional, Union, Any
from ..core import improved_operations as ops
from ..text import tokenizer


class HDCExpert:
    """
    An expert in the HDC-MoE model.
    
    Each expert is a specialized predictor that focuses on a specific
    part of the input space.
    """
    
    def __init__(self, dimensions: int, expert_id: int, sparsity: float = 0.9):
        """
        Initialize the expert.
        
        Args:
            dimensions: The dimensionality of the hypervectors.
            expert_id: The ID of the expert.
            sparsity: The sparsity of the hypervectors.
        """
        self.dimensions = dimensions
        self.expert_id = expert_id
        self.sparsity = sparsity
        
        # Create a random ID vector for this expert
        self.id_vector = ops.SparseHDVector.random(dimensions, sparsity)
        
        # Store context-target associations
        self.context_vectors = []
        self.target_vectors = []
    
    def train(self, context_vector: ops.SparseHDVector, target_vector: ops.SparseHDVector) -> None:
        """
        Train the expert on a context-target pair.
        
        Args:
            context_vector: The context hypervector.
            target_vector: The target hypervector.
        """
        self.context_vectors.append(context_vector)
        self.target_vectors.append(target_vector)
    
    def predict(self, context_vector: ops.SparseHDVector, top_k: int = 1) -> List[Tuple[int, float]]:
        """
        Predict the next token given a context.
        
        Args:
            context_vector: The context hypervector.
            top_k: The number of top predictions to return.
            
        Returns:
            A list of (target_index, similarity) tuples.
        """
        if not self.context_vectors:
            return []
        
        # Calculate similarity with all stored contexts
        similarities = ops.batch_hamming_distance(self.context_vectors, context_vector)
        similarities = 1.0 - similarities  # Convert distance to similarity
        
        # Get the indices of the top k similar contexts
        if len(similarities) < top_k:
            top_k = len(similarities)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return the corresponding target indices and similarities
        return [(i, similarities[i]) for i in top_indices]
    
    def memory_usage(self) -> Dict[str, int]:
        """
        Calculate the memory usage of the expert.
        
        Returns:
            A dictionary with memory usage statistics.
        """
        id_vector_bytes = self.id_vector.memory_usage()
        context_vectors_bytes = sum(v.memory_usage() for v in self.context_vectors)
        target_vectors_bytes = sum(v.memory_usage() for v in self.target_vectors)
        
        return {
            'id_vector_bytes': id_vector_bytes,
            'context_vectors_bytes': context_vectors_bytes,
            'target_vectors_bytes': target_vectors_bytes,
            'total_bytes': id_vector_bytes + context_vectors_bytes + target_vectors_bytes
        }


class HDCMoELanguageModel:
    """
    A language model using HDC-based Mixture of Experts.
    
    This model combines Hyperdimensional Computing with the Mixture of Experts
    architecture for efficient language modeling.
    """
    
    def __init__(self, n: int = 3, dimensions: int = 10000, n_experts: int = 10, 
                 sparsity: float = 0.9, top_k_experts: int = 2):
        """
        Initialize the language model.
        
        Args:
            n: The size of the n-grams.
            dimensions: The dimensionality of the hypervectors.
            n_experts: The number of experts in the mixture.
            sparsity: The sparsity of the hypervectors.
            top_k_experts: The number of top experts to use for prediction.
        """
        self.n = n
        self.dimensions = dimensions
        self.n_experts = n_experts
        self.sparsity = sparsity
        self.top_k_experts = top_k_experts
        
        # Create a tokenizer
        self.tokenizer = tokenizer.SimpleTokenizer()
        
        # Create experts
        self.experts = [HDCExpert(dimensions, i, sparsity) for i in range(n_experts)]
        
        # Store the vocabulary and token vectors
        self.vocab = []
        self.token_vectors = []
        
        # Create a vector cache
        self.vector_cache = ops.HDCVectorCache()
        
        # Enable memory tracking
        tracemalloc.start()
    
    def _encode_token(self, token: str) -> ops.SparseHDVector:
        """
        Encode a token as a sparse hypervector.
        
        Args:
            token: The token to encode.
            
        Returns:
            A sparse hypervector for the token.
        """
        # Check if the token is in the cache
        cached_vector = self.vector_cache.get(f"token:{token}")
        if cached_vector is not None:
            return cached_vector
        
        # Check if the token is in the vocabulary
        if token in self.vocab:
            idx = self.vocab.index(token)
            return self.token_vectors[idx]
        
        # If not, create a new random vector
        vector = ops.SparseHDVector.random(self.dimensions, self.sparsity)
        self.vocab.append(token)
        self.token_vectors.append(vector)
        
        # Cache the vector
        self.vector_cache.put(f"token:{token}", vector)
        
        return vector
    
    def _encode_context(self, context: List[str]) -> ops.SparseHDVector:
        """
        Encode a context (sequence of tokens) as a sparse hypervector.
        
        Args:
            context: The list of tokens in the context.
            
        Returns:
            A sparse hypervector for the context.
        """
        # Check if the context is in the cache
        context_key = f"context:{','.join(context)}"
        cached_vector = self.vector_cache.get(context_key)
        if cached_vector is not None:
            return cached_vector
        
        # Encode each token and apply position-dependent permutation
        vectors = []
        for i, token in enumerate(context):
            token_vector = self._encode_token(token)
            # Apply i permutations to encode position
            permuted = token_vector
            for _ in range(i):
                permuted = ops.sparse_permute(permuted)
            vectors.append(permuted)
        
        # Bundle the permuted vectors
        result = ops.sparse_bundle(vectors)
        
        # Cache the result
        self.vector_cache.put(context_key, result)
        
        return result
    
    def _select_experts(self, context_vector: ops.SparseHDVector) -> List[Tuple[int, float]]:
        """
        Select the top-k experts for a given context.
        
        Args:
            context_vector: The context hypervector.
            
        Returns:
            A list of (expert_index, similarity) tuples.
        """
        # Calculate similarity with all expert ID vectors
        expert_id_vectors = [expert.id_vector for expert in self.experts]
        similarities = ops.batch_hamming_distance(expert_id_vectors, context_vector)
        similarities = 1.0 - similarities  # Convert distance to similarity
        
        # Get the indices of the top k similar experts
        top_indices = np.argsort(similarities)[-self.top_k_experts:][::-1]
        
        # Return the corresponding expert indices and similarities
        return [(i, similarities[i]) for i in top_indices]
    
    def fit(self, texts: List[str]) -> None:
        """
        Train the language model on a corpus of texts.
        
        Args:
            texts: The list of text strings to train on.
        """
        # Process each text
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            
            # Skip texts that are too short
            if len(tokens) < self.n:
                continue
            
            # Process each n-gram
            for i in range(len(tokens) - self.n + 1):
                # Get the context (first n-1 tokens)
                context = tokens[i:i+self.n-1]
                
                # Get the target (last token)
                target = tokens[i+self.n-1]
                
                # Encode the context
                context_vector = self._encode_context(context)
                
                # Encode the target
                target_vector = self._encode_token(target)
                
                # Select experts for this context
                expert_indices = self._select_experts(context_vector)
                
                # Train the selected experts
                for expert_idx, _ in expert_indices:
                    self.experts[expert_idx].train(context_vector, target_vector)
    
    def predict_next(self, context: List[str], top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Predict the next token given a context.
        
        Args:
            context: The list of tokens in the context.
            top_k: The number of top predictions to return.
            
        Returns:
            A list of (token, score) tuples for the top k predictions.
        """
        # Ensure the context has the right length
        if len(context) != self.n - 1:
            raise ValueError(f"Context must have {self.n - 1} tokens")
        
        # Encode the context
        context_vector = self._encode_context(context)
        
        # Select experts for this context
        expert_indices = self._select_experts(context_vector)
        
        # Get predictions from each expert
        all_predictions = []
        for expert_idx, expert_weight in expert_indices:
            expert = self.experts[expert_idx]
            expert_predictions = expert.predict(context_vector, top_k)
            
            for target_idx, target_sim in expert_predictions:
                target_vector = expert.target_vectors[target_idx]
                
                # Find the most similar token in the vocabulary
                token_similarities = ops.batch_hamming_distance(self.token_vectors, target_vector)
                token_similarities = 1.0 - token_similarities  # Convert distance to similarity
                
                token_idx = np.argmax(token_similarities)
                token = self.vocab[token_idx]
                score = target_sim * expert_weight
                
                all_predictions.append((token, score))
        
        # Combine predictions from all experts
        combined_predictions = {}
        for token, score in all_predictions:
            if token in combined_predictions:
                combined_predictions[token] += score
            else:
                combined_predictions[token] = score
        
        # Sort by score and return top k
        sorted_predictions = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]
    
    def generate(self, seed: List[str], length: int = 10) -> List[str]:
        """
        Generate a sequence of tokens starting from a seed.
        
        Args:
            seed: The list of tokens to start with.
            length: The number of tokens to generate.
            
        Returns:
            The generated sequence of tokens.
        """
        # Ensure the seed has the right length
        if len(seed) != self.n - 1:
            raise ValueError(f"Seed must have {self.n - 1} tokens")
        
        # Initialize the generated sequence with the seed
        generated = seed.copy()
        
        # Generate tokens
        for _ in range(length):
            # Get the current context
            context = generated[-(self.n-1):]
            
            # Predict the next token
            predictions = self.predict_next(context)
            
            # Add the top prediction to the generated sequence
            if predictions:
                next_token, _ = predictions[0]
                generated.append(next_token)
            else:
                # If no prediction is available, stop generation
                break
        
        return generated
    
    def memory_usage(self) -> Dict[str, Any]:
        """
        Calculate the memory usage of the model.
        
        Returns:
            A dictionary with memory usage statistics.
        """
        # Get current memory usage from tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        
        # Calculate memory usage of token vectors
        token_vectors_bytes = sum(v.memory_usage() for v in self.token_vectors)
        
        # Calculate memory usage of experts
        expert_memory = [expert.memory_usage() for expert in self.experts]
        expert_total_bytes = sum(mem['total_bytes'] for mem in expert_memory)
        
        # Calculate total memory usage
        total_bytes = token_vectors_bytes + expert_total_bytes
        
        return {
            'token_vectors_bytes': token_vectors_bytes,
            'expert_memory': expert_memory,
            'expert_total_bytes': expert_total_bytes,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'tracemalloc_current_mb': current / (1024 * 1024),
            'tracemalloc_peak_mb': peak / (1024 * 1024)
        }
