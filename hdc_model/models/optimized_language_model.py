"""
Optimized language model implementation using Hyperdimensional Computing.

This module implements a memory-efficient language model using HDC principles,
with bit-packed binary vectors for reduced memory usage and improved performance.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ..core import optimized_operations as opt_ops
from ..text import tokenizer


class OptimizedNGramLanguageModel:
    """
    An optimized n-gram language model using Hyperdimensional Computing.
    
    This class implements a memory-efficient n-gram language model where sequences of
    n-1 tokens are used to predict the next token, using bit-packed binary vectors.
    """
    
    def __init__(self, n: int = 3, dimensions: int = 10000):
        """
        Initialize the language model.
        
        Args:
            n: The size of the n-grams.
            dimensions: The dimensionality of the hypervectors.
        """
        self.n = n
        self.dimensions = dimensions
        
        # Create a tokenizer
        self.tokenizer = tokenizer.SimpleTokenizer()
        
        # Store the vocabulary and token vectors
        self.vocab = []
        self.token_vectors = []
        
        # Store n-gram associations
        self.context_vectors = []
        self.target_vectors = []
    
    def _encode_token(self, token: str) -> opt_ops.BitPackedVector:
        """
        Encode a token as a bit-packed hypervector.
        
        Args:
            token: The token to encode.
            
        Returns:
            A bit-packed hypervector for the token.
        """
        # Check if the token is in the vocabulary
        if token in self.vocab:
            idx = self.vocab.index(token)
            return self.token_vectors[idx]
        
        # If not, create a new random vector
        vector = opt_ops.BitPackedVector.random(self.dimensions)
        self.vocab.append(token)
        self.token_vectors.append(vector)
        
        return vector
    
    def _encode_context(self, context: List[str]) -> opt_ops.BitPackedVector:
        """
        Encode a context (sequence of tokens) as a bit-packed hypervector.
        
        Args:
            context: The list of tokens in the context.
            
        Returns:
            A bit-packed hypervector for the context.
        """
        # Encode each token and apply position-dependent permutation
        vectors = []
        for i, token in enumerate(context):
            token_vector = self._encode_token(token)
            # Apply i permutations to encode position
            permuted = token_vector
            for _ in range(i):
                permuted = opt_ops.permute_bit_packed(permuted)
            vectors.append(permuted)
        
        # Bundle the permuted vectors
        return opt_ops.bundle_bit_packed(vectors)
    
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
                
                # Store the association
                self.context_vectors.append(context_vector)
                self.target_vectors.append(target_vector)
    
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
        
        # Calculate similarity with all stored contexts
        similarities = []
        for stored_context in self.context_vectors:
            # Use Hamming distance for binary vectors (convert to similarity)
            distance = opt_ops.hamming_distance_bit_packed(context_vector, stored_context)
            similarity = 1.0 - distance
            similarities.append(similarity)
        
        if not similarities:
            return []
        
        # Get the indices of the top k similar contexts
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Get the corresponding target tokens and similarities
        predictions = []
        for idx in top_indices:
            target_vector = self.target_vectors[idx]
            
            # Find the most similar token in the vocabulary
            token_similarities = []
            for token_vector in self.token_vectors:
                distance = opt_ops.hamming_distance_bit_packed(target_vector, token_vector)
                token_similarities.append(1.0 - distance)
            
            token_idx = np.argmax(token_similarities)
            token = self.vocab[token_idx]
            score = similarities[idx]
            
            predictions.append((token, score))
        
        return predictions
    
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
    
    def memory_usage(self) -> Dict[str, int]:
        """
        Calculate the memory usage of the model.
        
        Returns:
            A dictionary with memory usage statistics.
        """
        # Calculate memory usage of token vectors
        token_vectors_bytes = sum(vec.packed_data.nbytes for vec in self.token_vectors)
        
        # Calculate memory usage of context vectors
        context_vectors_bytes = sum(vec.packed_data.nbytes for vec in self.context_vectors)
        
        # Calculate memory usage of target vectors
        target_vectors_bytes = sum(vec.packed_data.nbytes for vec in self.target_vectors)
        
        # Calculate total memory usage
        total_bytes = token_vectors_bytes + context_vectors_bytes + target_vectors_bytes
        
        return {
            'token_vectors_bytes': token_vectors_bytes,
            'context_vectors_bytes': context_vectors_bytes,
            'target_vectors_bytes': target_vectors_bytes,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024)
        }
