"""
Text encoding utilities for HDC.

This module provides utilities for encoding tokenized text data as hypervectors
for use with Hyperdimensional Computing models.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from ..core import operations


class RandomEncoder:
    """
    A random encoder for text tokens.
    
    This class implements a simple encoder that assigns a random hypervector
    to each token in the vocabulary.
    """
    
    def __init__(self, dimensions: int, vector_type: str = 'binary'):
        """
        Initialize the encoder.
        
        Args:
            dimensions: The dimensionality of the hypervectors.
            vector_type: The type of hypervectors to use ('binary' or 'bipolar').
        """
        self.dimensions = dimensions
        self.vector_type = vector_type
        self.token_vectors = {}
    
    def fit(self, vocab: List[str]) -> None:
        """
        Generate random vectors for each token in the vocabulary.
        
        Args:
            vocab: The list of tokens in the vocabulary.
        """
        for token in vocab:
            if self.vector_type == 'binary':
                self.token_vectors[token] = operations.random_binary(self.dimensions)
            elif self.vector_type == 'bipolar':
                self.token_vectors[token] = operations.random_bipolar(self.dimensions)
            else:
                raise ValueError(f"Unsupported vector type: {self.vector_type}")
    
    def encode_token(self, token: str) -> np.ndarray:
        """
        Encode a single token as a hypervector.
        
        Args:
            token: The token to encode.
            
        Returns:
            The hypervector for the token.
        """
        if token in self.token_vectors:
            return self.token_vectors[token]
        else:
            # Generate a new random vector for unknown tokens
            if self.vector_type == 'binary':
                vector = operations.random_binary(self.dimensions)
            elif self.vector_type == 'bipolar':
                vector = operations.random_bipolar(self.dimensions)
            else:
                raise ValueError(f"Unsupported vector type: {self.vector_type}")
            
            self.token_vectors[token] = vector
            return vector
    
    def encode_sequence(self, tokens: List[str]) -> np.ndarray:
        """
        Encode a sequence of tokens as a single hypervector.
        
        Args:
            tokens: The list of tokens to encode.
            
        Returns:
            The hypervector for the sequence.
        """
        if not tokens:
            # Return a zero vector for empty sequences
            if self.vector_type == 'binary':
                return np.zeros(self.dimensions, dtype=np.int8)
            elif self.vector_type == 'bipolar':
                return np.zeros(self.dimensions, dtype=np.int8)
        
        # Encode each token and permute based on position
        vectors = []
        for i, token in enumerate(tokens):
            token_vector = self.encode_token(token)
            permuted = operations.permute(token_vector, i)
            vectors.append(permuted)
        
        # Bundle the permuted vectors
        return operations.bundle(vectors)
    
    def similarity(self, token1: str, token2: str) -> float:
        """
        Calculate the similarity between two tokens.
        
        Args:
            token1: The first token.
            token2: The second token.
            
        Returns:
            The cosine similarity between the token vectors.
        """
        vector1 = self.encode_token(token1)
        vector2 = self.encode_token(token2)
        return operations.cosine_similarity(vector1, vector2)


class NGramEncoder:
    """
    An n-gram based encoder for text.
    
    This class implements an encoder that represents tokens as a bundle
    of their character n-grams.
    """
    
    def __init__(self, dimensions: int, n: int = 3, vector_type: str = 'binary'):
        """
        Initialize the encoder.
        
        Args:
            dimensions: The dimensionality of the hypervectors.
            n: The size of the character n-grams.
            vector_type: The type of hypervectors to use ('binary' or 'bipolar').
        """
        self.dimensions = dimensions
        self.n = n
        self.vector_type = vector_type
        self.ngram_vectors = {}
    
    def _get_ngrams(self, token: str) -> List[str]:
        """
        Extract character n-grams from a token.
        
        Args:
            token: The token to process.
            
        Returns:
            A list of character n-grams.
        """
        # Add start and end markers
        padded = '#' + token + '#'
        
        # Extract n-grams
        ngrams = []
        for i in range(len(padded) - self.n + 1):
            ngrams.append(padded[i:i+self.n])
        
        return ngrams
    
    def fit(self, vocab: List[str]) -> None:
        """
        Generate random vectors for each unique n-gram in the vocabulary.
        
        Args:
            vocab: The list of tokens in the vocabulary.
        """
        # Extract all unique n-grams from the vocabulary
        all_ngrams = set()
        for token in vocab:
            all_ngrams.update(self._get_ngrams(token))
        
        # Generate random vectors for each n-gram
        for ngram in all_ngrams:
            if self.vector_type == 'binary':
                self.ngram_vectors[ngram] = operations.random_binary(self.dimensions)
            elif self.vector_type == 'bipolar':
                self.ngram_vectors[ngram] = operations.random_bipolar(self.dimensions)
            else:
                raise ValueError(f"Unsupported vector type: {self.vector_type}")
    
    def encode_token(self, token: str) -> np.ndarray:
        """
        Encode a single token as a hypervector.
        
        Args:
            token: The token to encode.
            
        Returns:
            The hypervector for the token.
        """
        # Extract n-grams from the token
        ngrams = self._get_ngrams(token)
        
        if not ngrams:
            # Return a zero vector for empty tokens
            if self.vector_type == 'binary':
                return np.zeros(self.dimensions, dtype=np.int8)
            elif self.vector_type == 'bipolar':
                return np.zeros(self.dimensions, dtype=np.int8)
        
        # Get the vector for each n-gram
        vectors = []
        for ngram in ngrams:
            if ngram in self.ngram_vectors:
                vectors.append(self.ngram_vectors[ngram])
            else:
                # Generate a new random vector for unknown n-grams
                if self.vector_type == 'binary':
                    vector = operations.random_binary(self.dimensions)
                elif self.vector_type == 'bipolar':
                    vector = operations.random_bipolar(self.dimensions)
                else:
                    raise ValueError(f"Unsupported vector type: {self.vector_type}")
                
                self.ngram_vectors[ngram] = vector
                vectors.append(vector)
        
        # Bundle the n-gram vectors
        return operations.bundle(vectors)
    
    def encode_sequence(self, tokens: List[str]) -> np.ndarray:
        """
        Encode a sequence of tokens as a single hypervector.
        
        Args:
            tokens: The list of tokens to encode.
            
        Returns:
            The hypervector for the sequence.
        """
        if not tokens:
            # Return a zero vector for empty sequences
            if self.vector_type == 'binary':
                return np.zeros(self.dimensions, dtype=np.int8)
            elif self.vector_type == 'bipolar':
                return np.zeros(self.dimensions, dtype=np.int8)
        
        # Encode each token and permute based on position
        vectors = []
        for i, token in enumerate(tokens):
            token_vector = self.encode_token(token)
            permuted = operations.permute(token_vector, i)
            vectors.append(permuted)
        
        # Bundle the permuted vectors
        return operations.bundle(vectors)
    
    def similarity(self, token1: str, token2: str) -> float:
        """
        Calculate the similarity between two tokens.
        
        Args:
            token1: The first token.
            token2: The second token.
            
        Returns:
            The cosine similarity between the token vectors.
        """
        vector1 = self.encode_token(token1)
        vector2 = self.encode_token(token2)
        return operations.cosine_similarity(vector1, vector2)
