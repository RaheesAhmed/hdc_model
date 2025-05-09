"""
Language model implementation using Hyperdimensional Computing.

This module implements a simple language model using HDC principles,
where context is represented as a hypervector and used to predict the next token.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ..core import operations
from ..structures import hash_table, sequence
from ..text import tokenizer, encoder


class NGramLanguageModel:
    """
    An n-gram language model using Hyperdimensional Computing.
    
    This class implements a simple n-gram language model where sequences of
    n-1 tokens are used to predict the next token.
    """
    
    def __init__(self, n: int = 3, dimensions: int = 10000, vector_type: str = 'binary'):
        """
        Initialize the language model.
        
        Args:
            n: The size of the n-grams.
            dimensions: The dimensionality of the hypervectors.
            vector_type: The type of hypervectors to use ('binary' or 'bipolar').
        """
        self.n = n
        self.dimensions = dimensions
        self.vector_type = vector_type
        
        # Create a tokenizer
        self.tokenizer = tokenizer.SimpleTokenizer()
        
        # Create an encoder
        self.encoder = encoder.RandomEncoder(dimensions, vector_type)
        
        # Create a hash table for storing n-gram associations
        self.ngram_table = hash_table.HashTable(dimensions, vector_type)
        
        # Store the vocabulary
        self.vocab = []
        self.vocab_vectors = []
    
    def fit(self, texts: List[str]) -> None:
        """
        Train the language model on a corpus of texts.
        
        Args:
            texts: The list of text strings to train on.
        """
        # Tokenize the texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        
        # Build the vocabulary
        self.vocab = sorted(set(all_tokens))
        
        # Fit the encoder
        self.encoder.fit(self.vocab)
        
        # Store the vocabulary vectors
        self.vocab_vectors = [self.encoder.encode_token(token) for token in self.vocab]
        
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
                context_vector = self.encoder.encode_sequence(context)
                
                # Encode the target
                target_vector = self.encoder.encode_token(target)
                
                # Add to the hash table
                self.ngram_table.add(context_vector, target_vector)
    
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
        context_vector = self.encoder.encode_sequence(context)
        
        # Query the hash table
        prediction_vector = self.ngram_table.get(context_vector)
        
        # Calculate similarity with all vocabulary vectors
        similarities = [operations.cosine_similarity(prediction_vector, vec) for vec in self.vocab_vectors]
        
        # Get the top k predictions
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_predictions = [(self.vocab[i], similarities[i]) for i in top_indices]
        
        return top_predictions
    
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
    
    def perplexity(self, text: str) -> float:
        """
        Calculate the perplexity of the model on a text.
        
        Args:
            text: The text to evaluate.
            
        Returns:
            The perplexity score (lower is better).
        """
        tokens = self.tokenizer.tokenize(text)
        
        # Skip texts that are too short
        if len(tokens) < self.n:
            return float('inf')
        
        # Calculate log probability for each n-gram
        log_prob_sum = 0.0
        count = 0
        
        for i in range(len(tokens) - self.n + 1):
            # Get the context
            context = tokens[i:i+self.n-1]
            
            # Get the target
            target = tokens[i+self.n-1]
            
            # Predict the next token
            predictions = self.predict_next(context)
            
            # Find the target in the predictions
            target_prob = 0.0
            for token, score in predictions:
                if token == target:
                    # Convert similarity to probability
                    target_prob = (score + 1) / 2  # Scale from [-1, 1] to [0, 1]
                    break
            
            # Add log probability
            if target_prob > 0:
                log_prob_sum += np.log2(target_prob)
                count += 1
        
        # Calculate perplexity
        if count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / count
        perplexity = 2 ** (-avg_log_prob)
        
        return perplexity


class ContextualLanguageModel:
    """
    A contextual language model using Hyperdimensional Computing.
    
    This class implements a more advanced language model that uses a sliding
    window of context with position-dependent encoding.
    """
    
    def __init__(self, context_size: int = 5, dimensions: int = 10000, vector_type: str = 'binary'):
        """
        Initialize the language model.
        
        Args:
            context_size: The size of the context window.
            dimensions: The dimensionality of the hypervectors.
            vector_type: The type of hypervectors to use ('binary' or 'bipolar').
        """
        self.context_size = context_size
        self.dimensions = dimensions
        self.vector_type = vector_type
        
        # Create a tokenizer
        self.tokenizer = tokenizer.SimpleTokenizer()
        
        # Create an encoder
        self.encoder = encoder.NGramEncoder(dimensions, n=3, vector_type=vector_type)
        
        # Create a hash table for storing context-target associations
        self.context_table = hash_table.HashTable(dimensions, vector_type)
        
        # Store the vocabulary
        self.vocab = []
        self.vocab_vectors = []
    
    def fit(self, texts: List[str]) -> None:
        """
        Train the language model on a corpus of texts.
        
        Args:
            texts: The list of text strings to train on.
        """
        # Tokenize the texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        
        # Build the vocabulary
        self.vocab = sorted(set(all_tokens))
        
        # Fit the encoder
        self.encoder.fit(self.vocab)
        
        # Store the vocabulary vectors
        self.vocab_vectors = [self.encoder.encode_token(token) for token in self.vocab]
        
        # Process each text
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            
            # Skip texts that are too short
            if len(tokens) <= self.context_size:
                continue
            
            # Process each position
            for i in range(len(tokens) - self.context_size):
                # Get the context
                context = tokens[i:i+self.context_size]
                
                # Get the target
                target = tokens[i+self.context_size]
                
                # Create a sequence for the context
                context_seq = sequence.Sequence(self.dimensions, self.vector_type)
                
                # Add each token to the sequence with position encoding
                for j, token in enumerate(context):
                    token_vector = self.encoder.encode_token(token)
                    context_seq.append(token_vector)
                
                # Get the context vector
                context_vector = context_seq.to_vector()
                
                # Encode the target
                target_vector = self.encoder.encode_token(target)
                
                # Add to the hash table
                self.context_table.add(context_vector, target_vector)
    
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
        if len(context) != self.context_size:
            raise ValueError(f"Context must have {self.context_size} tokens")
        
        # Create a sequence for the context
        context_seq = sequence.Sequence(self.dimensions, self.vector_type)
        
        # Add each token to the sequence with position encoding
        for i, token in enumerate(context):
            token_vector = self.encoder.encode_token(token)
            context_seq.append(token_vector)
        
        # Get the context vector
        context_vector = context_seq.to_vector()
        
        # Query the hash table
        prediction_vector = self.context_table.get(context_vector)
        
        # Calculate similarity with all vocabulary vectors
        similarities = [operations.cosine_similarity(prediction_vector, vec) for vec in self.vocab_vectors]
        
        # Get the top k predictions
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_predictions = [(self.vocab[i], similarities[i]) for i in top_indices]
        
        return top_predictions
    
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
        if len(seed) != self.context_size:
            raise ValueError(f"Seed must have {self.context_size} tokens")
        
        # Initialize the generated sequence with the seed
        generated = seed.copy()
        
        # Generate tokens
        for _ in range(length):
            # Get the current context
            context = generated[-self.context_size:]
            
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
