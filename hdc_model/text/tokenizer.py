"""
Tokenization utilities for HDC text processing.

This module provides utilities for tokenizing text data for use with
Hyperdimensional Computing models.
"""

import re
import string
from typing import List, Dict, Set, Optional, Union


class SimpleTokenizer:
    """
    A simple tokenizer for text data.
    
    This class implements a basic tokenizer that splits text into words,
    removes punctuation, and converts to lowercase.
    """
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        """
        Initialize the tokenizer.
        
        Args:
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to remove punctuation.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.vocab = set()
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            A list of tokens.
        """
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into tokens
        tokens = text.split()
        
        return tokens
    
    def fit(self, texts: List[str]) -> None:
        """
        Build the vocabulary from a list of texts.
        
        Args:
            texts: The list of text strings to process.
        """
        # Reset the vocabulary
        self.vocab = set()
        
        # Process each text
        for text in texts:
            tokens = self.tokenize(text)
            self.vocab.update(tokens)
        
        # Create the word-to-index and index-to-word mappings
        self.word_to_index = {word: i for i, word in enumerate(sorted(self.vocab))}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a text string as a list of token indices.
        
        Args:
            text: The text to encode.
            
        Returns:
            A list of token indices.
        """
        tokens = self.tokenize(text)
        return [self.word_to_index.get(token, -1) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode a list of token indices back to a text string.
        
        Args:
            indices: The list of token indices.
            
        Returns:
            The decoded text string.
        """
        tokens = [self.index_to_word.get(idx, '') for idx in indices if idx in self.index_to_word]
        return ' '.join(tokens)


class NGramTokenizer:
    """
    A character n-gram tokenizer for text data.
    
    This class implements a tokenizer that extracts character n-grams from text.
    """
    
    def __init__(self, n: int = 3, lowercase: bool = True):
        """
        Initialize the tokenizer.
        
        Args:
            n: The size of the n-grams.
            lowercase: Whether to convert text to lowercase.
        """
        self.n = n
        self.lowercase = lowercase
        self.vocab = set()
        self.ngram_to_index = {}
        self.index_to_ngram = {}
        self.vocab_size = 0
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string into character n-grams.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            A list of character n-grams.
        """
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Extract n-grams
        ngrams = []
        for i in range(len(text) - self.n + 1):
            ngrams.append(text[i:i+self.n])
        
        return ngrams
    
    def fit(self, texts: List[str]) -> None:
        """
        Build the vocabulary from a list of texts.
        
        Args:
            texts: The list of text strings to process.
        """
        # Reset the vocabulary
        self.vocab = set()
        
        # Process each text
        for text in texts:
            ngrams = self.tokenize(text)
            self.vocab.update(ngrams)
        
        # Create the ngram-to-index and index-to-ngram mappings
        self.ngram_to_index = {ngram: i for i, ngram in enumerate(sorted(self.vocab))}
        self.index_to_ngram = {i: ngram for ngram, i in self.ngram_to_index.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a text string as a list of n-gram indices.
        
        Args:
            text: The text to encode.
            
        Returns:
            A list of n-gram indices.
        """
        ngrams = self.tokenize(text)
        return [self.ngram_to_index.get(ngram, -1) for ngram in ngrams]
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode a list of n-gram indices back to a text string.
        
        Note: This is an approximation since n-grams may overlap.
        
        Args:
            indices: The list of n-gram indices.
            
        Returns:
            The decoded text string.
        """
        ngrams = [self.index_to_ngram.get(idx, '') for idx in indices if idx in self.index_to_ngram]
        if not ngrams:
            return ''
        
        # Simple reconstruction by concatenating n-grams
        text = ngrams[0]
        for ngram in ngrams[1:]:
            text += ngram[-1]
        
        return text
