"""
Tests for the HDC data structures.
"""

import numpy as np
import pytest
from hdc_model.core import operations
from hdc_model.structures import hash_table, sequence


def test_hash_table_binary():
    """Test hash table with binary vectors."""
    d = 1000
    table = hash_table.HashTable(d, 'binary')
    
    # Create some test vectors
    key1 = operations.random_binary(d)
    value1 = operations.random_binary(d)
    key2 = operations.random_binary(d)
    value2 = operations.random_binary(d)
    
    # Test adding and retrieving
    table.add(key1, value1)
    retrieved1 = table.get(key1)
    
    # The retrieved value should be the same as the original
    assert np.array_equal(retrieved1, value1)
    
    # Add another key-value pair
    table.add(key2, value2)
    retrieved2 = table.get(key2)
    
    # The retrieved values should be the same as the originals
    assert np.array_equal(retrieved2, value2)
    
    # Test size
    assert table.size() == 2
    
    # Test contains
    assert table.contains(key1)
    assert table.contains(key2)
    assert not table.contains(operations.random_binary(d))
    
    # Test remove
    assert table.remove(key1)
    assert table.size() == 1
    assert not table.contains(key1)
    assert table.contains(key2)
    
    # Test clear
    table.clear()
    assert table.size() == 0
    assert not table.contains(key2)


def test_hash_table_bipolar():
    """Test hash table with bipolar vectors."""
    d = 1000
    table = hash_table.HashTable(d, 'bipolar')
    
    # Create some test vectors
    key1 = operations.random_bipolar(d)
    value1 = operations.random_bipolar(d)
    key2 = operations.random_bipolar(d)
    value2 = operations.random_bipolar(d)
    
    # Test adding and retrieving
    table.add(key1, value1)
    retrieved1 = table.get(key1)
    
    # The retrieved value should be the same as the original
    assert np.array_equal(retrieved1, value1)
    
    # Add another key-value pair
    table.add(key2, value2)
    retrieved2 = table.get(key2)
    
    # The retrieved values should be the same as the originals
    assert np.array_equal(retrieved2, value2)


def test_sequence_binary():
    """Test sequence with binary vectors."""
    d = 1000
    seq = sequence.Sequence(d, 'binary')
    
    # Create some test vectors
    elem1 = operations.random_binary(d)
    elem2 = operations.random_binary(d)
    elem3 = operations.random_binary(d)
    
    # Test append and get
    seq.append(elem1)
    assert np.array_equal(seq.get(0), elem1)
    
    seq.append(elem2)
    assert np.array_equal(seq.get(1), elem2)
    
    # Test extend
    seq.extend([elem3])
    assert np.array_equal(seq.get(2), elem3)
    
    # Test size
    assert seq.size() == 3
    
    # Test contains
    assert seq.contains(elem1)
    assert seq.contains(elem2)
    assert seq.contains(elem3)
    assert not seq.contains(operations.random_binary(d))
    
    # Test index_of
    assert seq.index_of(elem1) == 0
    assert seq.index_of(elem2) == 1
    assert seq.index_of(elem3) == 2
    assert seq.index_of(operations.random_binary(d)) == -1
    
    # Test to_vector
    vector = seq.to_vector()
    assert vector.shape == (d,)
    
    # Test ngram
    ngrams = seq.ngram(2)
    assert len(ngrams) == 2
    
    # Test clear
    seq.clear()
    assert seq.size() == 0


def test_sequence_bipolar():
    """Test sequence with bipolar vectors."""
    d = 1000
    seq = sequence.Sequence(d, 'bipolar')
    
    # Create some test vectors
    elem1 = operations.random_bipolar(d)
    elem2 = operations.random_bipolar(d)
    elem3 = operations.random_bipolar(d)
    
    # Test append and get
    seq.append(elem1)
    assert np.array_equal(seq.get(0), elem1)
    
    seq.append(elem2)
    assert np.array_equal(seq.get(1), elem2)
    
    # Test extend
    seq.extend([elem3])
    assert np.array_equal(seq.get(2), elem3)
    
    # Test size
    assert seq.size() == 3
