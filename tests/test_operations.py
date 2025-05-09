"""
Tests for the core HDC operations.
"""

import numpy as np
import pytest
from hdc_model.core import operations


def test_random_binary():
    """Test random binary vector generation."""
    d = 10000
    v = operations.random_binary(d)
    
    assert v.shape == (d,)
    assert set(np.unique(v)).issubset({0, 1})
    
    # Test sparsity
    v_sparse = operations.random_binary(d, sparsity=0.8)
    assert np.mean(v_sparse) < 0.3  # Should be around 0.2


def test_random_bipolar():
    """Test random bipolar vector generation."""
    d = 10000
    v = operations.random_bipolar(d)
    
    assert v.shape == (d,)
    assert set(np.unique(v)).issubset({-1, 1})


def test_bind_binary():
    """Test binary binding operation."""
    d = 100
    x = operations.random_binary(d)
    y = operations.random_binary(d)
    
    bound = operations.bind_binary(x, y)
    
    assert bound.shape == (d,)
    assert set(np.unique(bound)).issubset({0, 1})
    
    # Test that binding is its own inverse
    unbound = operations.bind_binary(bound, y)
    assert np.array_equal(unbound, x)


def test_bind_bipolar():
    """Test bipolar binding operation."""
    d = 100
    x = operations.random_bipolar(d)
    y = operations.random_bipolar(d)
    
    bound = operations.bind_bipolar(x, y)
    
    assert bound.shape == (d,)
    assert set(np.unique(bound)).issubset({-1, 1})
    
    # Test that binding is its own inverse
    unbound = operations.bind_bipolar(bound, y)
    assert np.array_equal(unbound, x)


def test_bundle_binary():
    """Test binary bundling operation."""
    d = 100
    vectors = [operations.random_binary(d) for _ in range(5)]
    
    bundled = operations.bundle_binary(vectors)
    
    assert bundled.shape == (d,)
    assert set(np.unique(bundled)).issubset({0, 1})


def test_bundle_bipolar():
    """Test bipolar bundling operation."""
    d = 100
    vectors = [operations.random_bipolar(d) for _ in range(5)]
    
    bundled = operations.bundle_bipolar(vectors)
    
    assert bundled.shape == (d,)
    assert set(np.unique(bundled)).issubset({-1, 0, 1})  # 0 can occur if sum is 0


def test_permute():
    """Test permutation operation."""
    d = 100
    v = operations.random_binary(d)
    
    permuted = operations.permute(v)
    
    assert permuted.shape == (d,)
    assert not np.array_equal(permuted, v)  # Should be different
    
    # Test that inverse permutation works
    unpermuted = operations.inverse_permute(permuted)
    assert np.array_equal(unpermuted, v)


def test_hamming_distance():
    """Test Hamming distance calculation."""
    d = 100
    x = operations.random_binary(d)
    y = x.copy()
    
    # Same vectors should have distance 0
    assert operations.hamming_distance(x, x) == 0
    
    # Flip some bits
    num_flips = 20
    indices = np.random.choice(d, num_flips, replace=False)
    y[indices] = 1 - y[indices]
    
    # Distance should be approximately num_flips/d
    assert abs(operations.hamming_distance(x, y) - num_flips/d) < 0.01


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    d = 100
    x = operations.random_bipolar(d)
    
    # Same vectors should have similarity 1
    assert operations.cosine_similarity(x, x) == 1.0
    
    # Orthogonal vectors should have similarity around 0
    y = operations.random_bipolar(d)
    sim = operations.cosine_similarity(x, y)
    assert abs(sim) < 0.2  # Should be close to 0 for large d
    
    # Opposite vectors should have similarity -1
    assert operations.cosine_similarity(x, -x) == -1.0
