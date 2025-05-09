"""
Core operations for Hyperdimensional Computing.

This module implements the fundamental operations for HDC:
- Random vector generation
- Binding
- Bundling
- Similarity measures
- Permutation
"""

import numpy as np
from typing import List, Union, Tuple


def random_binary(dimensions: int, sparsity: float = 0.5) -> np.ndarray:
    """
    Generate a random binary hypervector.

    Args:
        dimensions: The dimensionality of the hypervector.
        sparsity: The proportion of 0s in the vector (default: 0.5).

    Returns:
        A binary numpy array of shape (dimensions,).
    """
    return np.random.binomial(1, 1 - sparsity, dimensions).astype(np.int8)


def random_bipolar(dimensions: int) -> np.ndarray:
    """
    Generate a random bipolar hypervector with values {-1, 1}.

    Args:
        dimensions: The dimensionality of the hypervector.

    Returns:
        A bipolar numpy array of shape (dimensions,).
    """
    return np.random.choice([-1, 1], dimensions).astype(np.int8)


def random_continuous(dimensions: int) -> np.ndarray:
    """
    Generate a random continuous hypervector with values from normal distribution.

    Args:
        dimensions: The dimensionality of the hypervector.

    Returns:
        A continuous numpy array of shape (dimensions,).
    """
    return np.random.normal(0, 1/np.sqrt(dimensions), dimensions).astype(np.float32)


def bind_binary(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bind two binary hypervectors using XOR operation.

    Args:
        x: First binary hypervector.
        y: Second binary hypervector.

    Returns:
        The bound hypervector.
    """
    return np.logical_xor(x, y).astype(np.int8)


def bind_bipolar(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bind two bipolar hypervectors using element-wise multiplication.

    Args:
        x: First bipolar hypervector.
        y: Second bipolar hypervector.

    Returns:
        The bound hypervector.
    """
    return x * y


def bind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bind two hypervectors. The operation depends on the type of vectors.

    Args:
        x: First hypervector.
        y: Second hypervector.

    Returns:
        The bound hypervector.
    """
    # Check if vectors are binary (0, 1)
    if set(np.unique(x)).issubset({0, 1}) and set(np.unique(y)).issubset({0, 1}):
        return bind_binary(x, y)
    # Check if vectors are bipolar (-1, 1)
    elif set(np.unique(x)).issubset({-1, 1}) and set(np.unique(y)).issubset({-1, 1}):
        return bind_bipolar(x, y)
    # If vectors are all zeros (empty table), return the other vector
    elif np.all(x == 0):
        return y.copy()
    elif np.all(y == 0):
        return x.copy()
    else:
        raise ValueError("Unsupported vector types for binding")


def unbind_binary(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Unbind a binary hypervector using XOR operation (same as binding).

    Args:
        x: The bound hypervector.
        y: The binding key.

    Returns:
        The unbound hypervector.
    """
    return bind_binary(x, y)  # XOR is its own inverse


def unbind_bipolar(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Unbind a bipolar hypervector using element-wise multiplication (same as binding).

    Args:
        x: The bound hypervector.
        y: The binding key.

    Returns:
        The unbound hypervector.
    """
    return bind_bipolar(x, y)  # Element-wise multiplication with {-1, 1} is its own inverse


def unbind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Unbind a hypervector. The operation depends on the type of vectors.

    Args:
        x: The bound hypervector.
        y: The binding key.

    Returns:
        The unbound hypervector.
    """
    # Check if vectors are binary (0, 1)
    if set(np.unique(x)).issubset({0, 1}) and set(np.unique(y)).issubset({0, 1}):
        return bind_binary(x, y)  # XOR is its own inverse
    # Check if vectors are bipolar (-1, 1)
    elif set(np.unique(x)).issubset({-1, 1}) and set(np.unique(y)).issubset({-1, 1}):
        return bind_bipolar(x, y)  # Element-wise multiplication with {-1, 1} is its own inverse
    # If vectors are all zeros (empty table), return zeros
    elif np.all(x == 0):
        return np.zeros_like(x)
    # If binding key is all zeros, return the bound vector
    elif np.all(y == 0):
        return x.copy()
    else:
        raise ValueError("Unsupported vector types for unbinding")


def bundle_binary(vectors: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Bundle binary hypervectors using majority rule.

    Args:
        vectors: List of binary hypervectors to bundle.
        threshold: The threshold for the majority rule (default: 0.5).

    Returns:
        The bundled hypervector.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list of vectors")

    # Sum the vectors
    summed = np.sum(vectors, axis=0)

    # Apply threshold
    return (summed >= threshold * len(vectors)).astype(np.int8)


def bundle_bipolar(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Bundle bipolar hypervectors using element-wise sum and sign.

    Args:
        vectors: List of bipolar hypervectors to bundle.

    Returns:
        The bundled hypervector.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list of vectors")

    # Sum the vectors
    summed = np.sum(vectors, axis=0)

    # Apply sign function
    return np.sign(summed).astype(np.int8)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Bundle hypervectors. The operation depends on the type of vectors.

    Args:
        vectors: List of hypervectors to bundle.

    Returns:
        The bundled hypervector.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list of vectors")

    # Check vector type
    first_vector = vectors[0]
    if set(np.unique(first_vector)).issubset({0, 1}):
        return bundle_binary(vectors)
    elif set(np.unique(first_vector)).issubset({-1, 1}):
        return bundle_bipolar(vectors)
    else:
        # For continuous vectors, just take the average
        return np.mean(vectors, axis=0)


def permute(vector: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Permute a hypervector by cyclic shift.

    Args:
        vector: The hypervector to permute.
        shift: The number of positions to shift (default: 1).

    Returns:
        The permuted hypervector.
    """
    return np.roll(vector, shift)


def inverse_permute(vector: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Apply the inverse permutation (cyclic shift in the opposite direction).

    Args:
        vector: The permuted hypervector.
        shift: The number of positions that were shifted (default: 1).

    Returns:
        The original hypervector.
    """
    return np.roll(vector, -shift)


def hamming_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Hamming distance between two binary hypervectors.

    Args:
        x: First binary hypervector.
        y: Second binary hypervector.

    Returns:
        The Hamming distance (normalized to [0, 1]).
    """
    return np.mean(x != y)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two hypervectors.

    Args:
        x: First hypervector.
        y: Second hypervector.

    Returns:
        The cosine similarity in [-1, 1].
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def dot_product_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the normalized dot product similarity between two hypervectors.

    Args:
        x: First hypervector.
        y: Second hypervector.

    Returns:
        The normalized dot product in [-1, 1] for bipolar vectors, [0, 1] for binary.
    """
    if np.array_equal(np.unique(x), np.array([0, 1])) and np.array_equal(np.unique(y), np.array([0, 1])):
        # For binary vectors, normalize by dimension
        return np.dot(x, y) / len(x)
    else:
        # For other vectors, use cosine similarity
        return cosine_similarity(x, y)
