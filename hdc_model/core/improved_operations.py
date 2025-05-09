"""
Improved core operations for Hyperdimensional Computing.

This module implements highly optimized versions of HDC operations:
- Efficient bit-packing using NumPy's packbits/unpackbits
- SIMD-accelerated operations with Numba
- Sparse vector support
- Vectorized operations for maximum performance
"""

import numpy as np
from typing import List, Union, Tuple, Optional
import numba
from numba import njit, prange, boolean, int32, uint8, uint32, float32
import math


class SparseHDVector:
    """
    A memory-efficient implementation of sparse binary hypervectors.
    
    This class stores only the indices of non-zero elements, dramatically
    reducing memory usage for sparse vectors (e.g., 10% non-zero elements).
    """
    
    def __init__(self, dimensions: int, sparsity: float = 0.9):
        """
        Initialize a sparse hypervector.
        
        Args:
            dimensions: The dimensionality of the hypervector.
            sparsity: The proportion of 0s in the vector (default: 0.9).
        """
        self.dimensions = dimensions
        self.sparsity = sparsity
        self.indices = np.array([], dtype=np.int32)
    
    @classmethod
    def random(cls, dimensions: int, sparsity: float = 0.9) -> 'SparseHDVector':
        """
        Generate a random sparse hypervector.
        
        Args:
            dimensions: The dimensionality of the hypervector.
            sparsity: The proportion of 0s in the vector (default: 0.9).
            
        Returns:
            A random SparseHDVector.
        """
        result = cls(dimensions, sparsity)
        
        # Calculate number of non-zero elements
        nnz = int(dimensions * (1 - sparsity))
        
        # Generate random indices
        result.indices = np.sort(np.random.choice(dimensions, nnz, replace=False))
        
        return result
    
    @classmethod
    def from_binary_array(cls, binary_array: np.ndarray) -> 'SparseHDVector':
        """
        Create a sparse hypervector from a binary numpy array.
        
        Args:
            binary_array: A binary numpy array with values 0 and 1.
            
        Returns:
            A SparseHDVector representing the same data.
        """
        dimensions = len(binary_array)
        result = cls(dimensions)
        
        # Get indices of non-zero elements
        result.indices = np.where(binary_array)[0].astype(np.int32)
        
        return result
    
    def to_binary_array(self) -> np.ndarray:
        """
        Convert the sparse hypervector to a binary numpy array.
        
        Returns:
            A binary numpy array with values 0 and 1.
        """
        result = np.zeros(self.dimensions, dtype=np.int8)
        result[self.indices] = 1
        return result
    
    def __len__(self) -> int:
        """
        Get the dimensionality of the vector.
        
        Returns:
            The number of dimensions.
        """
        return self.dimensions
    
    def memory_usage(self) -> int:
        """
        Calculate the memory usage in bytes.
        
        Returns:
            The memory usage in bytes.
        """
        return self.indices.nbytes


@njit(parallel=True)
def _fast_hamming_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Hamming distance between two binary arrays using SIMD.
    
    Args:
        x: First binary array.
        y: Second binary array.
        
    Returns:
        The Hamming distance (normalized to [0, 1]).
    """
    count = 0
    for i in prange(len(x)):
        if x[i] != y[i]:
            count += 1
    return count / len(x)


@njit
def _sparse_intersection_size(a: np.ndarray, b: np.ndarray) -> int:
    """
    Calculate the size of the intersection of two sorted arrays.
    
    Args:
        a: First sorted array of indices.
        b: Second sorted array of indices.
        
    Returns:
        The size of the intersection.
    """
    i, j = 0, 0
    count = 0
    
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    
    return count


@njit
def _sparse_union_size(a: np.ndarray, b: np.ndarray) -> int:
    """
    Calculate the size of the union of two sorted arrays.
    
    Args:
        a: First sorted array of indices.
        b: Second sorted array of indices.
        
    Returns:
        The size of the union.
    """
    i, j = 0, 0
    count = 0
    
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            count += 1
            i += 1
        elif a[i] > b[j]:
            count += 1
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    
    # Add remaining elements
    count += len(a) - i
    count += len(b) - j
    
    return count


def sparse_hamming_distance(x: SparseHDVector, y: SparseHDVector) -> float:
    """
    Calculate the Hamming distance between two sparse hypervectors.
    
    Args:
        x: First sparse hypervector.
        y: Second sparse hypervector.
        
    Returns:
        The Hamming distance (normalized to [0, 1]).
    """
    if x.dimensions != y.dimensions:
        raise ValueError("Vectors must have the same dimensions")
    
    # Calculate the size of the symmetric difference (XOR)
    intersection_size = _sparse_intersection_size(x.indices, y.indices)
    union_size = _sparse_union_size(x.indices, y.indices)
    xor_size = union_size - intersection_size
    
    # Normalize by the number of dimensions
    return xor_size / x.dimensions


def sparse_bind(x: SparseHDVector, y: SparseHDVector) -> SparseHDVector:
    """
    Bind two sparse hypervectors using XOR operation.
    
    Args:
        x: First sparse hypervector.
        y: Second sparse hypervector.
        
    Returns:
        The bound sparse hypervector.
    """
    if x.dimensions != y.dimensions:
        raise ValueError("Vectors must have the same dimensions")
    
    # Convert to binary arrays, bind, and convert back to sparse
    x_array = x.to_binary_array()
    y_array = y.to_binary_array()
    
    # XOR operation
    result_array = np.logical_xor(x_array, y_array).astype(np.int8)
    
    return SparseHDVector.from_binary_array(result_array)


@njit
def _majority_vote(vectors: np.ndarray, threshold: float) -> np.ndarray:
    """
    Perform a majority vote on binary vectors using SIMD.
    
    Args:
        vectors: 2D array of binary vectors.
        threshold: The threshold for the majority rule.
        
    Returns:
        The result of the majority vote.
    """
    n_vectors, n_dims = vectors.shape
    result = np.zeros(n_dims, dtype=np.int8)
    
    for j in prange(n_dims):
        count = 0
        for i in range(n_vectors):
            count += vectors[i, j]
        
        if count >= threshold * n_vectors:
            result[j] = 1
    
    return result


def sparse_bundle(vectors: List[SparseHDVector], threshold: float = 0.5) -> SparseHDVector:
    """
    Bundle sparse hypervectors using majority rule.
    
    Args:
        vectors: List of sparse hypervectors to bundle.
        threshold: The threshold for the majority rule (default: 0.5).
        
    Returns:
        The bundled sparse hypervector.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list of vectors")
    
    # Check that all vectors have the same dimensions
    dimensions = vectors[0].dimensions
    for v in vectors:
        if v.dimensions != dimensions:
            raise ValueError("All vectors must have the same dimensions")
    
    # Convert to binary arrays
    binary_arrays = np.array([v.to_binary_array() for v in vectors])
    
    # Perform majority vote
    result_array = _majority_vote(binary_arrays, threshold)
    
    return SparseHDVector.from_binary_array(result_array)


def sparse_permute(vector: SparseHDVector, shift: int = 1) -> SparseHDVector:
    """
    Permute a sparse hypervector by cyclic shift.
    
    Args:
        vector: The sparse hypervector to permute.
        shift: The number of positions to shift (default: 1).
        
    Returns:
        The permuted sparse hypervector.
    """
    # Create a new sparse vector
    result = SparseHDVector(vector.dimensions)
    
    # Permute the indices
    result.indices = (vector.indices + shift) % vector.dimensions
    
    # Sort the indices
    result.indices.sort()
    
    return result


def sparse_inverse_permute(vector: SparseHDVector, shift: int = 1) -> SparseHDVector:
    """
    Apply the inverse permutation to a sparse hypervector.
    
    Args:
        vector: The permuted sparse hypervector.
        shift: The number of positions that were shifted (default: 1).
        
    Returns:
        The original sparse hypervector.
    """
    return sparse_permute(vector, -shift)


# Efficient bit-packed operations using NumPy's packbits/unpackbits

def binary_to_packed(binary_array: np.ndarray) -> np.ndarray:
    """
    Convert a binary array to a packed bit array using NumPy's packbits.
    
    Args:
        binary_array: A binary numpy array with values 0 and 1.
        
    Returns:
        A packed bit array.
    """
    # Ensure the array is binary
    binary_array = binary_array.astype(bool)
    
    # Pack the bits
    return np.packbits(binary_array)


def packed_to_binary(packed_array: np.ndarray, length: int) -> np.ndarray:
    """
    Convert a packed bit array to a binary array using NumPy's unpackbits.
    
    Args:
        packed_array: A packed bit array.
        length: The length of the original binary array.
        
    Returns:
        A binary numpy array with values 0 and 1.
    """
    # Unpack the bits
    binary_array = np.unpackbits(packed_array)
    
    # Truncate to the original length
    return binary_array[:length].astype(np.int8)


@njit(parallel=True)
def _batch_hamming_distance(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Calculate Hamming distance between a query vector and multiple vectors.
    
    Args:
        vectors: 2D array of binary vectors.
        query: Query binary vector.
        
    Returns:
        Array of Hamming distances.
    """
    n_vectors = vectors.shape[0]
    n_dims = vectors.shape[1]
    distances = np.zeros(n_vectors, dtype=np.float32)
    
    for i in prange(n_vectors):
        count = 0
        for j in range(n_dims):
            if vectors[i, j] != query[j]:
                count += 1
        distances[i] = count / n_dims
    
    return distances


def batch_hamming_distance(vectors: List[SparseHDVector], query: SparseHDVector) -> np.ndarray:
    """
    Calculate Hamming distance between a query vector and multiple vectors.
    
    Args:
        vectors: List of sparse hypervectors.
        query: Query sparse hypervector.
        
    Returns:
        Array of Hamming distances.
    """
    # Convert to binary arrays
    binary_arrays = np.array([v.to_binary_array() for v in vectors])
    query_array = query.to_binary_array()
    
    # Calculate distances
    return _batch_hamming_distance(binary_arrays, query_array)


class HDCVectorCache:
    """
    Cache for frequently used hypervectors.
    
    This class provides a cache for frequently used hypervectors to avoid
    recomputing them during training and generation.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the cache.
        
        Args:
            max_size: The maximum number of vectors to cache.
        """
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
    
    def get(self, key: str) -> Optional[SparseHDVector]:
        """
        Get a vector from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached vector, or None if not found.
        """
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key: str, vector: SparseHDVector) -> None:
        """
        Add a vector to the cache.
        
        Args:
            key: The cache key.
            vector: The vector to cache.
        """
        # If the cache is full, remove the least accessed item
        if len(self.cache) >= self.max_size:
            min_key = min(self.access_count, key=self.access_count.get)
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = vector
        self.access_count[key] = 1
    
    def clear(self) -> None:
        """
        Clear the cache.
        """
        self.cache.clear()
        self.access_count.clear()
