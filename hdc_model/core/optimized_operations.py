"""
Optimized core operations for Hyperdimensional Computing.

This module implements optimized versions of the fundamental HDC operations:
- Bit-packed binary vectors (32 dimensions in a single 32-bit integer)
- SIMD-accelerated binding and bundling
- Sparse vector operations
"""

import numpy as np
from typing import List, Union, Tuple, Optional
import numba
from numba import njit, prange
import math


class BitPackedVector:
    """
    A memory-efficient implementation of binary hypervectors using bit packing.
    
    This class stores binary vectors (with values 0 and 1) by packing 32 dimensions
    into a single 32-bit integer, reducing memory usage by up to 32x compared to
    using a full numpy array.
    """
    
    def __init__(self, dimensions: int):
        """
        Initialize a bit-packed vector.
        
        Args:
            dimensions: The dimensionality of the hypervector.
        """
        self.dimensions = dimensions
        # Calculate how many 32-bit integers we need to store the vector
        self.n_ints = (dimensions + 31) // 32
        # Initialize the packed data
        self.packed_data = np.zeros(self.n_ints, dtype=np.uint32)
    
    @classmethod
    def from_binary_array(cls, binary_array: np.ndarray) -> 'BitPackedVector':
        """
        Create a bit-packed vector from a binary numpy array.
        
        Args:
            binary_array: A binary numpy array with values 0 and 1.
            
        Returns:
            A BitPackedVector representing the same data.
        """
        dimensions = len(binary_array)
        result = cls(dimensions)
        
        # Pack the binary array into integers
        for i in range(dimensions):
            if binary_array[i]:
                int_idx = i // 32
                bit_idx = i % 32
                result.packed_data[int_idx] |= (1 << bit_idx)
        
        return result
    
    @classmethod
    def random(cls, dimensions: int, sparsity: float = 0.5) -> 'BitPackedVector':
        """
        Generate a random bit-packed binary vector.
        
        Args:
            dimensions: The dimensionality of the hypervector.
            sparsity: The proportion of 0s in the vector (default: 0.5).
            
        Returns:
            A random BitPackedVector.
        """
        # Generate a random binary array
        binary_array = np.random.binomial(1, 1 - sparsity, dimensions).astype(np.int8)
        # Convert to bit-packed format
        return cls.from_binary_array(binary_array)
    
    def to_binary_array(self) -> np.ndarray:
        """
        Convert the bit-packed vector to a binary numpy array.
        
        Returns:
            A binary numpy array with values 0 and 1.
        """
        result = np.zeros(self.dimensions, dtype=np.int8)
        
        # Unpack the integers into a binary array
        for int_idx in range(self.n_ints):
            for bit_idx in range(32):
                i = int_idx * 32 + bit_idx
                if i < self.dimensions:
                    if self.packed_data[int_idx] & (1 << bit_idx):
                        result[i] = 1
        
        return result
    
    def __getitem__(self, index: int) -> int:
        """
        Get the value at the specified index.
        
        Args:
            index: The index to retrieve.
            
        Returns:
            The value (0 or 1) at the specified index.
        """
        if index < 0 or index >= self.dimensions:
            raise IndexError("Index out of bounds")
        
        int_idx = index // 32
        bit_idx = index % 32
        
        return 1 if self.packed_data[int_idx] & (1 << bit_idx) else 0
    
    def __setitem__(self, index: int, value: int) -> None:
        """
        Set the value at the specified index.
        
        Args:
            index: The index to set.
            value: The value (0 or 1) to set.
        """
        if index < 0 or index >= self.dimensions:
            raise IndexError("Index out of bounds")
        
        int_idx = index // 32
        bit_idx = index % 32
        
        if value:
            self.packed_data[int_idx] |= (1 << bit_idx)
        else:
            self.packed_data[int_idx] &= ~(1 << bit_idx)
    
    def __len__(self) -> int:
        """
        Get the dimensionality of the vector.
        
        Returns:
            The number of dimensions.
        """
        return self.dimensions


@njit(parallel=True)
def _fast_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fast XOR operation using Numba.
    
    Args:
        a: First array of uint32 values.
        b: Second array of uint32 values.
        
    Returns:
        XOR of the two arrays.
    """
    result = np.zeros_like(a)
    for i in prange(len(a)):
        result[i] = a[i] ^ b[i]
    return result


def bind_bit_packed(x: BitPackedVector, y: BitPackedVector) -> BitPackedVector:
    """
    Bind two bit-packed binary vectors using XOR operation.
    
    Args:
        x: First bit-packed binary vector.
        y: Second bit-packed binary vector.
        
    Returns:
        The bound bit-packed vector.
    """
    if x.dimensions != y.dimensions:
        raise ValueError("Vectors must have the same dimensions")
    
    result = BitPackedVector(x.dimensions)
    result.packed_data = _fast_xor(x.packed_data, y.packed_data)
    
    return result


@njit
def _count_ones(x: np.uint32) -> int:
    """
    Count the number of 1 bits in a 32-bit integer.
    
    Args:
        x: A 32-bit integer.
        
    Returns:
        The number of 1 bits.
    """
    # Use bit manipulation to count ones (Brian Kernighan's algorithm)
    count = 0
    while x:
        x &= x - 1  # Clear the least significant bit set
        count += 1
    return count


@njit(parallel=True)
def _fast_popcount(packed_data: np.ndarray) -> np.ndarray:
    """
    Count the number of 1 bits in each 32-bit integer.
    
    Args:
        packed_data: Array of 32-bit integers.
        
    Returns:
        Array with the count of 1 bits for each integer.
    """
    result = np.zeros(len(packed_data), dtype=np.int32)
    for i in prange(len(packed_data)):
        result[i] = _count_ones(packed_data[i])
    return result


def hamming_distance_bit_packed(x: BitPackedVector, y: BitPackedVector) -> float:
    """
    Calculate the Hamming distance between two bit-packed binary vectors.
    
    Args:
        x: First bit-packed binary vector.
        y: Second bit-packed binary vector.
        
    Returns:
        The Hamming distance (normalized to [0, 1]).
    """
    if x.dimensions != y.dimensions:
        raise ValueError("Vectors must have the same dimensions")
    
    # XOR the packed data
    xor_result = _fast_xor(x.packed_data, y.packed_data)
    
    # Count the number of 1 bits in the XOR result
    ones_count = np.sum(_fast_popcount(xor_result))
    
    # Normalize by the number of dimensions
    return ones_count / x.dimensions


@njit(parallel=True)
def _fast_majority_vote(vectors_data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Perform a fast majority vote on packed binary vectors.
    
    Args:
        vectors_data: 2D array of packed binary vectors.
        threshold: The threshold for the majority rule.
        
    Returns:
        The result of the majority vote.
    """
    n_vectors = vectors_data.shape[0]
    n_ints = vectors_data.shape[1]
    result = np.zeros(n_ints, dtype=np.uint32)
    
    # For each bit position
    for int_idx in prange(n_ints):
        for bit_idx in range(32):
            # Count how many vectors have this bit set
            count = 0
            for vec_idx in range(n_vectors):
                if vectors_data[vec_idx, int_idx] & (1 << bit_idx):
                    count += 1
            
            # Set the bit in the result if the count exceeds the threshold
            if count >= threshold * n_vectors:
                result[int_idx] |= (1 << bit_idx)
    
    return result


def bundle_bit_packed(vectors: List[BitPackedVector], threshold: float = 0.5) -> BitPackedVector:
    """
    Bundle bit-packed binary vectors using majority rule.
    
    Args:
        vectors: List of bit-packed binary vectors to bundle.
        threshold: The threshold for the majority rule (default: 0.5).
        
    Returns:
        The bundled bit-packed vector.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list of vectors")
    
    # Check that all vectors have the same dimensions
    dimensions = vectors[0].dimensions
    for v in vectors:
        if v.dimensions != dimensions:
            raise ValueError("All vectors must have the same dimensions")
    
    # Create a 2D array of the packed data
    vectors_data = np.array([v.packed_data for v in vectors])
    
    # Create the result vector
    result = BitPackedVector(dimensions)
    
    # Perform the majority vote
    result.packed_data = _fast_majority_vote(vectors_data, threshold)
    
    return result


def permute_bit_packed(vector: BitPackedVector, shift: int = 1) -> BitPackedVector:
    """
    Permute a bit-packed binary vector by cyclic shift.
    
    Args:
        vector: The bit-packed binary vector to permute.
        shift: The number of positions to shift (default: 1).
        
    Returns:
        The permuted bit-packed vector.
    """
    # Convert to binary array, permute, and convert back
    # This is not the most efficient implementation, but it's simple
    binary_array = vector.to_binary_array()
    permuted_array = np.roll(binary_array, shift)
    return BitPackedVector.from_binary_array(permuted_array)


def inverse_permute_bit_packed(vector: BitPackedVector, shift: int = 1) -> BitPackedVector:
    """
    Apply the inverse permutation to a bit-packed binary vector.
    
    Args:
        vector: The permuted bit-packed binary vector.
        shift: The number of positions that were shifted (default: 1).
        
    Returns:
        The original bit-packed vector.
    """
    return permute_bit_packed(vector, -shift)
