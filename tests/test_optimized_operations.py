"""
Tests for the optimized HDC operations.
"""

import numpy as np
import pytest
from hdc_model.core import operations
from hdc_model.core import optimized_operations as opt_ops


def test_bit_packed_vector_creation():
    """Test bit-packed vector creation."""
    d = 1000
    # Create a random binary array
    binary_array = operations.random_binary(d)

    # Create a bit-packed vector from the binary array
    bit_packed = opt_ops.BitPackedVector.from_binary_array(binary_array)

    # Convert back to binary array
    unpacked = bit_packed.to_binary_array()

    # Check that the unpacked array matches the original
    assert np.array_equal(binary_array, unpacked)

    # Check the dimensions
    assert len(bit_packed) == d

    # Check memory usage (should be less than the original)
    binary_size = binary_array.nbytes
    packed_size = bit_packed.packed_data.nbytes
    # For small vectors, the overhead might make the packed version not much smaller
    # Just verify it's smaller than the original
    assert packed_size < binary_size


def test_bit_packed_vector_random():
    """Test random bit-packed vector generation."""
    d = 10000

    # Generate a random bit-packed vector
    bit_packed = opt_ops.BitPackedVector.random(d, sparsity=0.7)

    # Convert to binary array
    binary_array = bit_packed.to_binary_array()

    # Check dimensions
    assert len(binary_array) == d

    # Check sparsity (approximately)
    ones_ratio = np.mean(binary_array)
    assert 0.25 < ones_ratio < 0.35  # Allow for some random variation


def test_bit_packed_vector_getitem():
    """Test getting items from a bit-packed vector."""
    d = 1000
    binary_array = operations.random_binary(d)
    bit_packed = opt_ops.BitPackedVector.from_binary_array(binary_array)

    # Check random indices
    for _ in range(10):
        idx = np.random.randint(0, d)
        assert bit_packed[idx] == binary_array[idx]

    # Check out of bounds
    with pytest.raises(IndexError):
        _ = bit_packed[d]
    with pytest.raises(IndexError):
        _ = bit_packed[-d-1]


def test_bit_packed_vector_setitem():
    """Test setting items in a bit-packed vector."""
    d = 1000
    bit_packed = opt_ops.BitPackedVector(d)

    # Set random indices
    for _ in range(100):
        idx = np.random.randint(0, d)
        value = np.random.randint(0, 2)
        bit_packed[idx] = value
        assert bit_packed[idx] == value

    # Check out of bounds
    with pytest.raises(IndexError):
        bit_packed[d] = 1
    with pytest.raises(IndexError):
        bit_packed[-d-1] = 0


def test_bind_bit_packed():
    """Test binding operation for bit-packed vectors."""
    d = 1000

    # Create two random binary arrays
    x_array = operations.random_binary(d)
    y_array = operations.random_binary(d)

    # Bind using the original operations
    bound_array = operations.bind_binary(x_array, y_array)

    # Convert to bit-packed vectors
    x_packed = opt_ops.BitPackedVector.from_binary_array(x_array)
    y_packed = opt_ops.BitPackedVector.from_binary_array(y_array)

    # Bind using the optimized operations
    bound_packed = opt_ops.bind_bit_packed(x_packed, y_packed)

    # Convert back to binary array
    bound_unpacked = bound_packed.to_binary_array()

    # Check that the results match
    assert np.array_equal(bound_array, bound_unpacked)

    # Test that binding is its own inverse
    unbound_packed = opt_ops.bind_bit_packed(bound_packed, y_packed)
    unbound_array = unbound_packed.to_binary_array()
    assert np.array_equal(unbound_array, x_array)


def test_hamming_distance_bit_packed():
    """Test Hamming distance calculation for bit-packed vectors."""
    d = 10000

    # Create two random binary arrays
    x_array = operations.random_binary(d)
    y_array = operations.random_binary(d)

    # Calculate Hamming distance using the original operations
    dist_original = operations.hamming_distance(x_array, y_array)

    # Convert to bit-packed vectors
    x_packed = opt_ops.BitPackedVector.from_binary_array(x_array)
    y_packed = opt_ops.BitPackedVector.from_binary_array(y_array)

    # Calculate Hamming distance using the optimized operations
    dist_optimized = opt_ops.hamming_distance_bit_packed(x_packed, y_packed)

    # Check that the results match (within floating-point precision)
    assert abs(dist_original - dist_optimized) < 1e-10


def test_bundle_bit_packed():
    """Test bundling operation for bit-packed vectors."""
    d = 1000
    n_vectors = 5

    # Create random binary arrays
    arrays = [operations.random_binary(d) for _ in range(n_vectors)]

    # Bundle using the original operations
    bundled_array = operations.bundle_binary(arrays)

    # Convert to bit-packed vectors
    packed_vectors = [opt_ops.BitPackedVector.from_binary_array(arr) for arr in arrays]

    # Bundle using the optimized operations
    bundled_packed = opt_ops.bundle_bit_packed(packed_vectors)

    # Convert back to binary array
    bundled_unpacked = bundled_packed.to_binary_array()

    # Check that the results match
    assert np.array_equal(bundled_array, bundled_unpacked)


def test_permute_bit_packed():
    """Test permutation operation for bit-packed vectors."""
    d = 1000
    shift = 5

    # Create a random binary array
    binary_array = operations.random_binary(d)

    # Permute using the original operations
    permuted_array = operations.permute(binary_array, shift)

    # Convert to bit-packed vector
    bit_packed = opt_ops.BitPackedVector.from_binary_array(binary_array)

    # Permute using the optimized operations
    permuted_packed = opt_ops.permute_bit_packed(bit_packed, shift)

    # Convert back to binary array
    permuted_unpacked = permuted_packed.to_binary_array()

    # Check that the results match
    assert np.array_equal(permuted_array, permuted_unpacked)

    # Test inverse permutation
    unpermuted_packed = opt_ops.inverse_permute_bit_packed(permuted_packed, shift)
    unpermuted_array = unpermuted_packed.to_binary_array()
    assert np.array_equal(unpermuted_array, binary_array)
