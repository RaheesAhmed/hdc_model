"""
Simple test script for HDC operations.
"""

import numpy as np
from hdc_model.core import operations

def main():
    print("Testing HDC operations...")
    
    # Test random vector generation
    d = 10000
    print(f"Generating random binary vector of dimension {d}...")
    v_binary = operations.random_binary(d)
    print(f"Binary vector shape: {v_binary.shape}")
    print(f"Binary vector unique values: {np.unique(v_binary)}")
    
    print("\nGenerating random bipolar vector...")
    v_bipolar = operations.random_bipolar(d)
    print(f"Bipolar vector shape: {v_bipolar.shape}")
    print(f"Bipolar vector unique values: {np.unique(v_bipolar)}")
    
    # Test binding
    print("\nTesting binding operation...")
    x_binary = operations.random_binary(d)
    y_binary = operations.random_binary(d)
    bound_binary = operations.bind(x_binary, y_binary)
    print(f"Binary binding result shape: {bound_binary.shape}")
    print(f"Binary binding result unique values: {np.unique(bound_binary)}")
    
    # Test bundling
    print("\nTesting bundling operation...")
    vectors_binary = [operations.random_binary(d) for _ in range(5)]
    bundled_binary = operations.bundle(vectors_binary)
    print(f"Binary bundling result shape: {bundled_binary.shape}")
    print(f"Binary bundling result unique values: {np.unique(bundled_binary)}")
    
    print("\nTests completed successfully!")

if __name__ == "__main__":
    main()
