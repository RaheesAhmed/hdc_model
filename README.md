# Hyperdimensional Computing (HDC) Model

A Python implementation of Hyperdimensional Computing based on Kanerva's work. This package provides tools for creating and manipulating high-dimensional vectors for cognitive computing and language modeling.

## Overview

Hyperdimensional Computing (HDC) is a computational framework inspired by the high-dimensional nature of the brain's neural activity. It uses high-dimensional random vectors (hypervectors) to represent and manipulate information. This approach offers several advantages:

- Robustness to noise and errors
- Efficient memory usage
- Ability to represent complex structures
- Parallelizable operations

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from hdc_model.core import operations

# Create random hypervectors
d = 10000  # dimensionality
vector1 = operations.random_binary(d)
vector2 = operations.random_binary(d)

# Perform binding operation
bound = operations.bind(vector1, vector2)

# Perform bundling operation
bundled = operations.bundle([vector1, vector2])

# Calculate similarity
similarity = operations.cosine_similarity(vector1, vector2)
```

## Features

- Core HDC operations (binding, bundling, permutation)
- Data structures for HDC (hash tables, sequences)
- Text processing utilities
- HDC-based models for language processing

## References

- Kanerva, P. (2009). Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors. Cognitive Computation, 1(2), 139-159.
- Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.

## License

MIT
