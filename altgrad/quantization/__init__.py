"""FP8 quantization formats and transfer functions.

This package provides the mathematical foundation for FP8 quantization,
including format specifications and bit-level transfer functions.

Available Formats:
    E0M7: Fixed-point in [-1, 1), 7 mantissa bits, no exponent
    E1M6: Two-scale format, 1 exponent bit, 6 mantissa bits
    E3M4: Moderate range, 3 exponent bits, 4 mantissa bits
    E5M2: Standard FP8 with wide range, IEEE-like semantics
    E7M0: Powers of 2 only, 7 exponent bits, no mantissa

Example:
    >>> from altgrad.quantization import E5M2
    >>> E5M2.to_real(0b00111100)  # = 1.0
    1.0
    >>> E5M2.to_bits(1.5)  # = 0b00111110
    62
"""

from altgrad.quantization.formats import (
    FP8Format,
    E0M7,
    E1M6,
    E3M4,
    E5M2,
    E7M0,
    FORMAT_REGISTRY,
)

__all__ = [
    "FP8Format",
    "E0M7",
    "E1M6",
    "E3M4",
    "E5M2",
    "E7M0",
    "FORMAT_REGISTRY",
]
