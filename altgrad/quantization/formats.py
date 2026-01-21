"""FP8 format registry with bit-level transfer functions.

This module provides the mathematical foundation for all FP8 quantization.
Each format's bit-index <-> real value mapping is precisely defined.

FP8 Format Structure:
- 8 bits total: sign (1) + exponent (E) + mantissa (M), where E+M=7
- E0M7: 0 exponent, 7 mantissa - fixed-point in [-1, 1)
- E1M6: 1 exponent, 6 mantissa - two scales
- E3M4: 3 exponent, 4 mantissa - moderate range
- E5M2: 5 exponent, 2 mantissa - standard FP8, wide range
- E7M0: 7 exponent, 0 mantissa - powers of 2 only

Transfer Functions:
- to_real(bit_index): Convert 8-bit pattern to real value
- to_bits(value): Convert real value to nearest 8-bit pattern
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class FP8Format:
    """Specification for an 8-bit floating-point format.

    Attributes:
        name: Format identifier (e.g., "E5M2")
        exponent_bits: Number of exponent bits (0-7)
        mantissa_bits: Number of mantissa bits (7-exponent_bits)
        bias: Exponent bias for normalized numbers
        has_inf: Whether format supports infinity representation
        has_nan: Whether format supports NaN representation
    """

    name: str
    exponent_bits: int
    mantissa_bits: int
    bias: int
    has_inf: bool = False
    has_nan: bool = False

    @property
    def max_representable_value(self) -> float:
        """Maximum finite positive value representable in this format."""
        if self.exponent_bits == 0:
            # E0M7: pure fixed-point
            max_mantissa = (1 << self.mantissa_bits) - 1
            return max_mantissa / (1 << self.mantissa_bits)

        # Standard floating-point
        if self.has_inf:
            # Max exponent is reserved for inf/nan
            max_exp = (1 << self.exponent_bits) - 2
        else:
            max_exp = (1 << self.exponent_bits) - 1

        max_mantissa = (1 << self.mantissa_bits) - 1
        mantissa_value = 1.0 + max_mantissa / (1 << self.mantissa_bits)
        return (2 ** (max_exp - self.bias)) * mantissa_value

    def to_real(self, bit_index: int) -> float:
        """Convert 8-bit pattern to real value.

        Args:
            bit_index: Integer in range [0, 255] representing the bit pattern

        Returns:
            Real number represented by this bit pattern
        """
        if not 0 <= bit_index <= 255:
            raise ValueError(f"bit_index must be in [0, 255], got {bit_index}")

        # Extract components
        sign = (bit_index >> 7) & 1
        mantissa_mask = (1 << self.mantissa_bits) - 1
        mantissa = bit_index & mantissa_mask

        if self.exponent_bits == 0:
            # E0M7: pure fixed-point format
            # Value = mantissa / 2^7, sign bit determines sign
            value = mantissa / (1 << self.mantissa_bits)
            return -value if sign else value

        # Extract exponent
        exponent = (bit_index >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)
        max_exp = (1 << self.exponent_bits) - 1

        # Handle special cases (inf, nan) for formats that support them
        if self.has_inf and exponent == max_exp:
            if mantissa == 0:
                return float('-inf') if sign else float('inf')
            if self.has_nan:
                return float('nan')

        # Handle zero
        if exponent == 0 and mantissa == 0:
            return -0.0 if sign else 0.0

        # Handle denormalized numbers (exponent == 0, mantissa != 0)
        if exponent == 0:
            # Denorm: value = 2^(1-bias) * (mantissa / 2^M)
            value = (2 ** (1 - self.bias)) * (mantissa / (1 << self.mantissa_bits))
            return -value if sign else value

        # Normalized numbers: value = 2^(exp-bias) * (1 + mantissa / 2^M)
        mantissa_value = 1.0 + mantissa / (1 << self.mantissa_bits)
        value = (2 ** (exponent - self.bias)) * mantissa_value
        return -value if sign else value

    def to_bits(self, value: float) -> int:
        """Convert real value to nearest 8-bit pattern.

        Uses round-to-nearest-even for ties.

        Args:
            value: Real number to quantize

        Returns:
            Integer in range [0, 255] representing the nearest bit pattern
        """
        # Handle special values
        if math.isnan(value):
            if self.has_nan:
                # NaN: max exponent, non-zero mantissa
                max_exp = (1 << self.exponent_bits) - 1
                return (max_exp << self.mantissa_bits) | 1
            # Format doesn't support NaN, return 0
            return 0

        # Handle sign
        sign = 1 if value < 0 or (value == 0 and math.copysign(1, value) < 0) else 0
        abs_value = abs(value)

        # Handle infinity
        if math.isinf(abs_value):
            if self.has_inf:
                max_exp = (1 << self.exponent_bits) - 1
                return (sign << 7) | (max_exp << self.mantissa_bits)
            # Format doesn't support inf, clamp to max
            return self._clamp_to_max(sign)

        # Handle zero
        if abs_value == 0:
            return sign << 7

        # E0M7: pure fixed-point format
        if self.exponent_bits == 0:
            return self._to_bits_fixed_point(value, sign, abs_value)

        return self._to_bits_floating(sign, abs_value)

    def _clamp_to_max(self, sign: int) -> int:
        """Return bit pattern for maximum finite value with given sign."""
        if self.has_inf:
            max_exp = (1 << self.exponent_bits) - 2
        else:
            max_exp = (1 << self.exponent_bits) - 1
        max_mantissa = (1 << self.mantissa_bits) - 1
        return (sign << 7) | (max_exp << self.mantissa_bits) | max_mantissa

    def _to_bits_fixed_point(self, value: float, sign: int, abs_value: float) -> int:
        """Convert to bits for E0M7 fixed-point format."""
        max_mantissa = (1 << self.mantissa_bits) - 1
        max_value = max_mantissa / (1 << self.mantissa_bits)

        # Clamp to representable range
        if abs_value > max_value:
            abs_value = max_value

        # Convert to fixed-point with rounding
        scaled = abs_value * (1 << self.mantissa_bits)
        mantissa = int(round(scaled))

        # Clamp mantissa
        mantissa = min(mantissa, max_mantissa)

        return (sign << 7) | mantissa

    def _to_bits_floating(self, sign: int, abs_value: float) -> int:
        """Convert to bits for standard floating-point formats."""
        max_exp = (1 << self.exponent_bits) - 1
        max_normal_exp = max_exp - 1 if self.has_inf else max_exp

        # Calculate smallest denormalized value
        min_denorm = (2 ** (1 - self.bias)) / (1 << self.mantissa_bits)

        # Handle values too small to represent
        if abs_value < min_denorm / 2:
            return sign << 7  # Round to zero

        # Handle denormalized numbers
        min_normal = 2 ** (1 - self.bias)
        if abs_value < min_normal:
            return self._to_bits_denorm(sign, abs_value)

        # Calculate exponent
        log_val = math.floor(math.log2(abs_value))
        exponent = log_val + self.bias

        # Handle overflow (clamp to max or inf)
        if exponent > max_normal_exp:
            if self.has_inf:
                # Return infinity
                return (sign << 7) | (max_exp << self.mantissa_bits)
            else:
                return self._clamp_to_max(sign)

        # Handle underflow to denorm
        if exponent < 1:
            return self._to_bits_denorm(sign, abs_value)

        # Calculate mantissa with rounding
        significand = abs_value / (2 ** (exponent - self.bias))
        mantissa_float = (significand - 1.0) * (1 << self.mantissa_bits)

        # Round to nearest even
        mantissa = self._round_to_nearest_even(mantissa_float)
        max_mantissa = (1 << self.mantissa_bits) - 1

        # Handle mantissa overflow (round up to next exponent)
        if mantissa > max_mantissa:
            mantissa = 0
            exponent += 1
            if exponent > max_normal_exp:
                if self.has_inf:
                    return (sign << 7) | (max_exp << self.mantissa_bits)
                else:
                    return self._clamp_to_max(sign)

        return (sign << 7) | (exponent << self.mantissa_bits) | mantissa

    def _to_bits_denorm(self, sign: int, abs_value: float) -> int:
        """Convert to bits for denormalized numbers."""
        # Denorm: value = 2^(1-bias) * (mantissa / 2^M)
        # mantissa = value * 2^M / 2^(1-bias) = value * 2^(M + bias - 1)
        scale = 2 ** (self.mantissa_bits + self.bias - 1)
        mantissa_float = abs_value * scale

        mantissa = self._round_to_nearest_even(mantissa_float)
        max_mantissa = (1 << self.mantissa_bits) - 1

        # Handle mantissa overflow (becomes smallest normal)
        if mantissa > max_mantissa:
            # Becomes normalized with exp=1, mantissa=0
            return (sign << 7) | (1 << self.mantissa_bits)

        return (sign << 7) | mantissa

    def _round_to_nearest_even(self, value: float) -> int:
        """Round to nearest integer, with ties going to even."""
        floor_val = math.floor(value)
        frac = value - floor_val

        if frac < 0.5:
            return int(floor_val)
        elif frac > 0.5:
            return int(floor_val) + 1
        else:
            # Exactly 0.5: round to even
            if int(floor_val) % 2 == 0:
                return int(floor_val)
            else:
                return int(floor_val) + 1


# Format definitions
# E0M7: Fixed-point format with values in [-1, 1)
E0M7 = FP8Format(
    name="E0M7",
    exponent_bits=0,
    mantissa_bits=7,
    bias=0,
    has_inf=False,
    has_nan=False,
)

# E1M6: Two-scale format
E1M6 = FP8Format(
    name="E1M6",
    exponent_bits=1,
    mantissa_bits=6,
    bias=0,
    has_inf=False,
    has_nan=False,
)

# E3M4: Moderate range format (~0.06 to ~124)
E3M4 = FP8Format(
    name="E3M4",
    exponent_bits=3,
    mantissa_bits=4,
    bias=1,  # bias=1 for range ~0.06 to ~124
    has_inf=False,
    has_nan=False,
)

# E5M2: Standard FP8 format with wide dynamic range (IEEE-like)
E5M2 = FP8Format(
    name="E5M2",
    exponent_bits=5,
    mantissa_bits=2,
    bias=15,  # Standard bias = 2^(E-1) - 1 = 2^4 - 1 = 15
    has_inf=True,
    has_nan=True,
)

# E7M0: Powers of 2 only (extreme format for testing)
E7M0 = FP8Format(
    name="E7M0",
    exponent_bits=7,
    mantissa_bits=0,
    bias=63,  # Standard bias = 2^(E-1) - 1 = 2^6 - 1 = 63
    has_inf=False,
    has_nan=False,
)

# Format registry
FORMAT_REGISTRY: Dict[str, FP8Format] = {
    "E0M7": E0M7,
    "E1M6": E1M6,
    "E3M4": E3M4,
    "E5M2": E5M2,
    "E7M0": E7M0,
}

__all__ = [
    "FP8Format",
    "E0M7",
    "E1M6",
    "E3M4",
    "E5M2",
    "E7M0",
    "FORMAT_REGISTRY",
]
