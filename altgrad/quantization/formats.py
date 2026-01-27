"""Floating-point format registry with bit-level transfer functions.

Supports both 8-bit and 16-bit formats with arbitrary exponent/mantissa splits.

Format Structure:
- N bits total: sign (1) + exponent (E) + mantissa (M), where E+M = N-1
- 8-bit: E+M=7 (E0M7 through E7M0)
- 16-bit: E+M=15 (E0M15 through E15M0)

Standard formats:
- BF16: E8M7 (16-bit, same exponent range as FP32)
- FP16: E5M10 (16-bit, IEEE half precision)
- FP8 E5M2: Standard ML 8-bit format
- FP8 E4M3: Common ML 8-bit format

Transfer Functions:
- to_real(bit_index): Convert bit pattern to real value
- to_bits(value): Convert real value to nearest bit pattern
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class FPFormat:
    """Specification for a floating-point format with arbitrary bit width.

    Attributes:
        name: Format identifier (e.g., "E5M2", "E8M7")
        total_bits: Total bits including sign (8 or 16)
        exponent_bits: Number of exponent bits
        mantissa_bits: Number of mantissa bits
        bias: Exponent bias for normalized numbers
        has_inf: Whether format supports infinity representation
        has_nan: Whether format supports NaN representation
    """

    name: str
    total_bits: int
    exponent_bits: int
    mantissa_bits: int
    bias: int
    has_inf: bool = False
    has_nan: bool = False

    def __post_init__(self):
        """Validate format specification."""
        expected_mantissa = self.total_bits - 1 - self.exponent_bits
        if self.mantissa_bits != expected_mantissa:
            raise ValueError(
                f"Invalid format: {self.total_bits} bits with E{self.exponent_bits} "
                f"should have M{expected_mantissa}, got M{self.mantissa_bits}"
            )

    @property
    def max_bit_value(self) -> int:
        """Maximum bit pattern value."""
        return (1 << self.total_bits) - 1

    @property
    def sign_shift(self) -> int:
        """Bit position of sign bit."""
        return self.total_bits - 1

    @property
    def max_representable_value(self) -> float:
        """Maximum finite positive value representable in this format."""
        if self.exponent_bits == 0:
            # Pure fixed-point
            max_mantissa = (1 << self.mantissa_bits) - 1
            return max_mantissa / (1 << self.mantissa_bits)

        # Standard floating-point
        if self.has_inf:
            max_exp = (1 << self.exponent_bits) - 2
        else:
            max_exp = (1 << self.exponent_bits) - 1

        max_mantissa = (1 << self.mantissa_bits) - 1
        mantissa_value = 1.0 + max_mantissa / (1 << self.mantissa_bits)

        exp_value = max_exp - self.bias
        # Protect against overflow for extreme formats
        if exp_value > 1023:
            return float('inf')
        if exp_value < -1074:
            return 0.0

        return (2 ** exp_value) * mantissa_value

    @property
    def min_positive_value(self) -> float:
        """Minimum positive value (smallest denorm)."""
        if self.exponent_bits == 0:
            return 1.0 / (1 << self.mantissa_bits)

        exp_value = 1 - self.bias - self.mantissa_bits
        # Protect against underflow for extreme formats
        if exp_value < -1074:
            return 0.0
        if exp_value > 1023:
            return float('inf')

        return 2 ** exp_value

    def to_real(self, bit_index: int) -> float:
        """Convert bit pattern to real value.

        Args:
            bit_index: Integer representing the bit pattern

        Returns:
            Real number represented by this bit pattern
        """
        if not 0 <= bit_index <= self.max_bit_value:
            raise ValueError(f"bit_index must be in [0, {self.max_bit_value}], got {bit_index}")

        # Extract components
        sign = (bit_index >> self.sign_shift) & 1
        mantissa_mask = (1 << self.mantissa_bits) - 1
        mantissa = bit_index & mantissa_mask

        if self.exponent_bits == 0:
            # Pure fixed-point format
            value = mantissa / (1 << self.mantissa_bits)
            return -value if sign else value

        # Extract exponent
        exponent = (bit_index >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)
        max_exp = (1 << self.exponent_bits) - 1

        # Handle special cases (inf, nan)
        if self.has_inf and exponent == max_exp:
            if mantissa == 0:
                return float('-inf') if sign else float('inf')
            if self.has_nan:
                return float('nan')

        # Handle zero
        if exponent == 0 and mantissa == 0:
            return -0.0 if sign else 0.0

        # Handle denormalized numbers
        if exponent == 0:
            value = (2 ** (1 - self.bias)) * (mantissa / (1 << self.mantissa_bits))
            return -value if sign else value

        # Normalized numbers
        mantissa_value = 1.0 + mantissa / (1 << self.mantissa_bits)
        value = (2 ** (exponent - self.bias)) * mantissa_value
        return -value if sign else value

    def to_bits(self, value: float) -> int:
        """Convert real value to nearest bit pattern.

        Args:
            value: Real number to quantize

        Returns:
            Integer representing the nearest bit pattern
        """
        # Handle special values
        if math.isnan(value):
            if self.has_nan:
                max_exp = (1 << self.exponent_bits) - 1
                return (max_exp << self.mantissa_bits) | 1
            return 0

        # Handle sign
        sign = 1 if value < 0 or (value == 0 and math.copysign(1, value) < 0) else 0
        abs_value = abs(value)

        # Handle infinity
        if math.isinf(abs_value):
            if self.has_inf:
                max_exp = (1 << self.exponent_bits) - 1
                return (sign << self.sign_shift) | (max_exp << self.mantissa_bits)
            return self._clamp_to_max(sign)

        # Handle zero
        if abs_value == 0:
            return sign << self.sign_shift

        # Fixed-point format
        if self.exponent_bits == 0:
            return self._to_bits_fixed_point(sign, abs_value)

        return self._to_bits_floating(sign, abs_value)

    def _clamp_to_max(self, sign: int) -> int:
        """Return bit pattern for maximum finite value with given sign."""
        if self.has_inf:
            max_exp = (1 << self.exponent_bits) - 2
        else:
            max_exp = (1 << self.exponent_bits) - 1
        max_mantissa = (1 << self.mantissa_bits) - 1
        return (sign << self.sign_shift) | (max_exp << self.mantissa_bits) | max_mantissa

    def _to_bits_fixed_point(self, sign: int, abs_value: float) -> int:
        """Convert to bits for fixed-point format."""
        max_mantissa = (1 << self.mantissa_bits) - 1
        max_value = max_mantissa / (1 << self.mantissa_bits)

        if abs_value > max_value:
            abs_value = max_value

        scaled = abs_value * (1 << self.mantissa_bits)
        mantissa = int(round(scaled))
        mantissa = min(mantissa, max_mantissa)

        return (sign << self.sign_shift) | mantissa

    def _to_bits_floating(self, sign: int, abs_value: float) -> int:
        """Convert to bits for floating-point formats."""
        max_exp = (1 << self.exponent_bits) - 1
        max_normal_exp = max_exp - 1 if self.has_inf else max_exp

        min_denorm = (2 ** (1 - self.bias)) / (1 << self.mantissa_bits)

        if abs_value < min_denorm / 2:
            return sign << self.sign_shift

        min_normal = 2 ** (1 - self.bias)
        if abs_value < min_normal:
            return self._to_bits_denorm(sign, abs_value)

        log_val = math.floor(math.log2(abs_value))
        exponent = log_val + self.bias

        if exponent > max_normal_exp:
            if self.has_inf:
                return (sign << self.sign_shift) | (max_exp << self.mantissa_bits)
            else:
                return self._clamp_to_max(sign)

        if exponent < 1:
            return self._to_bits_denorm(sign, abs_value)

        significand = abs_value / (2 ** (exponent - self.bias))
        mantissa_float = (significand - 1.0) * (1 << self.mantissa_bits)

        mantissa = self._round_to_nearest_even(mantissa_float)
        max_mantissa = (1 << self.mantissa_bits) - 1

        if mantissa > max_mantissa:
            mantissa = 0
            exponent += 1
            if exponent > max_normal_exp:
                if self.has_inf:
                    return (sign << self.sign_shift) | (max_exp << self.mantissa_bits)
                else:
                    return self._clamp_to_max(sign)

        return (sign << self.sign_shift) | (exponent << self.mantissa_bits) | mantissa

    def _to_bits_denorm(self, sign: int, abs_value: float) -> int:
        """Convert to bits for denormalized numbers."""
        scale = 2 ** (self.mantissa_bits + self.bias - 1)
        mantissa_float = abs_value * scale

        mantissa = self._round_to_nearest_even(mantissa_float)
        max_mantissa = (1 << self.mantissa_bits) - 1

        if mantissa > max_mantissa:
            return (sign << self.sign_shift) | (1 << self.mantissa_bits)

        return (sign << self.sign_shift) | mantissa

    def _round_to_nearest_even(self, value: float) -> int:
        """Round to nearest integer, with ties going to even."""
        floor_val = math.floor(value)
        frac = value - floor_val

        if frac < 0.5:
            return int(floor_val)
        elif frac > 0.5:
            return int(floor_val) + 1
        else:
            if int(floor_val) % 2 == 0:
                return int(floor_val)
            else:
                return int(floor_val) + 1


# Backward compatibility alias
FP8Format = FPFormat


def _compute_bias(exponent_bits: int, standard: bool = True) -> int:
    """Compute standard IEEE-style bias for given exponent bits.

    Args:
        exponent_bits: Number of exponent bits
        standard: If True, use IEEE bias (2^(E-1) - 1)

    Returns:
        Bias value
    """
    if exponent_bits == 0:
        return 0
    if standard:
        return (1 << (exponent_bits - 1)) - 1
    return 0


# ============================================================================
# 8-BIT FORMATS (E+M=7)
# ============================================================================

# All 8-bit E/M splits
FP8_E0M7 = FPFormat(name="FP8_E0M7", total_bits=8, exponent_bits=0, mantissa_bits=7, bias=0)
FP8_E1M6 = FPFormat(name="FP8_E1M6", total_bits=8, exponent_bits=1, mantissa_bits=6, bias=0)
FP8_E2M5 = FPFormat(name="FP8_E2M5", total_bits=8, exponent_bits=2, mantissa_bits=5, bias=1)
FP8_E3M4 = FPFormat(name="FP8_E3M4", total_bits=8, exponent_bits=3, mantissa_bits=4, bias=3)
FP8_E4M3 = FPFormat(name="FP8_E4M3", total_bits=8, exponent_bits=4, mantissa_bits=3, bias=7, has_inf=False, has_nan=True)
FP8_E5M2 = FPFormat(name="FP8_E5M2", total_bits=8, exponent_bits=5, mantissa_bits=2, bias=15, has_inf=True, has_nan=True)
FP8_E6M1 = FPFormat(name="FP8_E6M1", total_bits=8, exponent_bits=6, mantissa_bits=1, bias=31)
FP8_E7M0 = FPFormat(name="FP8_E7M0", total_bits=8, exponent_bits=7, mantissa_bits=0, bias=63)


# ============================================================================
# 16-BIT FORMATS (E+M=15)
# ============================================================================

# Standard 16-bit formats
FP16 = FPFormat(name="FP16", total_bits=16, exponent_bits=5, mantissa_bits=10, bias=15, has_inf=True, has_nan=True)  # IEEE half
BF16 = FPFormat(name="BF16", total_bits=16, exponent_bits=8, mantissa_bits=7, bias=127, has_inf=True, has_nan=True)  # Brain float

# All 16-bit E/M splits
FP16_E0M15 = FPFormat(name="FP16_E0M15", total_bits=16, exponent_bits=0, mantissa_bits=15, bias=0)
FP16_E1M14 = FPFormat(name="FP16_E1M14", total_bits=16, exponent_bits=1, mantissa_bits=14, bias=0)
FP16_E2M13 = FPFormat(name="FP16_E2M13", total_bits=16, exponent_bits=2, mantissa_bits=13, bias=1)
FP16_E3M12 = FPFormat(name="FP16_E3M12", total_bits=16, exponent_bits=3, mantissa_bits=12, bias=3)
FP16_E4M11 = FPFormat(name="FP16_E4M11", total_bits=16, exponent_bits=4, mantissa_bits=11, bias=7)
FP16_E5M10 = FP16  # Same as IEEE FP16
FP16_E6M9 = FPFormat(name="FP16_E6M9", total_bits=16, exponent_bits=6, mantissa_bits=9, bias=31)
FP16_E7M8 = FPFormat(name="FP16_E7M8", total_bits=16, exponent_bits=7, mantissa_bits=8, bias=63)
FP16_E8M7 = BF16  # Same as BF16
FP16_E9M6 = FPFormat(name="FP16_E9M6", total_bits=16, exponent_bits=9, mantissa_bits=6, bias=255)
FP16_E10M5 = FPFormat(name="FP16_E10M5", total_bits=16, exponent_bits=10, mantissa_bits=5, bias=511)
FP16_E11M4 = FPFormat(name="FP16_E11M4", total_bits=16, exponent_bits=11, mantissa_bits=4, bias=1023)
FP16_E12M3 = FPFormat(name="FP16_E12M3", total_bits=16, exponent_bits=12, mantissa_bits=3, bias=2047)
FP16_E13M2 = FPFormat(name="FP16_E13M2", total_bits=16, exponent_bits=13, mantissa_bits=2, bias=4095)
FP16_E14M1 = FPFormat(name="FP16_E14M1", total_bits=16, exponent_bits=14, mantissa_bits=1, bias=8191)
FP16_E15M0 = FPFormat(name="FP16_E15M0", total_bits=16, exponent_bits=15, mantissa_bits=0, bias=16383)


# ============================================================================
# FORMAT REGISTRIES
# ============================================================================

# 8-bit formats
FP8_FORMATS: Dict[str, FPFormat] = {
    "FP8_E0M7": FP8_E0M7,
    "FP8_E1M6": FP8_E1M6,
    "FP8_E2M5": FP8_E2M5,
    "FP8_E3M4": FP8_E3M4,
    "FP8_E4M3": FP8_E4M3,
    "FP8_E5M2": FP8_E5M2,
    "FP8_E6M1": FP8_E6M1,
    "FP8_E7M0": FP8_E7M0,
}

# 16-bit formats
FP16_FORMATS: Dict[str, FPFormat] = {
    "FP16": FP16,
    "BF16": BF16,
    "FP16_E0M15": FP16_E0M15,
    "FP16_E1M14": FP16_E1M14,
    "FP16_E2M13": FP16_E2M13,
    "FP16_E3M12": FP16_E3M12,
    "FP16_E4M11": FP16_E4M11,
    "FP16_E5M10": FP16,
    "FP16_E6M9": FP16_E6M9,
    "FP16_E7M8": FP16_E7M8,
    "FP16_E8M7": BF16,
    "FP16_E9M6": FP16_E9M6,
    "FP16_E10M5": FP16_E10M5,
    "FP16_E11M4": FP16_E11M4,
    "FP16_E12M3": FP16_E12M3,
    "FP16_E13M2": FP16_E13M2,
    "FP16_E14M1": FP16_E14M1,
    "FP16_E15M0": FP16_E15M0,
}

# Combined registry (all formats)
FORMAT_REGISTRY: Dict[str, FPFormat] = {
    **FP8_FORMATS,
    **FP16_FORMATS,
    # Legacy aliases (without FP8_ prefix)
    "E0M7": FP8_E0M7,
    "E1M6": FP8_E1M6,
    "E2M5": FP8_E2M5,
    "E3M4": FP8_E3M4,
    "E4M3": FP8_E4M3,
    "E5M2": FP8_E5M2,
    "E6M1": FP8_E6M1,
    "E7M0": FP8_E7M0,
}

# Legacy aliases for backward compatibility
E0M7 = FP8_E0M7
E1M6 = FP8_E1M6
E2M5 = FP8_E2M5
E3M4 = FP8_E3M4
E4M3 = FP8_E4M3
E5M2 = FP8_E5M2
E6M1 = FP8_E6M1
E7M0 = FP8_E7M0


def list_formats(bits: int = None) -> list:
    """List available formats.

    Args:
        bits: Filter by bit width (8 or 16), or None for all

    Returns:
        List of format names
    """
    if bits == 8:
        return list(FP8_FORMATS.keys())
    elif bits == 16:
        return list(FP16_FORMATS.keys())
    else:
        return list(FORMAT_REGISTRY.keys())


def format_summary() -> str:
    """Generate summary table of all formats."""
    lines = []
    lines.append("=" * 80)
    lines.append("FLOATING-POINT FORMAT REGISTRY")
    lines.append("=" * 80)
    lines.append("")

    lines.append("8-BIT FORMATS (E+M=7)")
    lines.append("-" * 80)
    lines.append(f"{'Name':<12} {'Exp':>4} {'Man':>4} {'Bias':>6} {'Max Value':>15} {'Min Positive':>15}")
    lines.append("-" * 80)
    for name, fmt in FP8_FORMATS.items():
        lines.append(
            f"{name:<12} {fmt.exponent_bits:>4} {fmt.mantissa_bits:>4} {fmt.bias:>6} "
            f"{fmt.max_representable_value:>15.6g} {fmt.min_positive_value:>15.6g}"
        )
    lines.append("")

    lines.append("16-BIT FORMATS (E+M=15)")
    lines.append("-" * 80)
    lines.append(f"{'Name':<12} {'Exp':>4} {'Man':>4} {'Bias':>6} {'Max Value':>15} {'Min Positive':>15}")
    lines.append("-" * 80)
    for name, fmt in FP16_FORMATS.items():
        if name in ("FP16_E5M10", "FP16_E8M7"):
            continue  # Skip aliases
        lines.append(
            f"{name:<12} {fmt.exponent_bits:>4} {fmt.mantissa_bits:>4} {fmt.bias:>6} "
            f"{fmt.max_representable_value:>15.6g} {fmt.min_positive_value:>15.6g}"
        )
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


__all__ = [
    # Core class
    "FPFormat",
    "FP8Format",  # Backward compatibility alias
    # 8-bit formats
    "FP8_E0M7", "FP8_E1M6", "FP8_E2M5", "FP8_E3M4",
    "FP8_E4M3", "FP8_E5M2", "FP8_E6M1", "FP8_E7M0",
    # 16-bit formats
    "FP16", "BF16",
    "FP16_E0M15", "FP16_E1M14", "FP16_E2M13", "FP16_E3M12",
    "FP16_E4M11", "FP16_E6M9", "FP16_E7M8", "FP16_E9M6",
    "FP16_E10M5", "FP16_E11M4", "FP16_E12M3", "FP16_E13M2",
    "FP16_E14M1", "FP16_E15M0",
    # Registries
    "FP8_FORMATS", "FP16_FORMATS", "FORMAT_REGISTRY",
    # Legacy aliases
    "E0M7", "E1M6", "E2M5", "E3M4", "E4M3", "E5M2", "E6M1", "E7M0",
    # Utilities
    "list_formats", "format_summary",
]
