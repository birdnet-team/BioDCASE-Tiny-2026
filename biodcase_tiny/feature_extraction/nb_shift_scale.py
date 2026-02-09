import numpy as np
from numba import njit, int16, int32
from biodcase_tiny.feature_extraction.nb_isqrt import most_significant_bit


@njit(int32(int16[:]))
def shift_scale_up(input_array):
    """Scale an int16 array via bit shift, with the highest possible shift that doesn't cause overflow.
    Return the shift."""
    size = len(input_array)
    max_val = np.abs(input_array).max()
    scale_bits = 16 - most_significant_bit(max_val) - 1

    if scale_bits <= 0:
        scale_bits = 0

    for i in range(size):
        input_array[i] = input_array[i] * (1 << scale_bits)

    return scale_bits


@njit()
def shift_scale_down(input_array, scale_bits):
    for i in range(len(input_array)):
        input_array[i] = input_array[i] >> scale_bits
