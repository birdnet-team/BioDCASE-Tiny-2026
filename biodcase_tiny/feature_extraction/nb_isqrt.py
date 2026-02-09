import numpy as np
import numba as nb


@nb.njit
def most_significant_bit(num):
    """Find the position of the most significant bit in an integer."""
    if num == 0:
        return 0
    pos = 0
    while num:
        num >>= 1
        pos += 1
    return pos


@nb.njit(nb.uint16(nb.uint32))
def sqrt32(num):
    """
    Calculate the square root of a 32-bit unsigned integer.
    Returns a 16-bit unsigned integer result.
    """
    if num == 0:
        return 0

    res = np.uint32(0)
    max_bit_number = 32 - most_significant_bit(num)
    max_bit_number |= 1
    bit = np.uint32(1) << np.uint32(31 - max_bit_number)
    iterations = (31 - max_bit_number) // 2 + 1

    while iterations > 0:
        if num >= res + bit:
            num -= res + bit
            res = (res >> np.uint32(1)) + bit
        else:
            res >>= np.uint32(1)

        bit >>= np.uint32(2)
        iterations -= 1

    # Do rounding - if we have the bits
    if num > res and res != 0xFFFF:
        res += 1

    return np.uint16(res)


@nb.njit(nb.uint32(nb.uint64))
def sqrt64(num):
    """
    Calculate the square root of a 64-bit unsigned integer.
    Returns a 32-bit unsigned integer result.
    """
    # Take a shortcut for numbers that fit in 32 bits
    if (num >> 32) == 0:
        return sqrt32(np.uint32(num))

    res = np.uint64(0)
    max_bit_number = 64 - most_significant_bit(num)
    max_bit_number |= 1
    bit = np.uint64(1) << np.uint64(63 - max_bit_number)
    iterations = (63 - max_bit_number) // 2 + 1

    while iterations > 0:
        if num >= res + bit:
            num -= res + bit
            res = (res >> np.uint64(1)) + bit
        else:
            res >>= np.uint64(1)

        bit >>= np.uint64(2)
        iterations -= 1

    # Do rounding - if we have the bits
    if num > res and res != 0xFFFFFFFF:
        res += 1

    return np.uint32(res)


@nb.njit()
def vec_sqrt64(input_array):
    """
    Apply sqrt64 to each element of the input array
    Uses parallel processing for better performance on large arrays.

    Parameters:
    -----------
    input_array : ndarray of uint64
        Input array to process
    """
    num_channels = len(input_array)
    output_array = np.empty(num_channels, dtype=np.uint32)
    for i in nb.prange(num_channels):
        output_array[i] = sqrt64(input_array[i])
    return output_array