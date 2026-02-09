import numpy as np
import numba as nb

MAX_UINT16 = np.iinfo(np.uint16).max

# Create the lookup table as a numpy array
kLogLut = np.array([
    0,    224,  442,  654,  861,  1063, 1259, 1450, 1636, 1817, 1992, 2163,
    2329, 2490, 2646, 2797, 2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848,
    3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714, 4791, 4864, 4934,
    5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
    5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633,
    5626, 5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370,
    5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762,
    4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
    3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659,
    2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224,
    1094, 963,  830,  695,  559,  421,  282,  142,  0,    0
], dtype=np.uint16)

# Constants
kLogSegmentsLog2 = 7
kLogScale = 65536
kLogScaleLog2 = 16
kLogCoeff = 45426


@nb.njit(nb.uint32(nb.uint32))
def most_significant_bit32(x):
    temp = nb.uint32(0)
    while x:
        x = x >> 1
        temp += 1
    return temp


@nb.njit(nb.uint32(nb.uint32, nb.uint32))
def log2_fraction_part32(x, log2x):
    # Part 1
    frac = nb.int32(x - (1 << log2x))
    if log2x < kLogScaleLog2:
        frac <<= kLogScaleLog2 - log2x
    else:
        frac >>= log2x - kLogScaleLog2

    # Part 2
    base_seg = nb.uint32(frac >> (kLogScaleLog2 - kLogSegmentsLog2))
    seg_unit = nb.uint32((nb.int32(1) << kLogScaleLog2) >> kLogSegmentsLog2)

    # ASSERT(base_seg < kLogSegments) would be here in the original code
    c0 = nb.int32(kLogLut[base_seg])
    c1 = nb.int32(kLogLut[base_seg + 1])
    seg_base = nb.int32(seg_unit * base_seg)
    rel_pos = nb.int32(((c1 - c0) * (frac - seg_base)) >> kLogScaleLog2)

    return nb.uint32(frac + c0 + rel_pos)


@nb.njit(nb.uint32(nb.uint32, nb.uint64))
def log32(x, out_scale):
    # ASSERT(x != 0)
    integer = most_significant_bit32(x) - 1
    fraction = log2_fraction_part32(x, integer)
    log2 = (integer << kLogScaleLog2) + fraction
    round_val = nb.uint32(kLogScale // 2)

    # Use np.uint64 to handle the multiplication without overflow
    loge = nb.uint32(((np.uint64(kLogCoeff) * np.uint64(log2)) + np.uint64(round_val)) >> kLogScaleLog2)

    # Finally scale to our output scale
    loge_scaled = nb.uint32((out_scale * loge + round_val) >> kLogScaleLog2)

    return loge_scaled


@nb.njit(nb.uint16[:](nb.uint32[:], nb.uint32, nb.uint8))
def vec_log32(input_array, out_scale, correction_bits):
    """
    Apply log32 to each element of the input array and scale down by scale_down_bits.
    Uses parallel processing for better performance on large arrays.

    Parameters:
    -----------
    input_array
        Input array to process
    out_scale :
        Number of bits to shift left before taking log
    """
    num_channels = len(input_array)
    output_array = np.empty(num_channels, dtype=np.uint16)
    for i in nb.prange(num_channels):
        output_array[i] = np.minimum(log32(input_array[i] << correction_bits, out_scale), MAX_UINT16)
    return output_array