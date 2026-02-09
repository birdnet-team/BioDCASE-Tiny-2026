import numpy as np
import numba as nb
from numba import int16, int32, uint16, boolean, optional

# Constants
ADD_ROUND_MULT = 0x7fff
MULT_SHIFT_CONST = 0x7fff


@nb.njit(int16(int16, int16, int16, int16, int16, int32))
def xtfixed_bf_1(a0, a1, a2, a3, a4, result_shift):
    result = a0 * MULT_SHIFT_CONST
    result -= int32(a1) * int32(a2) + int32(a3) * int32(a4)
    result += ADD_ROUND_MULT
    result = result >> result_shift
    return int16(result)


@nb.njit(int16(int16, int16, int16, int16, int16, int32))
def xtfixed_bf_2(a0, a1, a2, a3, a4, result_shift):
    result = a0 * MULT_SHIFT_CONST
    result -= (int32(a1) * int32(a2) - int32(a3) * int32(a4))
    result += ADD_ROUND_MULT
    result = result >> result_shift
    return int16(result)


@nb.njit(int16(int16, int16, int16, int16, int16, int32))
def xtfixed_bf_3(a0, a1, a2, a3, a4, result_shift):
    result = a0 * MULT_SHIFT_CONST
    result += int32(a1) * int32(a2) + int32(a3) * int32(a4)
    result += ADD_ROUND_MULT
    result = result >> result_shift
    return int16(result)


@nb.njit(int16(int16, int16, int16, int16, int16, int32))
def xtfixed_bf_4(a0, a1, a2, a3, a4, result_shift):
    result = a0 * MULT_SHIFT_CONST
    result += int32(a1) * int32(a2) - int32(a3) * int32(a4)
    result += ADD_ROUND_MULT
    result = result >> result_shift
    return int16(result)


@nb.njit(boolean(int32))
def dsp_is_power_of_two(x):
    return (x != 0) and ((x & (x - 1)) == 0)


@nb.njit(int32(int32))
def dsp_power_of_two(x):
    for i in range(32):
        x = x >> 1
        if x == 0:
            return i
    return 0


@nb.njit(uint16(uint16, uint16, int32))
def reverse_sc16(x, N, order):
    b = x
    b = (b & 0xff00) >> 8 | (b & 0x00ff) << 8
    b = (b & 0xf0f0) >> 4 | (b & 0x0f0f) << 4
    b = (b & 0xcccc) >> 2 | (b & 0x3333) << 2
    b = (b & 0xaaaa) >> 1 | (b & 0x5555) << 1
    return b >> (16 - order)


@nb.njit(boolean(int16[:], int32))
def dsps_gen_w_r2_sc16(w, N):
    if not dsp_is_power_of_two(N):
        return False  # Error

    e = 2.0 * np.pi / N

    for i in range(N >> 1):
        w[2*i] = int16(np.int16(32767 * np.cos(i * e)))
        w[2*i+1] = int16(np.int16(32767 * np.sin(i * e)))

    return True


@nb.njit()
def dsps_cplx2reC_sc16(data, N):
    if not dsp_is_power_of_two(N):
        return False  # Error

    n2 = N << 1  # we will operate with int32 indexes

    for i in range(N // 4):
        kl_re = data[2*(i+1)]
        kl_im = data[2*(i+1)+1]
        nl_re = data[2*(N-i-1)]
        nl_im = data[2*(N-i-1)+1]
        kh_re = data[2*(i+1+N//2)]
        kh_im = data[2*(i+1+N//2)+1]
        nh_re = data[2*(N-i-1-N//2)]
        nh_im = data[2*(N-i-1-N//2)+1]

        data[i*2+0+2] = kl_re + nl_re
        data[i*2+1+2] = kl_im - nl_im

        data[n2-i*2-1-N] = kh_re + nh_re
        data[n2-i*2-2-N] = kh_im - nh_im

        data[i*2+0+2+N] = kl_im + nl_im
        data[i*2+1+2+N] = kl_re - nl_re

        data[n2-i*2-1] = kh_im + nh_im
        data[n2-i*2-2] = kh_re - nh_re

    data[N] = data[1]
    data[1] = 0
    data[N+1] = 0

    return True


@nb.njit()
def dsps_cplx2real_sc16_ansi(data, N, w_table):
    order = dsp_power_of_two(N)

    # Original formula...
    # result[0].re = result[0].re + result[0].im;
    # result[N].re = result[0].re - result[0].im;
    # result[0].im = 0;
    # result[N].im = 0;
    # Optimized one:
    tmp_re = data[0]
    data[0] = (tmp_re + data[1]) >> 1
    data[1] = (tmp_re - data[1]) >> 1

    for k in range(1, N // 2 + 1):
        fpk_re = data[2*k]
        fpk_im = data[2*k+1]
        fpnk_re = data[2*(N-k)]
        fpnk_im = data[2*(N-k)+1]

        f1k_re = fpk_re + fpnk_re
        f1k_im = fpk_im - fpnk_im
        f2k_re = fpk_re - fpnk_re
        f2k_im = fpk_im + fpnk_im

        table_index = reverse_sc16(k, N, order)

        w_re = w_table[2*table_index]
        w_im = w_table[2*table_index+1]

        tw_re = (w_re * f2k_im - w_im * f2k_re) >> 15
        tw_im = (w_re * f2k_re + w_im * f2k_im) >> 15

        data[2*k] = (f1k_re + tw_re) >> 2
        data[2*k+1] = (f1k_im - tw_im) >> 2
        data[2*(N-k)] = (f1k_re - tw_re) >> 2
        data[2*(N-k)+1] = -(f1k_im + tw_im) >> 2

    return True


@nb.njit(boolean(int16[:], int32))
def dsps_bit_rev_sc16_ansi(data, N):
    """Bit reversal for complex int16 data"""
    if not dsp_is_power_of_two(N):
        return False

    # Create a view of data as uint32 (combining real/imag parts)
    data_u32 = np.zeros(N, dtype=np.uint32)
    for i in range(N):
        data_u32[i] = (int(data[2*i]) & 0xFFFF) | ((int(data[2*i+1]) & 0xFFFF) << 16)

    j = 0
    for i in range(1, N-1):
        k = N >> 1
        while k <= j:
            j -= k
            k >>= 1
        j += k
        if i < j:
            # Swap complex values as 32-bit
            temp = data_u32[j]
            data_u32[j] = data_u32[i]
            data_u32[i] = temp

    # Copy back to original array
    for i in range(N):
        data[2*i] = int16(data_u32[i] & 0xFFFF)
        data[2*i+1] = int16(data_u32[i] >> 16)

    return True


@nb.njit(boolean(int16[:], int32, int16[:]))
def dsps_fft2r_sc16_ansi(data, N, sc_table):
    """
    Perform FFT on int16 data

    Parameters:
    -----------
    data :
        Input/output array of int16 type, alternating real and imaginary parts
    N :
        Number of complex points
    sc_table :
        Pre-computed twiddle factors table

    Returns:
    --------
    success : bool
        True if FFT was successful
    """
    if not dsp_is_power_of_two(N):
        return False

    # Create temporary arrays for complex values
    # This avoids the memory layout issues
    data_re = np.zeros(N, dtype=np.int16)
    data_im = np.zeros(N, dtype=np.int16)

    # Extract real and imaginary parts
    for i in range(N):
        data_re[i] = data[2*i]
        data_im[i] = data[2*i+1]

    # Main FFT loop
    ie = 1
    N2 = N // 2

    while N2 > 0:
        ia = 0
        for j in range(ie):
            cs_re = sc_table[2*j]
            cs_im = sc_table[2*j+1]

            for i in range(N2):
                m = ia + N2

                # Get values
                a_data_re = data_re[ia]
                a_data_im = data_im[ia]
                m_data_re = data_re[m]
                m_data_im = data_im[m]

                # Butterfly operations
                # These match the C implementation exactly
                m1_re = xtfixed_bf_1(a_data_re, cs_re, m_data_re, cs_im, m_data_im, 16)
                m1_im = xtfixed_bf_2(a_data_im, cs_re, m_data_im, cs_im, m_data_re, 16)

                m2_re = xtfixed_bf_3(a_data_re, cs_re, m_data_re, cs_im, m_data_im, 16)
                m2_im = xtfixed_bf_4(a_data_im, cs_re, m_data_im, cs_im, m_data_re, 16)

                # Store results
                data_re[m] = m1_re
                data_im[m] = m1_im
                data_re[ia] = m2_re
                data_im[ia] = m2_im

                ia += 1

            ia += N2

        ie <<= 1
        N2 >>= 1

    # Copy back to the original array
    for i in range(N):
        data[2*i] = data_re[i]
        data[2*i+1] = data_im[i]

    return True


@nb.njit(boolean(int16[:], optional(int16[:])))
def dsps_fft2r_sc16(data, sc_table=None):
    """
    Main FFT function that handles initialization and execution

    Parameters:
    -----------
    data :
        Input/output data array of int16 type, alternating real and imaginary
    sc_table :
        Pre-computed twiddle factors table

    Returns:
    --------
    success : bool
        True if FFT was successful
    """
    N = len(data) // 2  # Number of complex points

    if not dsp_is_power_of_two(N):
        return False

    # Generate twiddle factors if not provided
    if sc_table is None:
        sc_table = np.zeros(N*2, dtype=np.int16)
        if not dsps_gen_w_r2_sc16(sc_table, N*2):
            return False
        if not dsps_bit_rev_sc16_ansi(sc_table, N):
            return False

    # Run FFT
    return dsps_fft2r_sc16_ansi(data, N, sc_table)


def gen_twiddle(N):
    w_table = np.zeros(2 * N, dtype=np.int16)
    dsps_gen_w_r2_sc16(w_table, 2 * N)
    dsps_bit_rev_sc16_ansi(w_table, N)
    return w_table


def do_fft(data, sc_table):
    cpx_interlvd = np.zeros(data.shape[0] * 2, dtype=data.dtype)
    cpx_interlvd[::2] = data
    n = len(cpx_interlvd) // 2
    if not dsps_fft2r_sc16_ansi(cpx_interlvd, n, sc_table):
        raise ValueError("Window size is not a power of 2")
    dsps_bit_rev_sc16_ansi(cpx_interlvd, n)
    dsps_cplx2real_sc16_ansi(cpx_interlvd, n, sc_table)
    return cpx_interlvd