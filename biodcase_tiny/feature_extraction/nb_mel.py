# taken this code from tflite-micro, but reimplemented the core bit in numba
# to avoid participants having to compile C++ code to run the pipeline
from dataclasses import dataclass

import numpy as np
import numba as nb
from numpy.typing import NDArray


# // Convert a `freq` in Hz to its value on the Mel scale.
# // See: https://en.wikipedia.org/wiki/Mel_scale
# // This function is only intended to be used wrapped as the python freq_to_mel
# // Why can't we just implement it in Python/numpy?
# // The original "Speech Micro" code is written in C and uses 32-bit 'float'
# // C types. Python's builtin floating point type is 64-bit wide, which results
# // in small differences in the output of the Python and C log() functions.
# // A Py wrapper is used in order to establish bit exactness with "Speech Micro",
# // while recognizing the slight loss in precision.
# float FreqToMel(float freq) { return 1127.0f * log1pf(freq / 700.0f); }
def freq_to_mel(freq):
    freq = np.float32(freq)
    return 1127.0 * np.log1p(freq / 700.0)


# A note about precision:
# The code to calculate center frequencies and weights uses floating point
# extensively. The original speech micro code is written in C and uses
# 32-bit 'float' C types. Python's floating point type is 64-bit by default,
# which resulted in slight differences that made verification harder.
# In order to establish parity with speech micro, and recognizing the slight
# loss in precision, numpy.float32 was used throughout this code instead of
# the default Python 'float' type. For the same reason, the function freq_to_mel
# wraps the same FreqToMel() C function used by Speech Micro.

FILTER_BANK_ALIGNMENT = 1
FILTER_BANK_CHANNEL_BLOCK_SIZE = 1
FILTER_BANK_WEIGHT_SCALING_BITS = 12


@dataclass
class FilterBankConstants:
    fft_start_index: int
    fft_end_index: int
    weights: NDArray[np.int16]
    unweights: NDArray[np.int16]
    n_channels: int
    ch_freq_starts: NDArray[np.int16]
    ch_weight_starts: NDArray[np.int16]
    ch_widths: NDArray[np.int16]


def _calc_center_freq(channel_num, lower_freq_limit, upper_freq_limit):
    """Calculate the center frequencies of filter_bank spectrum filter banks."""
    if lower_freq_limit < 0:
        raise ValueError("Lower frequency limit must be non negative")
    if lower_freq_limit > upper_freq_limit:
        raise ValueError("Lower frequency limit can't be larger than upper limit")
    mel_lower = freq_to_mel(lower_freq_limit)
    mel_upper = freq_to_mel(upper_freq_limit)
    mel_span = mel_upper - mel_lower
    mel_spacing = mel_span / np.float32(channel_num)
    channels = np.arange(1, channel_num + 1, dtype=np.float32)
    return mel_lower + (mel_spacing * channels)


def _quantize_filterbank_weight(float_weight, scale_bits):
    """Scale float filterbank weights return the integer weights and unweights."""
    weight = int(float_weight * (1 << scale_bits))
    unweight = int((1 - float_weight) * (1 << scale_bits))
    return weight, unweight


def _init_filter_bank_weights(spectrum_size, sample_rate, alignment,
                              channel_block_size, num_channels,
                              lower_band_limit, upper_band_limit):
    """Initialize mel-spectrum filter bank weights."""
    # How should we align things to index counts given the byte alignment?
    item_size = np.dtype("int16").itemsize
    if alignment < item_size:
        index_alignment = 1
    else:
        index_alignment = int(alignment / item_size)

    channel_frequency_starts = np.zeros(num_channels + 1, dtype=np.int16)
    channel_weight_starts = np.zeros(num_channels + 1, dtype=np.int16)
    channel_widths = np.zeros(num_channels + 1, dtype=np.int16)

    actual_channel_starts = np.zeros(num_channels + 1, dtype=np.int16)
    actual_channel_widths = np.zeros(num_channels + 1, dtype=np.int16)

    center_mel_freqs = _calc_center_freq(num_channels + 1, lower_band_limit,
                                         upper_band_limit)

    # (spectrum_size - 1) to exclude DC. Emulate Hidden Markov Model Toolkit (HTK)
    hz_per_sbin = (sample_rate / 2) / (spectrum_size - 1)
    # (1 + ...) to exclude DC.
    start_index = round(1 + (lower_band_limit / hz_per_sbin))

    # For each channel, we need to figure out what frequencies belong to it, and
    # how much padding we need to add so that we can efficiently multiply the
    # weights and unweights for accumulation. To simplify the multiplication
    # logic, all channels will have some multiplication to do (even if there are
    # no frequencies that accumulate to that channel) - they will be directed to
    # a set of zero weights.
    chan_freq_index_start = start_index
    weight_index_start = 0
    needs_zeros = 0

    for chan in range(num_channels + 1):
        # Keep jumping frequencies until we overshoot the bound on this channel.
        freq_index = chan_freq_index_start
        while freq_to_mel(freq_index * hz_per_sbin) <= center_mel_freqs[chan]:
            freq_index += 1

        width = freq_index - chan_freq_index_start
        actual_channel_starts[chan] = chan_freq_index_start
        actual_channel_widths[chan] = width

        if width == 0:
            # This channel doesn't actually get anything from the frequencies, it's
            # always zero. We need then to insert some 'zero' weights into the
            # output, and just redirect this channel to do a single multiplication at
            # this point. For simplicity, the zeros are placed at the beginning of
            # the weights arrays, so we have to go and update all the other
            # weight_starts to reflect this shift (but only once).
            channel_frequency_starts[chan] = 0
            channel_weight_starts[chan] = 0
            channel_widths[chan] = channel_block_size
            if needs_zeros == 0:
                needs_zeros = 1
                for j in range(chan):
                    channel_weight_starts[j] += channel_block_size
                weight_index_start += channel_block_size
        else:
            # How far back do we need to go to ensure that we have the proper
            # alignment?
            aligned_start = int(
                chan_freq_index_start / index_alignment) * index_alignment
            aligned_width = (chan_freq_index_start - aligned_start + width)
            padded_width = (int(
                (aligned_width - 1) / channel_block_size) + 1) * channel_block_size

            channel_frequency_starts[chan] = aligned_start
            channel_weight_starts[chan] = weight_index_start
            channel_widths[chan] = padded_width
            weight_index_start += padded_width
        chan_freq_index_start = freq_index

    # Allocate the two arrays to store the weights - weight_index_start contains
    # the index of what would be the next set of weights that we would need to
    # add, so that's how many weights we need to allocate.
    num_weights = weight_index_start
    weights = np.zeros(num_weights, dtype=np.int16)
    unweights = np.zeros(num_weights, dtype=np.int16)

    # Next pass, compute all the weights. Since everything has been memset to
    # zero, we only need to fill in the weights that correspond to some frequency
    # for a channel.
    end_index = 0
    mel_low = freq_to_mel(lower_band_limit)
    for chan in range(num_channels + 1):
        frequency = actual_channel_starts[chan]
        num_frequencies = actual_channel_widths[chan]
        frequency_offset = frequency - channel_frequency_starts[chan]
        weight_start = channel_weight_starts[chan]
        if chan == 0:
            denom_val = mel_low
        else:
            denom_val = center_mel_freqs[chan - 1]
        for j in range(num_frequencies):
            num = np.float32(center_mel_freqs[chan] -
                             freq_to_mel(frequency * hz_per_sbin))
            den = np.float32(center_mel_freqs[chan] - denom_val)
            weight = num / den
            # Make the float into an integer for the weights (and unweights).
            # Explicetly cast to int64. Numpy 2.0 introduces downcasting if we don't
            weight_index = weight_start + np.int64(frequency_offset) + j
            weights[weight_index], unweights[
                weight_index] = _quantize_filterbank_weight(
                weight, FILTER_BANK_WEIGHT_SCALING_BITS)
            # Explicetly cast to int64. Numpy 2.0 introduces downcasting if we don't
            frequency = np.int64(frequency) + 1
        if frequency > end_index:
            end_index = frequency

    if end_index >= spectrum_size:
        raise ValueError("Lower frequency limit can't be larger than upper limit")

    return FilterBankConstants(
        fft_start_index=start_index,
        fft_end_index=end_index,
        weights=weights,
        unweights=unweights,
        n_channels=num_channels,
        ch_freq_starts=channel_frequency_starts,
        ch_weight_starts=channel_weight_starts,
        ch_widths=channel_widths
    )


def calc_start_end_indices(fft_length, sample_rate, num_channels,
                           lower_band_limit, upper_band_limit):
    """Returns the range of FFT indices needed by filter_bank-spectrum filter bank.

    The caller can use the indices to avoid calculating the energy of FFT bins
    that won't be used.

    Args:
      fft_length: Length of FFT, in bins.
      sample_rate: Sample rate, in Hz.
      num_channels: Number of filter_bank-spectrum filter bank channels.
      lower_band_limit: lower limit of filter_bank-spectrum filterbank, in Hz.
      upper_band_limit: upper limit of filter_bank-spectrum filterbank, in Hz.

    Returns:
      A pair: start and end indices, in the range [0, fft_length)

    Raises:
      ValueError: If fft_length isn't a power of 2
    """
    if fft_length % 2 != 0:
        raise ValueError("FFT length must be an even number")
    spectrum_size = fft_length / 2 + 1
    (start_index, end_index, _, _, _, _,
     _) = _init_filter_bank_weights(spectrum_size, sample_rate,
                                    FILTER_BANK_ALIGNMENT,
                                    FILTER_BANK_CHANNEL_BLOCK_SIZE, num_channels,
                                    lower_band_limit, upper_band_limit)
    return start_index, end_index


@nb.njit((
    nb.int32,      # num_channels
    nb.int16[:],   # channel_freq_starts
    nb.int16[:],   # channel_weight_starts
    nb.int16[:],   # channel_widths
    nb.int16[:],   # weights
    nb.int16[:],   # unweights
    nb.uint32[:],  # input_array
    nb.uint64[:]   # output_array
))
def _filter_bank(num_channels, channel_freq_starts,
                 channel_weight_starts, channel_widths,
                 weights, unweights, input_array,
                 output_array):
    """
    Accumulate filterbank channels and rescale the result back to uint32.
    Please discard the first element of the returned output, as it's just used as scratch

    Parameters:
    -----------
    num_channels :
        Number of filterbank channels
    channel_freq_starts :
        Starting frequency indices for each channel
    channel_weight_starts :
        Starting weight indices for each channel
    channel_widths :
        Width of each channel
    weights :
        Weight values
    unweights :
        Unweight values
    input_array :
        Input array to process
    output_array :
        Output array to store rescaled results (should be of size num_channels + 1)
    """
    weight_accumulator = np.uint64(0)
    unweight_accumulator = np.uint64(0)
    for i in range(num_channels + 1):
        freq_start = channel_freq_starts[i]
        weight_start = channel_weight_starts[i]

        for j in range(channel_widths[i]):
            weight_accumulator += np.uint64(weights[weight_start + j]) * np.uint64(input_array[freq_start + j])
            unweight_accumulator += np.uint64(unweights[weight_start + j]) * np.uint64(input_array[freq_start + j])

        output_array[i] = weight_accumulator
        weight_accumulator = unweight_accumulator
        unweight_accumulator = np.uint64(0)


def filter_bank(cs: FilterBankConstants, input_array):
    output_array = np.zeros(cs.n_channels + 1, dtype=np.uint64)
    _filter_bank(cs.n_channels, cs.ch_freq_starts,
                 cs.ch_weight_starts, cs.ch_widths,
                 cs.weights, cs.unweights, input_array,
                 output_array)
    return output_array[1:]
