from dataclasses import dataclass

import flatbuffers
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

import biodcase_tiny.feature_extraction.feature_config_generated as feature_config
from biodcase_tiny.feature_extraction.nb_fft import gen_twiddle, dsps_fft2r_sc16_ansi, do_fft
from biodcase_tiny.feature_extraction.nb_log32 import log32, vec_log32
from biodcase_tiny.feature_extraction.nb_mel import _init_filter_bank_weights, FILTER_BANK_ALIGNMENT, \
    FILTER_BANK_CHANNEL_BLOCK_SIZE, filter_bank, FilterBankConstants

from biodcase_tiny.feature_extraction.nb_isqrt import vec_sqrt64
from biodcase_tiny.feature_extraction.nb_shift_scale import shift_scale_up, shift_scale_down


@dataclass
class FeatureConstants:
    window_scaling_bits: np.uint8
    mel_post_scaling_bits: np.uint8
    hanning_window: NDArray[np.int16]
    fft_twiddle: NDArray[np.int16]
    mel_constants: FilterBankConstants


def make_constants(win_samples, sample_rate,  window_scaling_bits,
                   mel_n_channels, mel_low_hz, mel_high_hz, mel_post_scaling_bits):
    mel_constants = _init_filter_bank_weights(
        win_samples // 2, sample_rate, FILTER_BANK_ALIGNMENT,
        FILTER_BANK_CHANNEL_BLOCK_SIZE, mel_n_channels,
        mel_low_hz, mel_high_hz
    )
    hanning = np.round(np.hanning(win_samples) * (2 ** window_scaling_bits)).astype(np.int16)
    sc_table = gen_twiddle(win_samples)
    return FeatureConstants(
        window_scaling_bits=window_scaling_bits,
        mel_post_scaling_bits=mel_post_scaling_bits,
        hanning_window=hanning,
        fft_twiddle=sc_table,
        mel_constants=mel_constants
    )


def convert_constants(c: FeatureConstants):
    builder = flatbuffers.Builder(0)

    hanning_window_offset = builder.CreateNumpyVector(c.hanning_window)
    fft_twiddle_offset = builder.CreateNumpyVector(c.fft_twiddle)
    channel_freq_offset = builder.CreateNumpyVector(c.mel_constants.ch_freq_starts)
    channel_weight_offset = builder.CreateNumpyVector(c.mel_constants.ch_weight_starts)
    channel_width_offset = builder.CreateNumpyVector(c.mel_constants.ch_widths)
    weights_offset = builder.CreateNumpyVector(c.mel_constants.weights)
    unweights_offset = builder.CreateNumpyVector(c.mel_constants.unweights)

    # Build FilterbankConfig
    feature_config.FilterbankConfigStart(builder)
    feature_config.FilterbankConfigAddFftStartIdx(builder, c.mel_constants.fft_start_index)
    feature_config.FilterbankConfigAddFftEndIdx(builder, c.mel_constants.fft_end_index)
    feature_config.FilterbankConfigAddWeights(builder, weights_offset)
    feature_config.FilterbankConfigAddUnweights(builder, unweights_offset)
    feature_config.FilterbankConfigAddNumChannels(builder, c.mel_constants.n_channels)
    feature_config.FilterbankConfigAddChannelFrequencyStarts(builder, channel_freq_offset)
    feature_config.FilterbankConfigAddChannelWeightStarts(builder, channel_weight_offset)
    feature_config.FilterbankConfigAddChannelWidths(builder, channel_width_offset)
    fb_config_offset = feature_config.FilterbankConfigEnd(builder)

    # Build FeatureConfig
    feature_config.FeatureConfigStart(builder)
    feature_config.FeatureConfigAddWindowScalingBits(builder, c.window_scaling_bits)
    feature_config.FeatureConfigAddMelPostScalingBits(builder, c.mel_post_scaling_bits)
    feature_config.FeatureConfigAddHanningWindow(builder, hanning_window_offset)
    feature_config.FeatureConfigAddFftTwiddle(builder, fft_twiddle_offset)
    feature_config.FeatureConfigAddFbConfig(builder, fb_config_offset)
    feature_config_offset = feature_config.FeatureConfigEnd(builder)
    builder.Finish(feature_config_offset)
    buf = builder.Output()
    return buf


def apply_hanning(w, hanning, window_scaling_bits) -> NDArray[np.int16]:
    hanned = np.multiply(w, hanning, dtype=np.int32) >> window_scaling_bits
    hanned_clipped = np.clip(hanned, a_min=np.iinfo(np.int16).min, a_max=np.iinfo(np.int16).max)
    return hanned_clipped.astype(np.int16)


def energy(fft_vals):
    return np.power(fft_vals[0::2].astype(np.int32), 2).astype(np.uint32) + np.power(fft_vals[1::2].astype(np.int32), 2).astype(np.uint32)


def process_window(w, hanning, mel_constants: FilterBankConstants, fft_twiddle, window_scaling_bits, mel_post_scaling_bits, inference_mode=False):
    w = w.copy()
    hanned: NDArray[np.int16] = apply_hanning(w, hanning, window_scaling_bits)
    scaled_bits = shift_scale_up(hanned)
    fft = do_fft(hanned, fft_twiddle)
    rfft = fft[:len(fft)//2]

    fft_energy: NDArray[np.int32] = energy(rfft)
    fft_energy[:mel_constants.fft_start_index] = 0
    fft_energy[mel_constants.fft_end_index:] = 0
    mel_scaled: NDArray[np.uint64] = filter_bank(
        mel_constants,
        fft_energy
    )
    mel_sqrt: NDArray[np.uint64] = vec_sqrt64(mel_scaled)
    shift_scale_down(mel_sqrt, scaled_bits)
    mel_logged: NDArray[np.uint32] = vec_log32(mel_sqrt, 1 << mel_post_scaling_bits, 0)
    mel_logged[mel_logged == 65535] = 0
    # TODO: figure out scaling factors
    if inference_mode:
        # TODO: in the scope of the DCASE, this will never be hit
        mel_rescaled = mel_logged.astype(np.int8)
    else:
        mel_rescaled = mel_logged.astype(float)
    return mel_rescaled
