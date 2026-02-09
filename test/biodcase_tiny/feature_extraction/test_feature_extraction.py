import numpy as np

from biodcase_tiny.feature_extraction import feature_config_generated as feature_config
from biodcase_tiny.feature_extraction.feature_extraction import make_constants, convert_constants, process_window


class TestFeatureExtraction:

    def test_constants(self):
        cs = make_constants(
            win_samples=1024, sample_rate=16000, window_scaling_bits=16,
            mel_n_channels=20, mel_low_hz=100, mel_high_hz=8000
        )
        buf = convert_constants(cs)
        fc = feature_config.FeatureConfig.GetRootAsFeatureConfig(buf, 0)
        mc: feature_config.FilterbankConfig = fc.FbConfig()
        assert cs.window_scaling_bits == fc.WindowScalingBits()
        np.testing.assert_array_equal(cs.fft_twiddle, fc.FftTwiddleAsNumpy())
        np.testing.assert_array_equal(cs.hanning_window, fc.HanningWindowAsNumpy())

        assert cs.mel_constants.fft_start_index == mc.FftStartIdx()
        assert cs.mel_constants.fft_end_index == mc.FftEndIdx()
        assert cs.mel_constants.n_channels == mc.NumChannels()
        np.testing.assert_array_equal(cs.mel_constants.ch_freq_starts, mc.ChannelFrequencyStartsAsNumpy())
        np.testing.assert_array_equal(cs.mel_constants.ch_widths, mc.ChannelWidthsAsNumpy())
        np.testing.assert_array_equal(cs.mel_constants.ch_weight_starts, mc.ChannelWeightStartsAsNumpy())
        np.testing.assert_array_equal(cs.mel_constants.weights, mc.WeightsAsNumpy())
        np.testing.assert_array_equal(cs.mel_constants.unweights, mc.UnweightsAsNumpy())

