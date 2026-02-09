from dataclasses import dataclass, field
from itertools import product, chain

import librosa
import numpy as np
import pytest

# Import the functions to test
from biodcase_tiny.feature_extraction.nb_mel import (
    filter_bank,
    _init_filter_bank_weights,
    FILTER_BANK_ALIGNMENT,
    FILTER_BANK_CHANNEL_BLOCK_SIZE
)


@dataclass
class Signal:
    type: str
    sample_rate: int
    fft_size: int
    params: dict = field(default_factory=dict)


TEST_SIGNALS = list(
    chain(
        [
            Signal(type="white_noise", sample_rate=sr, fft_size=fft_size, params={"seed": seed})
            for fft_size, sr, seed in product([512, 1024], [16000, 44100], [42, 1337, 80085, 666, 999, 1234])
        ],
        [
            Signal(type="sine", sample_rate=sr, fft_size=fft_size, params={"freq": freq})
            for fft_size, sr, freq in product([512, 1024], [16000, 44100], [440, 800, 1000])
        ],
        [
            Signal(type="square", sample_rate=sr, fft_size=fft_size, params={"freq": freq})
            for fft_size, sr, freq in product([512, 1024], [16000, 44100], [440, 800, 1000])
        ]
    )
)



class TestFilterBank:

    def create_filter_bank_constants(self, fft_size, sample_rate, num_channels,
                                     lower_band_limit=20.0, upper_band_limit=4000.0):
        """Helper to create FilterBankConstants for testing"""
        spectrum_size = fft_size // 2 + 1
        return _init_filter_bank_weights(
            spectrum_size=spectrum_size,
            sample_rate=sample_rate,
            alignment=FILTER_BANK_ALIGNMENT,
            channel_block_size=FILTER_BANK_CHANNEL_BLOCK_SIZE,
            num_channels=num_channels,
            lower_band_limit=lower_band_limit,
            upper_band_limit=upper_band_limit
        )

    def create_librosa_mel_filterbank(self, fft_size, sample_rate, num_channels,
                                      lower_band_limit=20.0, upper_band_limit=4000.0):
        """Create a librosa mel filterbank for comparison"""
        return librosa.filters.mel(
            sr=sample_rate,
            n_fft=fft_size,
            n_mels=num_channels,
            fmin=lower_band_limit,
            fmax=upper_band_limit,
            htk=True,  # Use HTK formula which matches our implementation,
            norm=None,
        )

    def generate_spectrum(self, s: Signal):
        """Generate test spectrum data"""
        if s.type == "white_noise":
            # Generate white noise
            np.random.seed(s.params["seed"])  # For reproducibility
            signal = np.random.normal(0, 1, s.fft_size)
            signal /= np.max(signal)
        elif s.type == "sine":
            # Generate sine wave
            freq = s.params["freq"]  # Hz (A4 note)
            t = np.linspace(0, 1, s.sample_rate)
            signal = np.sin(2 * np.pi * freq * t)
        elif s.type == "square":
            freq = s.params["freq"]  # Hz (A4 note)
            t = np.linspace(0, 1, s.sample_rate)
            signal = np.sin(2 * np.pi * freq * t)
            signal[signal > 0] = 1
            signal[signal < 0] = -1
        else:
            raise ValueError(f"Unknown signal type: {s.type}")

        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(signal, n=s.fft_size))
        # Convert to energy/power spectrum and scale to uint32
        energy_spectrum = (spectrum ** 2).astype(np.uint32)
        return energy_spectrum

    def apply_librosa_mel(self, mel_basis, spectrum):
        """Apply librosa's mel filterbank to spectrum"""
        # Convert to float for librosa
        spectrum_float = spectrum.astype(np.float32)
        # Apply mel filterbank
        return np.dot(mel_basis, spectrum_float)

    @pytest.mark.parametrize("signal", TEST_SIGNALS)
    @pytest.mark.parametrize("num_channels", [20, 40])
    def test_filter_bank_shape(self, signal, num_channels):
        """Test that filter_bank produces the expected output shape"""
        spectrum = self.generate_spectrum(signal)
        fb_constants = self.create_filter_bank_constants(
            fft_size=signal.fft_size,
            sample_rate=signal.sample_rate,
            num_channels=num_channels
        )
        mel_output = filter_bank(fb_constants, spectrum)
        assert mel_output.shape == (num_channels,)

    @pytest.mark.parametrize("signal", TEST_SIGNALS)
    @pytest.mark.parametrize("num_channels", [20, 40])
    def test_filter_bank_vs_librosa(self, signal, num_channels):
        """Compare filter_bank output with librosa's mel filterbank"""
        spectrum = self.generate_spectrum(signal)

        fb_constants = self.create_filter_bank_constants(
            fft_size=signal.fft_size,
            sample_rate=signal.sample_rate,
            num_channels=num_channels
        )
        nb_mel = filter_bank(fb_constants, spectrum)

        librosa_mel = self.create_librosa_mel_filterbank(
            fft_size=signal.fft_size,
            sample_rate=signal.sample_rate,
            num_channels=num_channels
        )
        librosa_mel_output = self.apply_librosa_mel(librosa_mel, spectrum)

        # Normalize both outputs for comparison
        nb_mel_norm = nb_mel / np.max(nb_mel)
        librosa_mel_norm = librosa_mel_output / np.max(librosa_mel_output)
        correlation = np.corrcoef(nb_mel_norm, librosa_mel_norm)[0, 1]

        # most bins are more correlated than this, but first one actually can differ quite a bit.
        # this is not surprising as that's the smallest bin (fewer ops -> less space to average out)
        assert correlation > 0.99, f"Correlation too low: {correlation}"

    @pytest.mark.parametrize("fft_size", [1024, 2048])
    @pytest.mark.parametrize("sample_rate", [16000])
    @pytest.mark.parametrize("num_channels", [40])
    def test_filter_bank_frequency_response(self, fft_size, sample_rate, num_channels):
        """Test filter bank's frequency response with impulses (in spectrum, not signal)"""
        # Create filter bank constants
        fb_constants = self.create_filter_bank_constants(
            fft_size=fft_size,
            sample_rate=sample_rate,
            num_channels=num_channels
        )

        # Create librosa mel filterbank for comparison
        librosa_mel_basis = self.create_librosa_mel_filterbank(
            fft_size=fft_size,
            sample_rate=sample_rate,
            num_channels=num_channels
        )

        # Get spectrum size
        spectrum_size = fft_size // 2 + 1

        # Test with impulses at different frequency bins
        test_bins = np.linspace(fb_constants.fft_start_index, fb_constants.fft_end_index - 1, 10).astype(int)

        for bin_idx in test_bins:
            # Create impulse spectrum
            impulse_spectrum = np.zeros(spectrum_size, dtype=np.uint32)
            impulse_spectrum[bin_idx] = 1000000  # Large value to ensure good SNR

            nb_mel = filter_bank(fb_constants, impulse_spectrum)
            librosa_mel = self.apply_librosa_mel(librosa_mel_basis, impulse_spectrum)

            nb_peak = np.argmax(nb_mel)
            librosa_peak = np.argmax(librosa_mel)
            assert nb_peak == librosa_peak, f"Peak mismatch at bin {bin_idx}: ours={nb_peak}, librosa={librosa_peak}"

    @pytest.mark.parametrize("fft_size", [1024, 2048])
    @pytest.mark.parametrize("sample_rate", [16000])
    def test_filter_bank_channel_spacing(self, fft_size, sample_rate):
        """Test that filter bank channels are properly spaced in mel scale"""
        num_channels = 40

        # Create filter bank constants
        fb_constants = self.create_filter_bank_constants(
            fft_size=fft_size,
            sample_rate=sample_rate,
            num_channels=num_channels
        )

        # Test with a flat spectrum (all bins equal)
        spectrum_size = fft_size // 2 + 1
        flat_spectrum = np.ones(spectrum_size, dtype=np.uint32) * 10000

        # Apply our filter bank
        mel_output = filter_bank(fb_constants, flat_spectrum)

        # With a flat spectrum, the output should roughly follow the bandwidth of each filter
        # Check that we don't have any zeros
        assert np.all(mel_output > 0), "Some mel bands have zero response"

        # Check that the variation between adjacent bands is reasonable
        # (this is a heuristic check, values may need adjustment)
        ratios = mel_output[1:] / mel_output[:-1]
        assert np.all(ratios > 0.5) and np.all(ratios < 2.0), "Extreme variation between adjacent bands: {ratios}"

    @pytest.mark.parametrize("fft_size", [1024, 2048])
    @pytest.mark.parametrize("sample_rate", [16000])
    @pytest.mark.parametrize("num_channels", [40])
    def test_filter_bank_edge_cases(self, fft_size, sample_rate, num_channels):
        """Test filter bank with edge cases"""
        # Create filter bank constants
        fb_constants = self.create_filter_bank_constants(
            fft_size=fft_size,
            sample_rate=sample_rate,
            num_channels=num_channels
        )

        # Test with all zeros
        zero_spectrum = np.zeros(fft_size // 2, dtype=np.uint32)
        zero_output = filter_bank(fb_constants, zero_spectrum)
        assert np.all(zero_output == 0), "Zero input should produce zero output"

        # Test with very large values (check for overflow)
        large_spectrum = np.ones(fft_size // 2, dtype=np.uint32) * np.iinfo(np.uint32).max
        large_output = filter_bank(fb_constants, large_spectrum)
        assert np.all(large_output > 0), "Large input should produce non-zero output"
        assert np.all(np.isfinite(large_output)), "Output should be finite"
