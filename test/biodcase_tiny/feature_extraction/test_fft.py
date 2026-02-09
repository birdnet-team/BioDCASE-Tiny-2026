import pytest
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt

# Import the functions to test
from biodcase_tiny.feature_extraction.nb_fft import gen_twiddle, dsps_cplx2real_sc16_ansi, do_fft


class TestFFT:

    @pytest.fixture
    def twiddle_factors(self):
        """Generate twiddle factors for various FFT sizes"""
        return {
            64: gen_twiddle(64),
            128: gen_twiddle(128),
            256: gen_twiddle(256),
            512: gen_twiddle(512),
            1024: gen_twiddle(1024)
        }

    def int16_to_complex(self, data):
        """Convert interleaved int16 array to complex array"""
        return np.array([complex(data[i], data[i+1]) for i in range(0, len(data), 2)])

    def complex_to_int16(self, data):
        """Convert complex array to interleaved int16 array"""
        result = np.zeros(len(data) * 2, dtype=np.int16)
        for i in range(len(data)):
            result[2*i] = np.int16(np.real(data[i]))
            result[2*i+1] = np.int16(np.imag(data[i]))
        return result

    def normalize_fft_output(self, data, n):
        """Normalize FFT output for comparison"""
        return data / n

    @pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
    def test_sine_wave(self, n, twiddle_factors):
        """Test FFT with a simple sine wave"""
        # Generate sine wave
        freq = 5  # Hz
        sample_rate = n  # Hz
        t = np.arange(n) / sample_rate
        signal = np.sin(2 * np.pi * freq * t)

        # Scale to int16 range
        signal_int16 = (signal * np.iinfo(np.int16).max).astype(np.int16)

        # Run our FFT
        our_fft = do_fft(signal_int16, twiddle_factors[n])

        # Run NumPy FFT
        numpy_fft = np.fft.fft(signal)

        # Convert our FFT result to complex for comparison
        our_fft_complex = self.int16_to_complex(our_fft)

        # Check peak frequencies match
        our_peak_idx = np.argmax(np.abs(our_fft_complex[1:n//2])) + 1
        numpy_peak_idx = np.argmax(np.abs(numpy_fft[1:n//2])) + 1

        #Plot for visual inspection (uncomment if needed)
        plt.figure(figsize=(12, 6))
        plt.subplot(311)
        plt.plot(signal_int16)
        plt.title('Signal')
        plt.subplot(312)
        plt.plot(np.abs(our_fft_complex[:n//2]))
        plt.title('Our FFT')
        plt.subplot(313)
        plt.plot(np.abs(numpy_fft[:n//2]))
        plt.title('NumPy FFT')
        plt.tight_layout()
        plt.show()

        assert numpy_peak_idx == freq
        assert our_peak_idx == freq

    @pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
    def test_impulse(self, n, twiddle_factors):
        """Test FFT with an impulse signal"""
        # Create impulse signal
        signal = np.zeros(n, dtype=np.int16)
        signal[0] = 32767  # Max int16 value

        # Run our FFT
        our_fft = do_fft(signal, twiddle_factors[n])

        # Run NumPy FFT
        numpy_fft = np.fft.fft(signal)

        # Convert our FFT result to complex for comparison
        our_fft_complex = self.int16_to_complex(our_fft)

        # For an impulse, all frequency bins should have approximately equal magnitude
        our_magnitudes = np.abs(our_fft_complex)
        numpy_magnitudes = np.abs(numpy_fft)

        # Check if magnitudes are approximately constant (allowing for fixed-point errors)
        assert np.std(our_magnitudes) / np.mean(our_magnitudes) < 0.1

        # Check if our FFT shape matches NumPy's FFT shape
        our_normalized = our_magnitudes / np.max(our_magnitudes)
        numpy_normalized = numpy_magnitudes / np.max(numpy_magnitudes)
        assert_allclose(our_normalized, numpy_normalized, rtol=0.2, atol=0.2)

    @pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
    def test_multiple_frequencies(self, n, twiddle_factors):
        """Test FFT with multiple frequency components"""
        # Generate signal with multiple frequencies
        t = np.arange(n) / n
        freqs = [5, 20, 50]  # Hz
        signal = np.zeros(n)
        for freq in freqs:
            if freq < n//2:  # Only include frequencies below Nyquist
                signal += np.sin(2 * np.pi * freq * t)

        # Scale to int16 range
        signal_int16 = (signal * 10000 / len(freqs)).astype(np.int16)

        # Run our FFT
        our_fft = do_fft(signal_int16, twiddle_factors[n])

        # Run NumPy FFT
        numpy_fft = np.fft.fft(signal_int16)

        # Convert our FFT result to complex for comparison
        our_fft_complex = self.int16_to_complex(our_fft)

        # Find peaks in both FFTs
        our_peaks = []
        numpy_peaks = []

        # Use a simple peak finding algorithm
        for i in range(1, n//2-1):
            if (np.abs(our_fft_complex[i]) > np.abs(our_fft_complex[i-1]) and
                np.abs(our_fft_complex[i]) > np.abs(our_fft_complex[i+1]) and
                np.abs(our_fft_complex[i]) > np.mean(np.abs(our_fft_complex[:n//2])) * 3):
                our_peaks.append(i)

            if (np.abs(numpy_fft[i]) > np.abs(numpy_fft[i-1]) and
                np.abs(numpy_fft[i]) > np.abs(numpy_fft[i+1]) and
                np.abs(numpy_fft[i]) > np.mean(np.abs(numpy_fft[:n//2])) * 3):
                numpy_peaks.append(i)

        # Check if we found the expected frequency peaks
        for freq in freqs:
            if freq < n//2:  # Only check frequencies below Nyquist
                assert freq in our_peaks or freq-1 in our_peaks or freq+1 in our_peaks
                assert freq in numpy_peaks or freq-1 in numpy_peaks or freq+1 in numpy_peaks

    @pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
    def test_real_to_complex_conversion(self, n, twiddle_factors):
        """Test the real-to-complex conversion function"""
        # Generate a real signal
        t = np.arange(n) / n
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

        # Scale to int16 range
        signal_int16 = (signal * 10000).astype(np.int16)

        # Run FFT
        fft_result = do_fft(signal_int16, twiddle_factors[n])

        # Create a copy for real-to-complex conversion
        real_result = fft_result.copy()

        # Apply real-to-complex conversion
        dsps_cplx2real_sc16_ansi(real_result, n, twiddle_factors[n])

        # Compare with NumPy's rfft
        numpy_rfft = np.fft.rfft(signal_int16)

        # Convert our result to complex for comparison
        our_rfft_complex = self.int16_to_complex(real_result[:n+2])  # Only first n/2+1 complex values are meaningful

        # Check if the magnitudes have similar shape
        our_magnitudes = np.abs(our_rfft_complex[:n//2+1])
        numpy_magnitudes = np.abs(numpy_rfft)

        # Normalize for comparison
        our_normalized = our_magnitudes / np.max(our_magnitudes)
        numpy_normalized = numpy_magnitudes / np.max(numpy_magnitudes)

        # Check if peaks occur at the same frequencies
        our_peaks = np.where(our_normalized > 0.3)[0]
        numpy_peaks = np.where(numpy_normalized > 0.3)[0]

        assert len(our_peaks) == len(numpy_peaks)
        for op, np_p in zip(sorted(our_peaks), sorted(numpy_peaks)):
            assert abs(op - np_p) <= 1  # Allow for 1-bin difference due to fixed-point precision

    @pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
    def test_noise_signal(self, n, twiddle_factors):
        """Test FFT with random noise"""
        # Generate random noise
        np.random.seed(42)  # For reproducibility
        signal = np.random.normal(0, 1, n)

        # Scale to int16 range
        signal_int16 = (signal * 10000).astype(np.int16)

        # Run our FFT
        fft_result = do_fft(signal_int16, twiddle_factors[n])

        # Run NumPy FFT
        numpy_fft = np.fft.fft(signal_int16)

        # Convert our FFT result to complex for comparison
        our_fft_complex = self.int16_to_complex(fft_result)

        # For noise, check if the overall spectral shape is similar
        our_spectrum = np.abs(our_fft_complex[:n//2])
        numpy_spectrum = np.abs(numpy_fft[:n//2])

        # Smooth spectra for comparison (moving average)
        window_size = max(3, n // 64)
        our_smooth = np.convolve(our_spectrum, np.ones(window_size)/window_size, mode='valid')
        numpy_smooth = np.convolve(numpy_spectrum, np.ones(window_size)/window_size, mode='valid')

        # Normalize for comparison
        our_normalized = our_smooth / np.mean(our_smooth)
        numpy_normalized = numpy_smooth / np.mean(numpy_smooth)

        # Check correlation between the two spectra
        correlation = np.corrcoef(our_normalized, numpy_normalized)[0, 1]
        assert correlation > 0.7  # High correlation expected
