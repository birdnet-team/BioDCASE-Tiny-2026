import pytest
from biodcase_tiny.feature_extraction.nb_log32 import (log32, vec_log32)
import numpy as np


class TestLog32:

    @pytest.mark.parametrize(
        "out_scale, rel_err",
        [
            (1, 0.45),
            (10, 0.03),
            (100, 0.005),
            (1_000, 0.0004),
            (10_000, 1.7e-4),
            (100_000, 1.7e-4),
            (1_000_000, 1.7e-4)
        ]
    )
    def test_log32(self, out_scale, rel_err):
        arr = np.logspace(1, 31, 100, base=2, dtype=np.uint32)
        logs_i = np.array([log32(x, out_scale) for x in arr], dtype=np.uint32)
        logs_f = np.log(arr) * out_scale
        rel_err_arr = (np.abs(logs_f - logs_i) / logs_f)[logs_f>0]
        assert np.nanmax(rel_err_arr) < rel_err, np.nanargmax(rel_err_arr)

    @pytest.mark.parametrize(
        "correction_bits, out_scale, rel_err", [
        # replicate previous test (to a point, as vec_log32 returns int16 instead)
            (0, 1, 0.45),
            (0, 10, 0.03),
            (0, 100, 0.005),
            (0, 1_000, 0.0004),
            # (0, 10_000, 1.7e-4),  loge(2^31) * 10000 > max(uint16), so here vec_log32 will clip

            (24,         1, 0.03 ),     # narrow representation range, most logs are same val
            (16,         1, 0.04 ),
            ( 8,         1, 0.06 ),
            ( 2,    10_000, 0.7  ),     # logs_i almsot always max(uint16)
            ( 8,    10_000, 1.5e-4),    # logs_i almsot always max(uint16)
            (24,       100, 0.002),     # narrow representation range, most logsi are same val
            (16,    10_000, 0.8  ),     # logs_i is always max(uint16)
            ( 8, 1_000_000, 1    ),     # logs_i is always max(uint16)

        ])
    def test_vec_log32(self, correction_bits, out_scale, rel_err):
        arr = np.logspace(1, 31 - correction_bits, 100, base=2, dtype=np.uint32)
        logs_i = vec_log32(arr, np.uint32(out_scale), np.uint8(correction_bits))
        logs_f = np.log(arr << correction_bits) * out_scale
        rel_err_arr = (np.abs(logs_f - logs_i) / logs_f)[arr > 1]  # 1: is to avoid the first nan, coming from an expected div by zero
        assert np.max(rel_err_arr) < rel_err

