/* Copyright 2020 The TensorFlow Authors.
   Copyright 2025 BirdNET team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains code derived from TFLM, specifically the signal library.
Here we basically extract the core functions, without the tensorflow wrapper.
The reason for that was to avoid the root python project having a dependency
on the tflite signal python package, which requires compilation, not ideal in
the context of a DCASE contest where participants might not be versed
in the C++ stack.
==============================================================================*/
#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <esp_log.h>
#include <dsps_fft2r.h>

#include "feature_extraction.h"
#include "signal/src/filter_bank.h"
#include "signal/src/fft_auto_scale.h"
#include "signal/src/log.h"
#include "signal/src/square_root.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "dsp_common.h"


/**
 *
 * @param in_audio: contains complex audio samples, where the real and imaginary parts are interleaved
 * @param hanning
 * @param window_scaling_bits
 * @return
 */
esp_err_t apply_hanning(std::span<int16_t> in_audio, const flatbuffers::Vector<int16_t> *hanning, uint8_t window_scaling_bits) {
    int32_t h = 0;
    for (int i = 0; i < hanning->size(); i++) {
        h = std::clamp<int32_t>(
            (in_audio[i*2] * (*hanning)[i]) >> window_scaling_bits, INT16_MIN, INT16_MAX);
        in_audio[i*2] = static_cast<int16_t>(h);
    }
    return ESP_OK;
}

esp_err_t shift_scale_up(std::span<int16_t> arr, std::span<int16_t> out, int &scale_bits) {
  scale_bits = tflite::tflm_signal::FftAutoScale(arr.data(), arr.size(), out.data());
  return ESP_OK;
}

esp_err_t shift_scale_down(std::span<uint32_t> arr, std::span<uint32_t> out, int scale_bits) {
  for (int i=0; i < arr.size(); i++) {
      out[i] = arr[i] >> scale_bits;
  }
  return ESP_OK;
}

esp_err_t get_energy(std::span<int16_t> fft_vals, std::span<uint32_t> out) {
    for (int i = 0; i < out.size(); i++) {
        out[i] = static_cast<uint32_t>(fft_vals[i*2]) * fft_vals[i*2] + static_cast<uint32_t>(fft_vals[i*2 + 1]) * fft_vals[i*2 + 1];
    }
    return ESP_OK;
}

esp_err_t filter_bank(std::span<uint32_t> in, std::span<uint64_t> out,
    const FeatureConfigs::FilterbankConfig *fb_config) {
    tflite::tflm_signal::FilterbankConfig tflite_fb_config {
        .num_channels = fb_config->num_channels(),
        .channel_frequency_starts = fb_config->channel_frequency_starts()->data(),
        .channel_weight_starts =  fb_config->channel_weight_starts()->data(),
        .channel_widths =  fb_config->channel_widths()->data(),
        .weights = fb_config->weights()->data(),
        .unweights = fb_config->unweights()->data(),
        .output_scale = 1,
        .input_correction_bits = 0
    };
    FilterbankAccumulateChannels(&tflite_fb_config, in.data(), out.data());
    return ESP_OK;
}

esp_err_t do_sqrt64(std::span<uint64_t> data, std::span<uint32_t> out) {
    for (int i = 0; i < data.size(); i++) {
        out[i] = tflite::tflm_signal::Sqrt64(data[i]);
    }
    return ESP_OK;
}

esp_err_t do_log32(std::span<uint32_t> data, std::span<uint32_t> out, uint32_t out_scale) {
    for (int i = 0; i < data.size(); i++) {
        out[i] = tflite::tflm_signal::Log32(data[i], out_scale);
    }
    return ESP_OK;
}

esp_err_t rescale_to_int8(std::span<uint32_t> data, std::span<int8_t> out, int32_t data_min, int32_t data_max) {
    constexpr int32_t output_range = INT8_MIN - INT8_MAX;
    uint32_t data_range = data_max - data_min;
    uint32_t scale_down = std::min<uint32_t>(data_range / output_range, 1);
    for (size_t i = 0; i < data.size(); i++) {
        uint32_t positive = std::clamp<uint32_t>(data[i] - data_min, 0, data_range);
        uint32_t scaled = (positive + scale_down / 2) / scale_down; // Add scale_down/2 to round
        out[i] = static_cast<int8_t>(scaled - 128);
    }
    return ESP_OK;
}


esp_err_t init_feature_extraction(const FeatureConfigs::FeatureConfig *fc) {
    auto * twiddle = new int16_t[fc->fft_twiddle()->size()];
    std::memcpy(twiddle, fc->fft_twiddle()->data(), fc->fft_twiddle()->size() * sizeof(int16_t));
    auto ret = dsps_fft2r_init_sc16(twiddle, fc->fft_twiddle()->size());
    if (fc->fft_twiddle()->size() > CONFIG_DSP_MAX_FFT_SIZE) {
        ESP_LOGE(TAG, "fft twiddle table too big (fft window too big): %d", static_cast<int>(fc->fft_twiddle()->size()));
        return ESP_ERR_DSP_PARAM_OUTOFRANGE;
    }
    return ret;
}


esp_err_t extract_features(
    std::span<int16_t> in_audio,
    std::span<uint8_t> scratch_buffer,
    std::span<int8_t> out_features,
    const FeatureConfigs::FeatureConfig *fc,
    benchmark::MicroProfiler *profiler,
    int n_windows
    ) {
    size_t n_samples = in_audio.size() / 2;
    size_t n_rfft = n_samples / 2;

    if (!dsps_fft2r_sc16_initialized) {
        ESP_LOGE(TAG, "FFT not initialized");
        return ESP_FAIL;
    }
    if (!dsp_is_power_of_two(n_samples)) {
        ESP_LOGE(TAG, "Audio input should have power of two length");
        return ESP_ERR_DSP_INVALID_LENGTH;
    }
    if (n_samples != fc->hanning_window()->size()) {
        ESP_LOGE(TAG, "Audio n samples and hanning window size must be the same (%d != %lu)",
            n_samples, fc->hanning_window()->size());
        return ESP_FAIL;
    }

    // Hanning
    auto event_handle = profiler->BeginEvent("apply_hanning");
    apply_hanning(in_audio, fc->hanning_window(), fc->window_scaling_bits());
    profiler->EndEvent(event_handle);

    // Scale up
    int scale_bits = 0;
    event_handle = profiler->BeginEvent("shift_up");
    auto scaled_audio = in_audio;  // in place
    shift_scale_up(in_audio, scaled_audio, scale_bits);
    profiler->EndEvent(event_handle);

    // FFT
    event_handle = profiler->BeginEvent("FFT");
    dsps_fft2r_sc16_aes3(scaled_audio.data(), n_rfft);
    dsps_bit_rev_sc16_ansi(scaled_audio.data(), n_rfft);
    dsps_cplx2real_sc16_ansi(scaled_audio.data(), n_rfft);
    profiler->EndEvent(event_handle);

    // Energy
    event_handle = profiler->BeginEvent("Energy");
    auto fft_range = scaled_audio.subspan(
        fc->fb_config()->fft_start_idx() * 2,
        (fc->fb_config()->fft_end_idx() - fc->fb_config()->fft_start_idx() + 1) * 2);
    auto energy = std::span(reinterpret_cast<uint32_t*>(scratch_buffer.data()), fft_range.size() / 2);
    get_energy(fft_range, energy);
    profiler->EndEvent(event_handle);

    // Mel
    event_handle = profiler->BeginEvent("Mel");
    auto fb_res = std::span(reinterpret_cast<uint64_t*>(scratch_buffer.data()), fc->fb_config()->num_channels() + 1);
    filter_bank(energy, fb_res, fc->fb_config());
    fb_res = fb_res.subspan(1, fc->fb_config()->num_channels());  // discard first value of filter bank
    profiler->EndEvent(event_handle);

    event_handle = profiler->BeginEvent("sqrt(mel)");
    auto sqrt_res = std::span(reinterpret_cast<uint32_t*>(scratch_buffer.data()), fb_res.size());
    do_sqrt64(fb_res, sqrt_res);
    profiler->EndEvent(event_handle);

    event_handle = profiler->BeginEvent("shift_down");
    auto scaled_down = sqrt_res;  // in place
    shift_scale_down(sqrt_res, scaled_down, scale_bits);
    profiler->EndEvent(event_handle);

    event_handle = profiler->BeginEvent("log(sqrt(mel))");
    auto log_res = scaled_down;  // in place
    do_log32(sqrt_res, log_res, 1 << fc->mel_post_scaling_bits());
    profiler->EndEvent(event_handle);

    // int8 Rescale
    event_handle = profiler->BeginEvent("int8_rescale");
    rescale_to_int8(log_res, out_features, fc->mel_range_min(), fc->mel_range_max());
    profiler->EndEvent(event_handle);
    return ESP_OK;
}