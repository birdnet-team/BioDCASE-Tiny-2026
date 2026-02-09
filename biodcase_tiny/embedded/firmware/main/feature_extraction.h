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
#pragma once

#include <span>  // uncomment if we switch to C++20
#include <esp_err.h>
#include "feature_config_generated.h"
#include "esp_micro_profiler.h"


constexpr auto TAG = "bm";


/**
 *
 * @param in_audio: contains complex audio samples, where the real and imaginary parts are interleaved
 * @param hanning
 * @param window_scaling_bits
 * @return
 */
esp_err_t apply_hanning(
    std::span<int16_t> in_audio, const flatbuffers::Vector<int32_t> *hanning, uint8_t window_scaling_bits);

esp_err_t get_energy(std::span<int16_t> fft_vals, std::span<uint32_t> out);

esp_err_t filter_bank(
    std::span<uint32_t> in, std::span<uint64_t> out, const FeatureConfigs::FilterbankConfig &fb_config);

esp_err_t do_sqrt64(std::span<uint64_t> data, std::span<uint32_t> out);

esp_err_t do_log32(std::span<uint32_t> data, std::span<uint32_t> out, uint32_t out_scale);

esp_err_t rescale_to_int8(std::span<uint32_t> data, std::span<int8_t> out, int32_t data_min, int32_t data_max);

esp_err_t init_feature_extraction(const FeatureConfigs::FeatureConfig *fc);

esp_err_t extract_features(
    std::span<int16_t> in_audio,
    std::span<uint8_t> scratch_buffer,
    std::span<int8_t> out_features,
    const FeatureConfigs::FeatureConfig *fc,
    benchmark::MicroProfiler *profiler,
    int n_windows);