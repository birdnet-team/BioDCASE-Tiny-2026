/* Copyright 2024 The TensorFlow Authors
   Copyright 2025 BirdNET team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This code was adapted from the benchmark utility of TFLM.
==============================================================================*/

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstring>
#include <initializer_list>
#include <memory>
#include <random>
#include <type_traits>
#include <span>
#include <cinttypes>


#include "esp_heap_caps.h"

#include "esp_micro_profiler.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"

#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "metrics.h"
#include "op_resolver.h"
#include "model.h"
#include "esp_micro_profiler.h"
#include "feature_config_generated.h"
#include "feature_config.h"
#include "feature_extraction.h"


namespace tflite {
namespace {

using Profiler = ::benchmark::MicroProfiler;

// Seed used for the random input. Input data shouldn't affect invocation
// timing so randomness isn't really needed.
constexpr uint32_t kRandomSeed = 0xFB;

constexpr size_t kTensorArenaSize = 6000000;
uint8_t* tensor_arena;


void SetRandomInput(const uint32_t random_seed,
                    tflite::MicroInterpreter& interpreter) {
  std::mt19937 eng(random_seed);
  std::uniform_int_distribution<uint32_t> dist(0, 255);

  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input_tensor(i);

    // Pre-populate input tensor with random values.
    int8_t* input_values = tflite::GetTensorData<int8_t>(input);
    for (size_t j = 0; j < input->bytes; ++j) {
      input_values[j] = dist(eng);
    }
  }
}


void SetRandomInput(const uint32_t random_seed, std::span<int16_t> audio_buf) {
  std::mt19937 eng(random_seed);
  std::uniform_int_distribution<int16_t> dist(INT16_MIN, INT16_MAX);
  for (size_t i = 0; i < audio_buf.size(); ++i) {
    audio_buf[i] = dist(eng);
  }
}

constexpr uint32_t kCrctabLen = 256;
uint32_t crctab[kCrctabLen];

void GenCRC32Table() {
  constexpr uint32_t kPolyN = 0xEDB88320;
  for (size_t index = 0; index < kCrctabLen; index++) {
    crctab[index] = index;
    for (int i = 0; i < 8; i++) {
      if (crctab[index] & 1) {
        crctab[index] = (crctab[index] >> 1) ^ kPolyN;
      } else {
        crctab[index] >>= 1;
      }
    }
  }
}

uint32_t ComputeCRC32(const uint8_t* data, const size_t data_length) {
  uint32_t crc32 = ~0U;

  for (size_t i = 0; i < data_length; i++) {
    // crctab is an array of 256 32-bit constants
    const uint32_t index = (crc32 ^ data[i]) & (kCrctabLen - 1);
    crc32 = (crc32 >> 8) ^ crctab[index];
  }

  // invert all bits of result
  crc32 ^= ~0U;
  return crc32;
}

void ShowOutputCRC32(std::span<int8_t> out_features) {
  GenCRC32Table();
  uint32_t crc32_value = ComputeCRC32(
    reinterpret_cast<uint8_t*>(out_features.data()), out_features.size());
  MicroPrintf("Output Features CRC32: 0x%X", crc32_value);
}

void ShowOutputCRC32(tflite::MicroInterpreter* interpreter) {
  GenCRC32Table();
  for (size_t i = 0; i < interpreter->outputs_size(); ++i) {
    TfLiteTensor* output = interpreter->output_tensor(i);
    uint8_t* output_values = tflite::GetTensorData<uint8_t>(output);
    uint32_t crc32_value = ComputeCRC32(output_values, output->bytes);
    MicroPrintf("Output CRC32: 0x%X", crc32_value);
  }
}

void ShowInputCRC32(std::span<int16_t> audio_in) {
  GenCRC32Table();
  uint32_t crc32_value = ComputeCRC32(
    reinterpret_cast<uint8_t*>(audio_in.data()), audio_in.size() * sizeof(int16_t));
  MicroPrintf("Audio Input CRC32: 0x%X", crc32_value);
}

void ShowInputCRC32(tflite::MicroInterpreter* interpreter) {
  GenCRC32Table();
  for (size_t i = 0; i < interpreter->inputs_size(); ++i) {
    TfLiteTensor* input = interpreter->input_tensor(i);
    uint8_t* input_values = tflite::GetTensorData<uint8_t>(input);
    uint32_t crc32_value = ComputeCRC32(input_values, input->bytes);
    MicroPrintf("Input CRC32: 0x%X", crc32_value);
  }
}

[[nodiscard]] int Benchmark(const uint8_t* model_data, const uint8_t* feature_extractor_data) {
  static Profiler profiler;
  uint32_t seed = kRandomSeed;
  TfLiteStatus status;

  tensor_arena = reinterpret_cast<uint8_t *>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM));

  uint32_t event_handle = profiler.BeginEvent("GetFeatureConfig");
  const FeatureConfigs::FeatureConfig* fc = FeatureConfigs::GetFeatureConfig(feature_extractor_data);
  profiler.EndEvent(event_handle);

  auto audio_n_samples = fc->hanning_window()->size();
  MicroPrintf("N audio samples: %d", audio_n_samples);
  auto audio_win_arr = (int16_t*) heap_caps_aligned_alloc(16, audio_n_samples * sizeof(int16_t) * 2, MALLOC_CAP_DEFAULT);
  std::span audio_win = {audio_win_arr, audio_n_samples * 2};  // *2 because complex numbers assumed
  std::span scratch_buffer = {
    reinterpret_cast<uint8_t*>(audio_win.data()), audio_win.size() * sizeof(int16_t)};
  std::span out_buffer = {
    reinterpret_cast<int8_t*>(audio_win.data()), audio_win.size() * sizeof(int16_t)};
  init_feature_extraction(fc);

  event_handle = profiler.BeginEvent("tflite::GetModel");
  const tflite::Model* model = tflite::GetModel(model_data);
  profiler.EndEvent(event_handle);

  event_handle = profiler.BeginEvent("tflite::CreateOpResolver");
  TflmOpResolver op_resolver;
  status = CreateOpResolver(op_resolver);
  if (status != kTfLiteOk) {
    MicroPrintf("tflite::CreateOpResolver failed");
    heap_caps_free(audio_win_arr);
    return -1;  // can't goto error here as interpreter must be initialized first
  }
  profiler.EndEvent(event_handle);

  event_handle = profiler.BeginEvent("tflite::MicroInterpreter instantiation");
  tflite::RecordingMicroInterpreter interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize, nullptr,
      &profiler);
  profiler.EndEvent(event_handle);
  //
  event_handle =
      profiler.BeginEvent("tflite::MicroInterpreter::AllocateTensors");
  status = interpreter.AllocateTensors();
  if (status != kTfLiteOk) {
    MicroPrintf("tflite::MicroInterpreter::AllocateTensors failed");
    goto error;
  }
  profiler.EndEvent(event_handle);
  profiler.LogTicksPerTagCsv();
  profiler.ClearEvents();

  MicroPrintf("");  // null MicroPrintf serves as a newline.

  // For streaming models, the interpreter will return kTfLiteAbort if the
  // model does not yet have enough data to make an inference. As such, we
  // need to invoke the interpreter multiple times until we either receive an
  // error or kTfLiteOk. This loop also works for non-streaming models, as
  // they'll just return kTfLiteOk after the first invocation.
  while (true) {
    SetRandomInput(seed, audio_win);
    SetRandomInput(seed, interpreter);

    ShowInputCRC32(audio_win);
    auto model_input = interpreter.input(0);
    auto n_windows = model_input->dims->data[1];
    extract_features(audio_win, scratch_buffer, out_buffer, fc, &profiler, n_windows);
    MicroPrintf("");
    auto total_ticks = profiler.LogTicksPerTagCsv();
    MicroPrintf("");
    profiler.ClearEvents();
    MicroPrintf("Total * n_windows = %" PRId64, total_ticks * n_windows);

    ShowOutputCRC32(out_buffer);
    seed++;

    ShowInputCRC32(&interpreter);
    MicroPrintf("");  // null MicroPrintf serves as a newline.

    status = interpreter.Invoke();
    if ((status != kTfLiteOk) && (static_cast<int>(status) != kTfLiteAbort)) {
      MicroPrintf("Model interpreter invocation failed: %d\n", status);
      goto error;
    }

    profiler.Log();
    MicroPrintf("");  // null MicroPrintf serves as a newline.
    profiler.LogTicksPerTagCsv();
    MicroPrintf("");  // null MicroPrintf serves as a newline.
    profiler.ClearEvents();

    ShowOutputCRC32(&interpreter);
    MicroPrintf("");  // null MicroPrintf serves as a newline.

    if (status == kTfLiteOk) {
      break;
    }
  }
  interpreter.GetMicroAllocator().PrintAllocations();
  return 0;

error:
  heap_caps_free(audio_win_arr);
  return -1;
}
}  // namespace
}  // namespace tflite


extern "C" void app_main() {
  MicroPrintf("\nConfigured arena size = %d\n", tflite::kTensorArenaSize);
  auto res = tflite::Benchmark(g_model, feature_config);
  MicroPrintf("Result=%d", res);
}

