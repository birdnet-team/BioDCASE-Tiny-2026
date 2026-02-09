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

This file contains code derived from TFLM. We modified the micro profiler code
to make use of the esp timing functions.
==============================================================================*/
#include "esp_micro_profiler.h"
#include "esp_timer.h"

#include <cinttypes>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_log.h"


namespace benchmark {
uint32_t MicroProfiler::BeginEvent(const char* tag) {
  if (num_events_ == kMaxEvents) {
    MicroPrintf(
        "MicroProfiler errored out because total number of events exceeded the "
        "maximum of %d.",
        kMaxEvents);
    TFLITE_ASSERT_FALSE;
  }

  tags_[num_events_] = tag;
  start_ticks_[num_events_] = esp_timer_get_time();
  end_ticks_[num_events_] = start_ticks_[num_events_] - 1;
  return num_events_++;
}

void MicroProfiler::EndEvent(uint32_t event_handle) {
  TFLITE_DCHECK(event_handle < kMaxEvents);
  end_ticks_[event_handle] = esp_timer_get_time();
}

int64_t MicroProfiler::GetTotalTicks() const {
  int64_t ticks = 0;
  for (int i = 0; i < num_events_; ++i) {
    ticks += end_ticks_[i] - start_ticks_[i];
  }
  return ticks;
}

void MicroProfiler::Log() const {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  for (int i = 0; i < num_events_; ++i) {
    int64_t ticks = end_ticks_[i] - start_ticks_[i];
    MicroPrintf("%s took %" PRId64 " microseconds.", tags_[i], ticks);
  }
#endif
}

void MicroProfiler::LogCsv() const {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  MicroPrintf("\"Event\",\"Tag\",\"microseconds\"");
  for (int i = 0; i < num_events_; ++i) {
#if defined(HEXAGON) || defined(CMSIS_NN)
    int64_t ticks = end_ticks_[i] - start_ticks_[i];
    MicroPrintf("%d,%s,%" PRId64, i, tags_[i], ticks);
#else
    int64_t ticks = end_ticks_[i] - start_ticks_[i];
    MicroPrintf("%d,%s,%" PRId64, i, tags_[i], ticks);
#endif
  }
#endif
}

int64_t MicroProfiler::LogTicksPerTagCsv() {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  MicroPrintf(
      "\"Unique Tag\",\"Total microseconds across all events with that tag.\"");
  int64_t total_ticks = 0;
  for (int i = 0; i < num_events_; ++i) {
    int64_t ticks = end_ticks_[i] - start_ticks_[i];
    TFLITE_DCHECK(tags_[i] != nullptr);
    int position = FindExistingOrNextPosition(tags_[i]);
    TFLITE_DCHECK(position >= 0);
    total_ticks_per_tag_[position].tag = tags_[i];
    total_ticks_per_tag_[position].ticks =
        total_ticks_per_tag_[position].ticks + ticks;
    total_ticks += ticks;
  }

  for (int i = 0; i < num_events_; ++i) {
    TicksPerTag each_tag_entry = total_ticks_per_tag_[i];
    if (each_tag_entry.tag == nullptr) {
      break;
    }
    MicroPrintf("%s, %" PRId64, each_tag_entry.tag, each_tag_entry.ticks);
  }
  MicroPrintf("\"total number of microseconds\", %" PRId64, total_ticks);
#endif
  return total_ticks;
}

// This method finds a particular array element in the total_ticks_per_tag array
// with the matching tag_name passed in the method. If it can find a
// matching array element that has the same tag_name, then it will return the
// position of the matching element. But if it unable to find a matching element
// with the given tag_name, it will return the next available empty position
// from the array.
int MicroProfiler::FindExistingOrNextPosition(const char* tag_name) {
  int pos = 0;
  for (; pos < num_events_; pos++) {
    TicksPerTag each_tag_entry = total_ticks_per_tag_[pos];
    if (each_tag_entry.tag == nullptr ||
        strcmp(each_tag_entry.tag, tag_name) == 0) {
      return pos;
    }
  }
  return pos < num_events_ ? pos : -1;
}

void MicroProfiler::ClearEvents() {
  for (int i = 0; i < num_events_; i++) {
    total_ticks_per_tag_[i].tag = nullptr;
    total_ticks_per_tag_[i].ticks = 0;
  }
  num_events_ = 0;
}

}  // namespace benchmark
