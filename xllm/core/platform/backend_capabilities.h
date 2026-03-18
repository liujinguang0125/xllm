/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <any>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace xllm {

struct BackendCapabilities {
  // ====== Common properties (all backends) ======
  std::string backend_name;  // "cuda", "npu", "mlu", "musa", "ilu"
  int32_t device_count = 0;
  int64_t total_memory = 0;  // bytes
  int64_t free_memory = 0;
  // ====== Compute properties ======
  int32_t compute_unit_count = 0;  // SM count (CUDA), AI Core count (NPU),
                                   // MLU Core count (MLU), etc.
  std::string chip_model;          // "A100", "H100", "Ascend910B", "MLU370"
  int32_t compute_capability_major = 0;
  int32_t compute_capability_minor = 0;

  // ====== Feature flags ======
  // Queryable set of supported features, e.g.:
  //   "flash_attention", "paged_attention", "fp8", "int8",
  //   "fused_moe", "pdl", "graph_capture", "vmm", ...
  std::unordered_set<std::string> supported_features;
  bool supports(const std::string& feature) const {
    return supported_features.count(feature) > 0;
  }

  // ====== Extensible backend-specific properties ======
  std::unordered_map<std::string, std::any> properties;

  template <typename T>
  T get(const std::string& key, const T& default_value = T{}) const {
    auto it = properties.find(key);
    if (it != properties.end()) {
      return std::any_cast<T>(it->second);
    }
    return default_value;
  }

  template <typename T>
  void set(const std::string& key, T&& value) {
    properties[key] = std::forward<T>(value);
  }
};

}  // namespace xllm
