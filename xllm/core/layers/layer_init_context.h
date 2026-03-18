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

#include "framework/model_context.h"

namespace xllm {
namespace layer {

struct LayerInitContext {
  // Core parameter (required by all layers)
  const ModelContext& model_context;

  // Layer index (needed by most layers)
  int32_t layer_id = -1;

  // Total number of layers (useful for determining first/last layer, etc.)
  int32_t total_layers = -1;

  // Backend identifier ("npu", "cuda", "mlu", "musa", "ilu", etc.)
  std::string backend;

  // Extensible property bag for backend/model-specific construction parameters,
  // avoiding interface changes for each new parameter
  std::unordered_map<std::string, std::any> extra_params;

  // Type-safe accessor with default value
  template <typename T>
  T get_extra(const std::string& key, const T& default_value = T{}) const {
    auto it = extra_params.find(key);
    if (it != extra_params.end()) {
      return std::any_cast<T>(it->second);
    }
    return default_value;
  }

  template <typename T>
  void set_extra(const std::string& key, T&& value) {
    extra_params[key] = std::forward<T>(value);
  }

  // Convenience shortcuts delegating to ModelContext
  const ModelArgs& model_args() const { return model_context.get_model_args(); }

  const ParallelArgs& parallel_args() const {
    return model_context.get_parallel_args();
  }

  const QuantArgs& quant_args() const { return model_context.get_quant_args(); }

  torch::TensorOptions tensor_options() const {
    return model_context.get_tensor_options();
  }
};

}  // namespace layer
}  // namespace xllm
