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

#include <torch/torch.h>

#include <atomic>
#include <memory>
#include <optional>

#include "common/attention_metadata.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm {
namespace layer {

struct LayerForwardContext {
  // Core fields required by all backends
  torch::Tensor& x;
  KVCache& kv_cache;
  const ModelInputParams& input_params;

  // Fields used by common backends (CUDA, MLU, etc.)
  std::optional<torch::Tensor> residual;
  torch::Tensor positions;
  std::shared_ptr<AttentionMetadata> attn_metadata;

  // Fields used by NPU backend
  std::optional<torch::Tensor> cos_pos;
  std::optional<torch::Tensor> sin_pos;
  std::optional<torch::Tensor> attn_mask;

  // Async control (required by NPU, ignored by other backends)
  void* async_event = nullptr;
  std::atomic<bool>* event_flag = nullptr;
  int layer_id = 0;
};

}  // namespace layer
}  // namespace xllm
