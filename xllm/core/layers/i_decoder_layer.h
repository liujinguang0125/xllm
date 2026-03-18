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

#include "framework/state_dict/state_dict.h"
#include "layer_forward_context.h"

namespace xllm {
namespace layer {

class IDecoderLayer : public torch::nn::Module {
 public:
  virtual ~IDecoderLayer() = default;

  // Unified forward entry point
  virtual torch::Tensor forward(LayerForwardContext& ctx) = 0;

  // Weight loading (must be implemented)
  virtual void load_state_dict(const StateDict& state_dict) = 0;

  // Weight lifecycle management (default no-ops, override for backends like
  // NPU)
  virtual void verify_loaded_weights() const {}
  virtual void verify_loaded_weights(const std::string& prefix) const {}
  virtual void merge_loaded_weights() {}
  virtual void free_weights() {}
  virtual void reload_weights() {}
  virtual void reload_weights_from_device() {}
  virtual void merge_and_move_pinned_host() {}

  // Layer initialization callback (optional)
  virtual int64_t init_layer() { return 0; }
};

}  // namespace layer
}  // namespace xllm
