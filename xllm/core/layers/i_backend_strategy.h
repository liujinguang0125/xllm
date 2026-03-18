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

#include "layer_forward_context.h"

namespace xllm {
namespace layer {

class IBackendStrategy {
 public:
  virtual ~IBackendStrategy() = default;

  // ====== Construction-phase callbacks ======

  // Initialize accompanying components (Embedding, Norm, etc.)
  virtual void init_components(const ModelContext& context) = 0;

  // Called before each layer is created to inject backend-specific params
  // into LayerInitContext
  virtual void populate_layer_init_context(LayerInitContext& init_ctx,
                                           int layer_id) {}

  // ====== Forward-phase callbacks ======

  // Embedding lookup
  virtual torch::Tensor embed(torch::Tensor tokens) = 0;

  // Called before the layer loop to prepare shared context (cos/sin,
  // attn_mask, etc.)
  virtual void prepare_forward(torch::Tensor& h,
                               torch::Tensor& positions,
                               const ModelInputParams& input_params,
                               LayerForwardContext& ctx_template) = 0;

  // Per-layer pre-forward callback (e.g. set async event)
  virtual void before_layer(int layer_id,
                            const ModelInputParams& input_params,
                            LayerForwardContext& ctx) {}

  // Per-layer post-forward callback
  virtual void after_layer(int layer_id, LayerForwardContext& ctx) {}

  // Final norm processing
  virtual torch::Tensor finalize(torch::Tensor& h,
                                 std::optional<torch::Tensor>& residual) = 0;

  // ====== Weight lifecycle (delegates to embedding, norm, etc.) ======

  virtual void load_state_dict_components(const StateDict& state_dict) = 0;
  virtual void merge_loaded_weights_components() {}
  virtual void free_weights_components() {}
  virtual void reload_weights_components() {}
};

}  // namespace layer
}  // namespace xllm