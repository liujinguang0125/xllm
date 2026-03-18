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

namespace xllm {

template <typename BackendStrategyType>
class UnifiedLlmModelImplBase : public torch::nn::Module {
 public:
  UnifiedLlmModelImplBase(const std::string& model_type,
                          const ModelContext& context) {
    auto backend_name = context.get_backend();  // "npu" / "cuda" / "mlu"
    auto factory =
        layer::LayerRegistry::instance().get_factory(model_type, backend_name);

    strategy_ = std::make_unique<BackendStrategyType>();
    strategy_->init_components(context);

    auto model_args = context.get_model_args();
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      // 构造 LayerInitContext，传给工厂
      layer::LayerInitContext init_ctx{
          .model_context = context,
          .layer_id = i,
          .total_layers = model_args.n_layers(),
          .backend = backend_name,
      };
      // 后端策略可在此注入 extra 参数
      strategy_->populate_layer_init_context(init_ctx, i);
      layers_.push_back(factory(init_ctx));
    }
  }

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    auto h = strategy_->embed(tokens);

    // 构造 forward 上下文模板
    layer::LayerForwardContext ctx_template{
        .x = h,
        .kv_cache = kv_caches[0],
        .input_params = input_params,
    };
    strategy_->prepare_forward(h, positions, input_params, ctx_template);

    for (size_t i = 0; i < layers_.size(); i++) {
      ctx_template.kv_cache = kv_caches[i];
      strategy_->before_layer(i, input_params, ctx_template);
      h = layers_[i]->forward(ctx_template);
      strategy_->after_layer(i, ctx_template);
    }

    return ModelOutput(strategy_->finalize(h, ctx_template.residual));
  }

  void load_state_dict(const StateDict& state_dict) {
    strategy_->load_state_dict_components(state_dict);
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

 protected:
  std::unique_ptr<BackendStrategyType> strategy_;
  std::vector<std::shared_ptr<layer::IDecoderLayer>> layers_;
};

}  // namespace xllm
