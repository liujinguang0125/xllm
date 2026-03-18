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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "i_decoder_layer.h"
#include "layer_init_context.h"

namespace xllm {
namespace layer {

using DecoderLayerFactory = std::function<std::shared_ptr<IDecoderLayer>(
    const LayerInitContext& init_ctx)>;

class LayerRegistry {
 public:
  static LayerRegistry& instance() {
    static LayerRegistry inst;
    return inst;
  }

  void register_decoder_layer(const std::string& model_type,
                              const std::string& backend,
                              DecoderLayerFactory factory) {
    auto key = make_key(model_type, backend);
    factories_[key] = std::move(factory);
  }

  DecoderLayerFactory get_factory(const std::string& model_type,
                                  const std::string& backend) const {
    // Priority: exact match > fallback to "common"
    auto key = make_key(model_type, backend);
    auto it = factories_.find(key);
    if (it == factories_.end()) {
      auto fallback_key = make_key(model_type, "common");
      it = factories_.find(fallback_key);
    }
    CHECK(it != factories_.end())
        << "No decoder layer registered for model=" << model_type
        << " backend=" << backend << " (also tried fallback to 'common')";
    return it->second;
  }

  bool has_factory(const std::string& model_type,
                   const std::string& backend) const {
    return factories_.count(make_key(model_type, backend)) > 0 ||
           factories_.count(make_key(model_type, "common")) > 0;
  }

 private:
  static std::string make_key(const std::string& model_type,
                              const std::string& backend) {
    return model_type + "::" + backend;
  }

  std::unordered_map<std::string, DecoderLayerFactory> factories_;
};

// Registration macro
#define REGISTER_DECODER_LAYER(model_type, backend, factory_fn)        \
  static const bool model_type##_##backend##_layer_registered = []() { \
    ::xllm::layer::LayerRegistry::instance().register_decoder_layer(   \
        #model_type, #backend, factory_fn);                            \
    return true;                                                       \
  }()

}  // namespace layer
}  // namespace xllm
