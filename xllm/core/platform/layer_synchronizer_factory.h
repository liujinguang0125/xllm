#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "ascend/atb_device_context.h"
#include "device_context.h"
#include "device_type.h"

namespace xllm {
namespace platform {

class LayerSynchronizerFactory {
 public:
  LayerSynchronizerFactory() = delete;
  ~LayerSynchronizerFactory() = delete;

  // register creator
  using CreatorFunc = std::function<std::unique_ptr<LayerSynchronizer>(int)>;
  void register_creator(DeviceType device_type, CreatorFunc creator) {
    creators_[device_type] = creator;
  }

  std::unique_ptr<LayerSynchronizer> create_context(DeviceType type,
                                                    int device_id = 0);

  static LayerSynchronizerFactory& instance() {
    static LayerSynchronizerFactory factory;

    return factory;
  }

 private:
  LayerSynchronizerFactory();

  std::unordered_map<DeviceType, CreatorFunc> creators_;
};

inline std::unique_ptr<LayerSynchronizer> create_device_context(
    nt device_id = 0,
    DeviceType type = DeviceType::NPU_ASCEND) {
  return LayerSynchronizerFactory::instance().create_context(type, device_id);
}

}  // namespace platform
}  // namespace xllm
