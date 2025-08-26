#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "ascend/atb_device_context.h"
#include "device_context.h"
#include "device_type.h"

namespace xllm {
namespace platform {

class DeviceContextFactory {
 public:
  DeviceContextFactory() = delete;
  ~DeviceContextFactory() = delete;

  // register creator
  using CreatorFunc = std::function<std::unique_ptr<DeviceContext>(int)>;
  void register_creator(DeviceType device_type, CreatorFunc creator) {
    creators_[device_type] = creator;
  }

  std::unique_ptr<DeviceContext> create_context(DeviceType type,
                                                int device_id = 0);

  static DeviceContextFactory& instance() {
    static DeviceContextFactory factory;

    return factory;
  }

 private:
  DeviceContextFactory();

  std::unordered_map<DeviceType, CreatorFunc> creators_;
};

inline std::unique_ptr<DeviceContext> create_device_context(
    nt device_id = 0,
    DeviceType type = DeviceType::NPU_ASCEND) {
  return DeviceContextFactory::instance().create_context(type, device_id);
}

}  // namespace platform
}  // namespace xllm