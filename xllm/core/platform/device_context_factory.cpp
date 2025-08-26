
#include "device_context_factory.h"

#include "ascend/atb_device_context.h"

namespace xllm {
namespace hal {

DeviceContextFactory::DeviceContextFactory() {
  // #ifdef WITH_ASCEND
  register_creator(DeviceType::NPU_ASCEND, [](int device_id) {
    return std::make_unique<AtbDeviceContext>(device_id);
  });
  // #endif
}

std::unique_ptr<DeviceContext> DeviceContextFactory::create_context(
    DeviceType type,
    int device_id = 0) {
  auto it = creators_.find(type);
  if (it != creators_.end()) {
    return it->second(device_id);
  }

  throw std::runtime_error("No registered creator for device type: " +
                           std::to_string(static_cast<int>(type)));
}

}  // namespace hal
}  // namespace xllm