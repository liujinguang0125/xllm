#pragma

#include "atb_device_context.h"

namespace xllm {
namespace platform {

NpuDeviceContext::NpuDeviceContext(int device_id, at::Device device)
    : DeviceContext(device_id), atb_workspace_(device) {
  // create device
  aclrtSetDevice(device_id_);

  // create context
  aclrtCreateContext(&context_, device_id_);

  // create default stream
  aclrtCreateStream(&default_stream_);
}

NpuDeviceContext::~NpuDeviceContext() {
  aclrtDestroyStream(default_stream_);
  aclrtDestroyContext(context_);
  aclrtResetDevice(device_id_);
}

void NpuDeviceContext::memcpy(void* dst,
                              const void* src,
                              size_t size,
                              MemcpyDirection dir) {
  aclrtMemcpyKind kind;
  switch (dir) {
    case MemcpyDirection::HOST_TO_DEVICE:
      kind = ACL_MEMCPY_HOST_TO_DEVICE;
      break;
    case MemcpyDirection::DEVICE_TO_HOST:
      kind = ACL_MEMCPY_DEVICE_TO_HOST;
      break;
    case MemcpyDirection::DEVICE_TO_DEVICE:
      kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
      break;
    case MemcpyDirection::NPU_TO_NPU_SHARED:
      kind = ACL_MEMCPY_INNER_DEVICE_TO_DEVICE;
      break;

    default:
      throw std::runtime_error("Unsupported memcpy direction for Ascend");
  }

  aclrtMemcpy(dst, size, src, size, kind);
}

}  // namespace platform
}  // namespace xllm