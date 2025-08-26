#pragma once

namespace xllm {
namespace platform {

enum class DeviceType {
  NPU_ASCEND = 0,  // Huawei Ascend
  NPU_MLU,         // cambricon MLU
  CPU,             // host CPU
  CUDA,            // Nvidia GPU
};

}  // namespace platform
}  // namespace xllm
