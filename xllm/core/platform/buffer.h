#pragma once

#include <torch/torch.h>

namespace xllm {
namespace platform {

// constexpr uint64_t KB_1 = 1024;
// constexpr uint64_t MB_1 = 1024 * 1024;
// constexpr uint64_t GB_1 = 1024 * 1024 * 1024;
// constexpr uint64_t DIM_NUM_2 = 2;

class Buffer {
 public:
  explicit Buffer(uint64_t buffer_size, at::Device device)
      : buffer_size_(buffer_size), device_(device) {};

  virtual ~Buffer() {};

  virtual void* get_buffer(uint64_t buffer_size) = 0;

 protected:
  void* buffer_ = nullptr;
  uint64_t buffer_size_ = 0;
  torch::Tensor at_tensor_;
  at::Device device_;
};

}  // namespace platform
}  // namespace xllm