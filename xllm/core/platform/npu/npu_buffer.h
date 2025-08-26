#pragma once

#include <acl/acl.h>

#include "atb/atb_infer.h"
#include "platform/buffer.h"

namespace xllm {
namespace platform {

class NpuBuffer : public Buffer {
 public:
  explicit NpuBuffer(uint64_t buffer_size, at::Device device);

  virtual ~NpuBuffer() {};

  virtual void* get_buffer(uint64_t buffer_size) override;

 private:
  torch::Tensor create_attensor(uint64_t buffer_size) const;
  at::Tensor create_attensor_from_tensor_desc(
      const atb::TensorDesc& tensorDesc) const;
};

}  // namespace platform
}  // namespace xllm
