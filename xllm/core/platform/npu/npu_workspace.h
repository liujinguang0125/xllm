#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "npu_buffer.h"
#include "platform/workspace.h"

namespace xllm {
namespace platform {

class NpuWorkspace : public Workspace {
 public:
  explicit NpuWorkspace(at::Device device);

  ~NpuWorkspace();

  virtual void* get_workspace_buffer(uint64_t buffer_size) override;

 private:
  static std::map<int32_t, std::unique_ptr<NpuBuffer>> buffer_map_;
};

}  // namespace platform
}  // namespace xllm
