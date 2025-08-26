#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "buffer.h"

namespace xllm {
namespace platform {

class Workspace {
 public:
  Workspace() = default;

  virtual ~Workspace() = default;

  virtual void* get_workspace_buffer(uint64_t buffer_size) = 0;
};

}  // namespace platform
}  // namespace xllm