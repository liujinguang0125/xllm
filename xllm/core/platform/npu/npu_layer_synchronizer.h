#pragma once

#include <acl/acl.h>

#include "platform/layer_synchronizer.h"

namespace xllm {
namespace platform {

class NpuLayerSynchronizer : public LayerSynchronizer {
 public:
  explicit NpuLayerSynchronizer(int64_t num_layers);

  ~NpuLayerSynchronizer();

  virtual void* get_event(const int64_t layer_index) override;

  virtual bool synchronize_layer(int64_t layer_index) override;

 protected:
  std::vector<aclrtEvent> events_;
};

}  // namespace platform
}  // namespace xllm