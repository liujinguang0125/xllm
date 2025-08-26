
#include <glog/logging.h>

#include "atb_layer_synchronizer.h"

namespace xllm {
namespace platform {

NpuLayerSynchronizer::NpuLayerSynchronizer(int64_t num_layers)
    : LayerSynchronizer(num_layers), events_(num_layers, nullptr) {
  uint32_t flags = ACL_EVENT_SYNC;
  for (int64_t i = 0; i < num_layers; ++i) {
    auto ret = aclrtCreateEventWithFlag(&events_[i], flags);
    CHECK(ret == ACL_SUCCESS) << "Create event failed.";
  }
}

NpuLayerSynchronizer::~NpuLayerSynchronizer() {
  for (int64_t i = 0; i < events_.size(); ++i) {
    aclrtDestroyEvent(events_[i]);
  }
}

void* NpuLayerSynchronizer::get_event(int64_t layer_index) {
  return &events_[layer_index];
}

bool NpuLayerSynchronizer::synchronize_layer(int64_t layer_index) {
  while (!event_record_flags_[layer_index].load(std::memory_order_acquire));
  auto ret = aclrtSynchronizeEvent(events_[layer_index]);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Synchronize event failed.";
    return false;
  }
  return true;
}

}  // namespace platform
}  // namespace xllm