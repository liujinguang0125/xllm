#pragma once

#include <atomic>
#include <vector>

namespace xllm {
namespace platform {

class LayerSynchronizer {
 public:
  explicit LayerSynchronizer(int64_t num_layers)
      : event_record_flags_(num_layers) {}

  virtual ~LayerSynchronizer() = default;

  virtual void* get_event(const int64_t layer_index) = 0;

  std::atomic<bool>& get_event_flag(int64_t layer_index) {
    return event_record_flags_[layer_index];
  }

  virtual bool synchronize_layer(int64_t layer_index) = 0;

 protected:
  std::vector<std::atomic<bool>> event_record_flags_;
};

}  // namespace platform
}  // namespace xllm