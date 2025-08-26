#pragma once

#include <memory>

#include "memory_direction.h"
#include "workspace.h"

namespace xllm {
namespace platform {

class DeviceContext {
 public:
  DeviceContext(int device_id) : device_id_(device_id) {};

  virtual ~DeviceContext() = default;

  // memory manage
  virtual void* allocate(size_t size) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual void memcpy(void* dst,
                      const void* src,
                      size_t size,
                      MemcpyDirection dir) = 0;

  virtual Workspace* get_workspace() = 0;

  virtual void* get_context() = 0;

  // stream manage
  virtual void* create_stream() = 0;
  virtual void destroy_stream(void* stream) = 0;
  virtual void stream_synchronize(void* stream) = 0;

  // event manage
  virtual void* create_event() = 0;
  virtual void record_event(void* event, void* stream) = 0;
  virtual void event_synchronize(void* event) = 0;

  // device attribute
  virtual size_t get_memory_capacity() = 0;
  virtual int get_sm_count() = 0;

 protected:
  int device_id_;
};

// std::shared_ptr<DeviceContext> create_device_context(int device_id);

}  // namespace platform
}  // namespace xllm