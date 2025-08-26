#pragma once

#include <torch/torch.h>

#include "atb/atb_infer.h"
#include "npu_workspace.h"
#include "platform/device_context.h"

namespace xllm {
namespace platform {

class NpuDeviceContext : public DeviceContext {
 public:
  explicit NpuDeviceContext(int device_id, at::Device device);
  virtual ~NpuDeviceContext();

  // memory manage
  virtual void* allocate(size_t size) override;
  virtual void deallocate(void* ptr) override;
  virtual void memcpy(void* dst,
                      const void* src,
                      size_t size,
                      MemcpyDirection dir) override;

  virtual Workspace* get_workspace() override;

  virtual void* get_context() override;

  // stream manage
  virtual void* create_stream() override;
  virtual void destroy_stream(void* stream) override;
  virtual void stream_synchronize(void* stream) override;

  // event manage
  virtual void* create_event() override;
  virtual void record_event(void* event, void* stream) override;
  virtual void event_synchronize(void* event) override;

  // device attribute
  virtual size_t get_memory_capacity() override;
  virtual int get_sm_count() override;

 private:
  aclrtContext context_;
  aclrtStream default_stream_;
  NpuWorkspace npu_workspace_;
};

}  // namespace platform
}  // namespace xllm