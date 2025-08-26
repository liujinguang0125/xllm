include(cc_library)
include(cc_test)

include_directories(
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/ascend
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/ascend/core/include
)

cc_library(
  NAME
    hal
  HDRS
    buffer.h 
    # device_context_factory.h
    device_context.h
    device_type.h
    layer_synchronizer.h
    memory_direction.h
    workspace.h
  SRCS
    buffer.cpp
    # device_context_factory.cpp
    layer_synchronizer.cpp
  DEPS
    :hal_ascend
    :parallel_state
    :state_dict
    :block
    :kv_cache
    :prefix_cache
    glog::glog
    gflags::gflags
    torch
)

add_subdirectory(ascend)
