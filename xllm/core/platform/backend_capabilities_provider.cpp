/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "backend_capabilities_provider.h"

#include "backend_capabilities.h"
// #include "cuda/cuda_utils.h"
// #include "npu/npu_utils.h"
// #include "mlu/mlu_utils.h"
// #include "musa/musa_utils.h"
// #include "ilu/ilu_utils.h"

BackendCapabilities BackendCapabilitiesProvider::query(int32_t device_id) {
  BackendCapabilities caps;

#if defined(USE_CUDA)
  caps.backend_name = "cuda";
  caps.device_count = c10::cuda::device_count();

  auto [major, minor] = cuda::get_compute_capability(device_id);
  caps.compute_capability_major = major;
  caps.compute_capability_minor = minor;
  caps.compute_unit_count = cuda::get_device_sm_count(device_id);

  // Populate chip model from compute capability
  // if (major == 8 && minor == 0) caps.chip_model = "A100";
  // else if (major == 9 && minor == 0) caps.chip_model = "H100";
  // ...

  // Feature flags
  caps.supported_features.insert("flash_attention");
  caps.supported_features.insert("paged_attention");
  if (major >= 9) {
    caps.supported_features.insert("fp8");
    caps.supported_features.insert("pdl");
  }
  caps.supported_features.insert("graph_capture");
  caps.supported_features.insert("vmm");

  // CUDA-specific extended properties
  caps.set("cuda_version", cuda::get_cuda_version());
  caps.set("sm_count", caps.compute_unit_count);

#elif defined(USE_NPU)
  caps.backend_name = "npu";
  caps.device_count = c10_npu::device_count();
  caps.supported_features.insert("graph_capture");
  caps.supported_features.insert("layer_synchronizer");
  // NPU-specific
  caps.set("acl_version", get_acl_version());

#elif defined(USE_MLU)
  caps.backend_name = "mlu";
  caps.device_count = torch_mlu::device_count();
  caps.supported_features.insert("flash_attention");

#elif defined(USE_MUSA)
  caps.backend_name = "musa";
  caps.device_count = c10::musa::device_count();
#endif

  return caps;
}