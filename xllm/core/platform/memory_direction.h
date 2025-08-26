#pragma once

namespace xllm {
namespace platform {

/**
 * Enumeration for memory copy direction
 *
 * Provides precise control over data transfer direction in heterogeneous
 * computing environments:
 * 1. Distinguishes transfer directions between host (CPU) and device
 * (GPU/NPU/XPU)
 * 2. Supports direct memory access (DMA) between devices
 * 3. Includes special identifiers for host-internal copies
 */
enum class MemcpyDirection {
  // Basic directions
  HOST_TO_HOST,      // Host memory → Host memory (traditional memcpy)
  HOST_TO_DEVICE,    // Host memory → Device memory
  DEVICE_TO_HOST,    // Device memory → Host memory
  DEVICE_TO_DEVICE,  // Device memory → Device memory

  // Advanced directions (hardware-optimized)
  DEVICE_TO_DEVICE_PEER,      // Direct cross-device copy (e.g. NVIDIA GPU P2P)
  HOST_TO_DEVICE_PINNED,      // Host pinned memory → Device (zero-copy
                              // optimization)
  DEVICE_TO_HOST_PINNED,      // Device → Host pinned memory
  DEVICE_TO_DEVICE_VIA_HOST,  // Device→Host→Device (fallback path)

  // Vendor-specific optimized paths
  NPU_TO_NPU_SHARED,  // Transfer between Ascend NPUs via HCCS bus
  GPU_TO_NPU_DIRECT,  // NVIDIA GPU→Huawei NPU direct transfer (e.g. via RoCE)
  NPU_TO_GPU_DIRECT,  // Huawei NPU→NVIDIA GPU direct transfer

  // Automatic direction detection
  AUTO  // Automatically selects optimal path based on buffer attributes
};

// Direction property checker utility
struct MemcpyDirectionUtil {
  static bool is_host_to_device(MemcpyDirection dir) {
    return dir == MemcpyDirection::HOST_TO_DEVICE ||
           dir == MemcpyDirection::HOST_TO_DEVICE_PINNED;
  }

  static bool is_device_to_host(MemcpyDirection dir) {
    return dir == MemcpyDirection::DEVICE_TO_HOST ||
           dir == MemcpyDirection::DEVICE_TO_HOST_PINNED;
  }

  static bool is_device_to_device(MemcpyDirection dir) {
    return dir == MemcpyDirection::DEVICE_TO_DEVICE ||
           dir == MemcpyDirection::DEVICE_TO_DEVICE_PEER ||
           dir == MemcpyDirection::NPU_TO_NPU_SHARED ||
           dir == MemcpyDirection::GPU_TO_NPU_DIRECT ||
           dir == MemcpyDirection::NPU_TO_GPU_DIRECT;
  }

  // Checks if intermediate host buffer is required
  static bool requires_host_buffer(MemcpyDirection dir) {
    return dir == MemcpyDirection::DEVICE_TO_DEVICE_VIA_HOST;
  }
};

}  // namespace platform
}  // namespace xllm