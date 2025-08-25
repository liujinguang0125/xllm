#include "base_layer.h"

namespace xllm {
namespace layer {

BaseLayer::BaseLayer(const Context& context)
    : device_(context.get_tensor_options().device()),
      name_(""),
      parallel_args_(context.get_parallel_args()) {
  auto quant_args = context.get_quant_args();
  if (!quant_args.quantize_type().empty()) {
    quantize_type_ = quant_args.quantize_type();
  }

  if (!quant_args.torch_dtype().empty()) {
    torch_dtype_ = quant_args.torch_dtype();
  }

  dp_size_ = parallel_args_.dp_size();
  dp_local_tp_size_ = parallel_args_.world_size() / dp_size_;
  dp_rank_ = parallel_args_.rank() / dp_local_tp_size_;
  CHECK_EQ(parallel_args_.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args_.rank() % dp_local_tp_size_;

  run_task_func_ = std::bind(
      &BaseLayer::run_task, this, std::placeholders::_1, std::placeholders::_2);
}

torch::Dtype BaseLayer::string2dtype(const std::string& dtype_str) {
  if (dtype_str.compare("float16") == 0) {
    return torch::kFloat16;
  } else if (dtype_str.compare("bfloat16") == 0) {
    return torch::kBFloat16;
  } else if (dtype_str.compare("float32") == 0) {
    return torch::kFloat32;
  } else if (dtype_str.compare("float64") == 0) {
    return torch::kFloat64;
  } else if (dtype_str.compare("int8") == 0) {
    return torch::kInt8;
  } else if (dtype_str.compare("int16") == 0) {
    return torch::kInt16;
  } else if (dtype_str.compare("int32") == 0) {
    return torch::kInt32;
  } else if (dtype_str.compare("int64") == 0) {
    return torch::kInt64;
  } else if (dtype_str.compare("uint8") == 0) {
    return torch::kUInt8;
  } else if (dtype_str.compare("bool") == 0) {
    return torch::kBool;
  }

  throw std::runtime_error("Unsupported dtype string");
}

void BaseLayer::correct_tensor_dtype(torch::Tensor& tensor,
                                     const std::string& tensorName) {
  if (absl::EndsWith(tensorName, "deq_scale") &&
      (torch_dtype_.compare("bfloat16") == 0)) {
    return;
  }

  if (tensor.dtype() != torch::kInt8 && tensor.dtype() != torch::kInt32 &&
      tensor.dtype() != torch::kInt64) {
    torch::Dtype dtype = string2dtype(torch_dtype_);
    tensor = tensor.to(dtype);
  }
}

void BaseLayer::set_weight(const StateDict& state_dict,
                           const std::string& tensor_name,
                           int weight_position) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      correct_tensor_dtype(mutable_tensor, tensor_name);
      at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
    }
  }
}

void BaseLayer::set_weight(const StateDict& state_dict,
                           const std::string& tensor_name,
                           int weight_position,
                           int dim) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      if (parallel_args_.world_size() <= 1) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
      } else {
        at_weight_tensors_[weight_position] =
            state_dict
                .get_sharded_tensor(tensor_name,
                                    /*dim=*/dim,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size())
                .to(device_);
        correct_tensor_dtype(at_weight_tensors_[weight_position], tensor_name);
      }
    }
  }
}

void BaseLayer::set_weight(const StateDict& state_dict,
                           const std::string& tensor_name,
                           int weight_position,
                           int dim,
                           int rank,
                           int world_size) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      if (world_size <= 1) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
      } else {
        at_weight_tensors_[weight_position] =
            state_dict
                .get_sharded_tensor(tensor_name,
                                    /*dim=*/dim,
                                    /*rank=*/rank,
                                    /*world_size=*/world_size)
                .to(device_);
        correct_tensor_dtype(at_weight_tensors_[weight_position], tensor_name);
      }
    }
  }
}

}  // namespace layer
}  // namespace xllm
