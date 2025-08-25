#include "atb_base.h"

#include <glog/logging.h>

namespace xllm::hf {
static std::atomic<bool> g_executeOk(true);

ATBBase::ATBBase(const Context& context)
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

  runTaskFunc_ = std::bind(
      &ATBBase::run_task, this, std::placeholders::_1, std::placeholders::_2);
}

void ATBBase::print_atbtensor(const atb::Tensor& tensor, int i) {
  static std::map<aclDataType, at::ScalarType> dtypeMap = {
      {ACL_BOOL, at::ScalarType::Bool},
      {ACL_UINT8, at::ScalarType::Byte},
      {ACL_INT8, at::ScalarType::Char},
      {ACL_FLOAT16, at::ScalarType::Half},
      {ACL_FLOAT, at::ScalarType::Float},
      {ACL_INT32, at::ScalarType::Int},
      {ACL_INT64, at::ScalarType::Long},
      {ACL_BF16, at::ScalarType::BFloat16},
  };
  std::vector<int64_t> sizes;
  for (uint64_t i = 0; i < tensor.desc.shape.dimNum; i++) {
    sizes.push_back(tensor.desc.shape.dims[i]);
  }
  at::IntArrayRef sizeRef(sizes);
  at::ScalarType dtype;
  auto it = dtypeMap.find(tensor.desc.dtype);
  if (it != dtypeMap.end()) {
    dtype = it->second;
  } else {
    throw std::runtime_error("AtTensor2Tensor: not support dtype");
  }
  at::Tensor atTensor = torch::empty(sizeRef, at::TensorOptions().dtype(dtype));
  if (!atTensor.is_contiguous()) {
    atTensor = atTensor.contiguous();
  }
  aclrtMemcpy(atTensor.data_ptr(),
              tensor.dataSize,
              tensor.deviceData,
              tensor.dataSize,
              ACL_MEMCPY_DEVICE_TO_HOST);
}

torch::Dtype ATBBase::string_2_dtype(const std::string& dtype_str) {
  if (dtype_str == "float16") {
    return torch::kFloat16;
  } else if (dtype_str == "bfloat16") {
    return torch::kBFloat16;
  } else if (dtype_str == "float32") {
    return torch::kFloat32;
  } else if (dtype_str == "float64") {
    return torch::kFloat64;
  } else if (dtype_str == "int8") {
    return torch::kInt8;
  } else if (dtype_str == "int16") {
    return torch::kInt16;
  } else if (dtype_str == "int32") {
    return torch::kInt32;
  } else if (dtype_str == "int64") {
    return torch::kInt64;
  } else if (dtype_str == "uint8") {
    return torch::kUInt8;
  } else if (dtype_str == "bool") {
    return torch::kBool;
  }
  throw std::runtime_error("Unsupported dtype string");
}

void ATBBase::correct_tensor_dtype(torch::Tensor& tensor,
                                   const std::string& tensorName) {
  if (absl::EndsWith(tensorName, "deq_scale") && torch_dtype_ == "bfloat16") {
    return;
  }
  if (tensor.dtype() != torch::kInt8 && tensor.dtype() != torch::kInt32 &&
      tensor.dtype() != torch::kInt64) {
    torch::Dtype dtype = string_2_dtype(torch_dtype_);
    tensor = tensor.to(dtype);
  }
}

void ATBBase::set_weight(const StateDict& state_dict,
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

void ATBBase::set_weight(const StateDict& state_dict,
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

void ATBBase::set_weight(const StateDict& state_dict,
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

void ATBBase::run_task(std::string taskName, std::function<int()> task) const {
  at_npu::native::OpCommand cmd;
  cmd.Name(taskName);
  cmd.SetCustomHandler(task);
  cmd.Run();
}

atb::Status ATBBase::execute_node(atb_speed::Model::Node& node,
                                  atb::Context* context,
                                  AtbWorkspace& workspace,
                                  int nodeId,
                                  aclrtEvent* event,
                                  std::atomic<bool>* event_flag) {
  if (!g_executeOk) {
    std::stringstream ss;
    ss << "execute fail, enable log: export ASDOPS_LOG_LEVEL=ERROR, export "
          "ASDOPS_LOG_TO_STDOUT=1 to find the "
          "first error.  For more details, see the npu official document. "
       << std::endl;
    throw std::runtime_error(ss.str());
  }
  context_ = context;
  atb::Status st =
      node.operation->Setup(node.variantPack, node.workspaceSize, context_);
  if (st != 0) {
    LOG(ERROR) << " setup layer node fail, not call execute";
    return st;
  }

  if (node.workspaceSize > 0) {
    node.workspace = workspace.get_workspace_buffer(node.workspaceSize);
  }

  runTaskFunc_(name_ + std::to_string(nodeId), [=]() {
    return execute_plan(
        node, name_ + std::to_string(nodeId), event, event_flag);
  });

  return st;
}

atb::Status ATBBase::execute_plan(const atb_speed::Model::Node& node,
                                  std::string opName_,
                                  aclrtEvent* event,
                                  std::atomic<bool>* event_flag) {
  atb::Status st = node.operation->Execute(
      node.variantPack, (uint8_t*)node.workspace, node.workspaceSize, context_);
  LOG_IF(ERROR, st != 0) << name_ << " execute plan fail, error code: " << st;
  if (st == 0 && event != nullptr) {
    aclrtStream stream = context_->GetExecuteStream();
    auto ret = aclrtRecordEvent(*event, stream);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Record event failed.";
      return st;
    }
    event_flag->store(true, std::memory_order_release);
  }
  return st;
}

}  // namespace xllm::hf
