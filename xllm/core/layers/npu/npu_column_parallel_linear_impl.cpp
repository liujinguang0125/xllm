#include "layers/npu/npu_column_parallel_linear_impl.h"

#include "framework/parallel_state.h"

namespace xllm::layer {

// std::shared_ptr<NpuColumnParallelLinearImpl>
// create_atb_column_parallel_linear_layer(const Context& context) {
//   return std::make_shared<NpuColumnParallelLinearImpl>(context);
// }

void NpuColumnParallelLinearImpl::param_from_args(
    atb_speed::common::LinearParallelParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.fusionLinearParam.isBF16 = args.dtype() == "bfloat16";
  param.unpadInputs = true;
  param.fusionLinearParam.transposeType = 1;
  if (parallel_args.world_size() > 1) {
    if (dp_size_ > 1) {
      param.tensorParallelInfo.rank = dp_local_tp_rank_;
      param.tensorParallelInfo.worldSize = dp_local_tp_size_;
    } else {
      param.tensorParallelInfo.rank = parallel_args.rank();
      param.tensorParallelInfo.worldSize = parallel_args.world_size();
    }
    param.parallelType = atb_speed::common::COLUMN_PARALLEL;
    param.tensorParallelInfo.commDomain = std::to_string(dp_rank_);
    // param.tensorParallelInfo.backend =
    // FLAGS_communication_backend;
    param.tensorParallelInfo.backend = "lccl";
  }
}

NpuColumnParallelLinearImpl::NpuColumnParallelLinearImpl(const Context& context)
    : NpuBaseLayer(context) {
  param_from_args(
      linear_param_, context.get_model_args(), context.get_parallel_args());
  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);
  at_out_tensors_.resize(1);

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
  tensor_placeholder_ = torch::zeros({1}).to(options);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
}

void NpuColumnParallelLinearImpl::verify_loaded_weights(
    const std::string weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

void NpuColumnParallelLinearImpl::merge_loaded_weights() {
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[0]);
  init_layer();
}

void NpuColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (dp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 0, dp_local_tp_rank_, dp_local_tp_size_);
  } else {
    set_weight(state_dict, "weight", 0, 0);
  }
  at_weight_tensors_[0] = at_weight_tensors_[0].to(dtype_);
}

int64_t NpuColumnParallelLinearImpl::init_layer() {
  name_ = "atb_parallel_linear_layer";
  model_name_ = "Atb Parallel Linear";
  run_task_func_ = std::bind(&NpuColumnParallelLinearImpl::run_task,
                             this,
                             std::placeholders::_1,
                             std::placeholders::_2);
  CHECK_OPERATION_STATUS_RETURN(init_node(linear_node_, linear_param_));

  return atb::NO_ERROR;
}

int64_t NpuColumnParallelLinearImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::common::LinearParallelParam& linearParam) {
  atb::Operation* operation = nullptr;
  atb::Status atbStatus =
      atb_speed::common::LinearParallel(linearParam, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Get unexpected input num: " << node.operation->GetInputNum();
    return -1;
  }
  if (node.operation->GetOutputNum() < 1) {
    LOG(ERROR) << "Get unexpected output num: "
               << node.operation->GetOutputNum();
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);

  node.inTensors.at(1) = &atb_weight_tensors_[0];

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  ATB_SPEED_LOG_DEBUG("AddLinear");

  return atb::NO_ERROR;
}

torch::Tensor NpuColumnParallelLinearImpl::forward(const torch::Tensor& input,
                                                   atb::Context* context,
                                                   AtbWorkspace& workspace,
                                                   int nodeId) {
  atb::Status st;
  build_node_variant_pack(linear_node_, input);
  st = execute_node(linear_node_, context, workspace, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;

  return at_out_tensors_.at(0);
}

void NpuColumnParallelLinearImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    const torch::Tensor& input) {
  internal_input = atb_speed::Utils::AtTensor2Tensor(input);

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.operation->GetInputNum());
  inTensorDescs.resize(node.operation->GetInputNum());

  atb::SVector<atb::TensorDesc> outTensorDescs;
  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.at(0) = internal_input;
  inTensorDescs.at(0) = internal_input.desc;
  // weight
  node.variantPack.inTensors.at(1) = *node.inTensors.at(1);
  inTensorDescs.at(1) = node.inTensors.at(1)->desc;

  for (int i = 2; i < 7; i++) {
    node.variantPack.inTensors.at(i) = placeholder_;
    inTensorDescs.at(i) = placeholder_.desc;
  }

  node.operation->InferShape(inTensorDescs, outTensorDescs);
  at::Tensor output =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));
  at_out_tensors_.at(0) = output;
  node.variantPack.outTensors.at(0) =
      atb_speed::Utils::AtTensor2Tensor(at_out_tensors_.at(0));
}

// AtbColumnParallelLinear::AtbColumnParallelLinear(const Context& context)
//     : ModuleHolder(create_atb_column_parallel_linear_layer(context)) {}

}  // namespace xllm::layer
