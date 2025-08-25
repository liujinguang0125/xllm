#include "npu_rms_norm_impl.h"

#include <glog/logging.h>

// #include "attn_mask.h"

namespace xllm::layer {

void NpuRmsNormImpl::param_from_args(atb::infer::RmsNormParam& param,
                                     const ModelArgs& args) {
  param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  param.normParam.epsilon = args.rms_norm_eps();
}

NpuRmsNormImpl::NpuRmsNormImpl(const Context& context) : NpuBaseLayer(context) {
  param_from_args(norm_param_, context.get_model_args());

  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
}

void NpuRmsNormImpl::verify_loaded_weights(const std::string weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final norm weight is not loaded for " << weight_str;
}

void NpuRmsNormImpl::merge_loaded_weights() {
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[0]);
  init_layer();
}

void NpuRmsNormImpl::load_state_dict(const StateDict& state_dict) {
  set_weight(state_dict, "weight", 0);
  at_weight_tensors_[0] = at_weight_tensors_[0].to(dtype_);
}

int64_t NpuRmsNormImpl::init_layer() {
  name_ = "rms_norm_layer";
  model_name_ = "llm";
  run_task_func_ = std::bind(&NpuRmsNormImpl::run_task,
                             this,
                             std::placeholders::_1,
                             std::placeholders::_2);
  CHECK_OPERATION_STATUS_RETURN(init_node(norm_node_, norm_param_));

  return atb::NO_ERROR;
}

int64_t NpuRmsNormImpl::init_node(atb_speed::Model::Node& node,
                                  atb::infer::RmsNormParam& param) {
  atb::Operation* operation = nullptr;
  atb::Status atbStatus = atb::CreateOperation(param, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);

  node.inTensors.at(1) = &atb_weight_tensors_[0];
  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor NpuRmsNormImpl::forward(torch::Tensor& x,
                                      atb::Context* context,
                                      AtbWorkspace& workspace,
                                      int node_id) {
  atb::Status st;

  build_node_variant_pack(norm_node_, x);
  st = execute_node(norm_node_, context, workspace, node_id);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;

  return x;
}

void NpuRmsNormImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                             torch::Tensor& x) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  node.variantPack.inTensors.at(0) = internal_tensors_;
  node.variantPack.inTensors.at(1) = *node.inTensors.at(1);
  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace xllm::layer
