#include "npu_deepseek_v2_decoder_layer_impl.h"

#include <gflags/gflags.h>

DECLARE_string(rank_tablefile);
DECLARE_string(communication_backend);
DECLARE_int32(expert_parallel_degree);

namespace xllm {
namespace layer {

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,  //[7168]
  IN_INPUT_NORM_BIAS = 1,    //[7168]
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_Q_PROJ_A_WEIGHT = 4,   //[1536, 7168]
  IN_Q_PROJ_A_BIAS = 5,     //[1536]
  IN_Q_PROJ_A_DESCALE = 6,  //[1536]
  IN_Q_PROJ_A_OFFSET = 7,   //[1]
  IN_Q_PROJ_A_SCALE = 8,    //[1]
  IN_Q_PROJ_A_COMPRESS_IDX = 9,
  IN_Q_PROJ_A_LAYERNORM_WEIGHT = 10,  //[1536]
  IN_Q_PROJ_A_LAYERNORM_BIAS = 11,    //[1536]

  IN_Q_PROJ_B_WEIGHT = 12,   //[6144, 1536]
  IN_Q_PROJ_B_BIAS = 13,     //[6144]
  IN_Q_PROJ_B_DESCALE = 14,  //[6144]
  IN_Q_PROJ_B_OFFSET = 15,   //[1]
  IN_Q_PROJ_B_SCALE = 16,    //[1]
  IN_Q_PROJ_B_COMPRESS_IDX = 17,

  IN_KV_PROJ_WITH_MQA_WEIGHT = 18,   //[576, 7168]
  IN_KV_PROJ_WITH_MQA_BIAS = 19,     //[576]
  IN_KV_PROJ_WITH_MQA_DESCALE = 20,  //[576]
  IN_KV_PROJ_WITH_MQA_OFFSET = 21,   //[1]
  IN_KV_PROJ_WITH_MQA_SCALE = 22,    //[1]
  IN_KV_PROJ_WITH_MQA_COMPRESS_IDX = 23,

  IN_KV_PROJ_A_LAYERNORM_WEIGHT = 24,  //[512]
  IN_KV_PROJ_A_LAYERNORM_BIAS = 25,

  IN_K_PROJ_B_FOR_Q_WEIGHT = 26,  //[8, 128, 512]
  IN_K_PROJ_B_FOR_Q_BIAS = 27,
  IN_K_PROJ_B_FOR_Q_DESCALE = 28,
  IN_K_PROJ_B_FOR_Q_OFFSET = 29,
  IN_K_PROJ_B_FOR_Q_SCALE = 30,
  IN_K_PROJ_B_FOR_Q_COMPRESS_IDX = 31,

  IN_V_PROJ_B_FOR_O_WEIGHT = 32,  //[32, 512, 128]
  IN_V_PROJ_B_FOR_O_BIAS = 33,
  IN_V_PROJ_B_FOR_O_DESCALE = 34,
  IN_V_PROJ_B_FOR_O_OFFSET = 35,
  IN_V_PROJ_B_FOR_O_SCALE = 36,
  IN_V_PROJ_B_FOR_O_COMPRESS_IDX = 37,

  IN_ATTENTION_OUT_WEIGHT = 38,   //[7168, 4096]
  IN_ATTENTION_OUT_BIAS = 39,     //[7168]
  IN_ATTENTION_OUT_DESCALE = 40,  //[7168]
  IN_ATTENTION_OUT_OFFSET = 41,
  IN_ATTENTION_OUT_SCALE = 42,
  IN_ATTENTION_OUT_COMPRESS_IDX = 43,

  IN_SELFATTENTION_OUT_NORM_WEIGHT = 44,  //[7168]
  IN_SELFATTENTION_OUT_NORM_BIAS = 45,
  IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT = 46,
  IN_SELFATTENTION_OUT_NEW_NORM_BIAS = 47,

  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = 48,  //[1024, 7168]]
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT = 49,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = 50,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = 51,  //[1024]
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT = 52,   //[1024]
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT = 53,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = 54,  //[7168, 512]
  IN_MLP_DOWN_BIAS_SHARED_EXPERT = 55,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT = 56,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT = 57,  //[7168]
  IN_MLP_DOWN_SCALE_SHARED_EXPERT = 58,   //[7168]
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = 59,

  IN_SHARED_EXPERT_GATE_WEIGHT = 60,
  IN_SHARED_EXPERT_GATE_BIAS = 61,
  IN_SHARED_EXPERT_GATE_DESCALE = 62,
  IN_SHARED_EXPERT_GATE_OFFSET = 63,
  IN_SHARED_EXPERT_GATE_SCALE = 64,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX = 65,

  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 66,  //[256, 7168]
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 67,    //[256]
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE = 68,
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET = 69,
  IN_BLOCK_SPARSE_MOE_GATE_SCALE = 70,
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 71,

  IN_MLP_GATEUP_WEIGHT_EXPERT = 72,  //[256, 7168, 1024]
  IN_MLP_GATEUP_BIAS_EXPERT = 73,
  IN_MLP_GATEUP_DESCALE_EXPERT = 74,
  IN_MLP_GATEUP_OFFSET_EXPERT = 75,  //[256, 1024]
  IN_MLP_GATEUP_SCALE_EXPERT = 76,   //[256, 1024]
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = 77,

  IN_MLP_DOWN_WEIGHT_EXPERT = 78,  //[256, 512, 7168]
  IN_MLP_DOWN_BIAS_EXPERT = 79,
  IN_MLP_DOWN_DESCALE_EXPERT = 80,
  IN_MLP_DOWN_OFFSET_EXPERT = 81,  //[256, 7168]
  IN_MLP_DOWN_SCALE_EXPERT = 82,   //[256, 7168]
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 83,
};

static const uint64_t WEIGHT_COUNT_PER_LAYER = 84;

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {};

static const std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_BIAS},

    {"self_attn.q_a_proj.weight", IN_Q_PROJ_A_WEIGHT},
    {"self_attn.q_a_proj.quant_bias", IN_Q_PROJ_A_BIAS},
    {"self_attn.q_a_proj.deq_scale", IN_Q_PROJ_A_DESCALE},
    {"self_attn.q_a_proj.input_offset", IN_Q_PROJ_A_OFFSET},
    {"self_attn.q_a_proj.input_scale", IN_Q_PROJ_A_SCALE},
    {"self_attn.q_a_layernorm.weight", IN_Q_PROJ_A_LAYERNORM_WEIGHT},
    {"self_attn.q_a_layernorm.bias", IN_Q_PROJ_A_LAYERNORM_BIAS},

    {"self_attn.q_proj.weight", IN_Q_PROJ_B_WEIGHT},
    {"self_attn.q_b_proj.weight", IN_Q_PROJ_B_WEIGHT},
    {"self_attn.q_b_proj.quant_bias", IN_Q_PROJ_B_BIAS},
    {"self_attn.q_b_proj.input_scale", IN_Q_PROJ_B_SCALE},
    {"self_attn.q_b_proj.deq_scale", IN_Q_PROJ_B_DESCALE},
    {"self_attn.q_b_proj.input_offset", IN_Q_PROJ_B_OFFSET},

    {"self_attn.kv_a_proj_with_mqa.weight", IN_KV_PROJ_WITH_MQA_WEIGHT},
    {"self_attn.kv_a_proj_with_mqa.quant_bias", IN_KV_PROJ_WITH_MQA_BIAS},
    {"self_attn.kv_a_proj_with_mqa.deq_scale", IN_KV_PROJ_WITH_MQA_DESCALE},
    {"self_attn.kv_a_proj_with_mqa.input_offset", IN_KV_PROJ_WITH_MQA_OFFSET},
    {"self_attn.kv_a_proj_with_mqa.input_scale", IN_KV_PROJ_WITH_MQA_SCALE},

    {"self_attn.kv_a_layernorm.weight", IN_KV_PROJ_A_LAYERNORM_WEIGHT},
    {"self_attn.kv_a_layernorm.bias", IN_KV_PROJ_A_LAYERNORM_BIAS},

    {"self_attn.kv_b_proj.weight", IN_K_PROJ_B_FOR_Q_WEIGHT},  // merge
    // {"self_attn.kv_b_proj.weight", IN_V_PROJ_B_FOR_O_WEIGHT},  // merge

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_ATTENTION_OUT_BIAS},
    {"self_attn.o_proj.deq_scale", IN_ATTENTION_OUT_DESCALE},
    {"self_attn.o_proj.input_offset", IN_ATTENTION_OUT_OFFSET},
    {"self_attn.o_proj.input_scale", IN_ATTENTION_OUT_SCALE},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_SELFATTENTION_OUT_NORM_BIAS},

    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.up_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.down_proj.weight_offset", IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.down_proj.weight_scale", IN_MLP_DOWN_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_offset",
     IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_scale",
     IN_MLP_DOWN_SCALE_SHARED_EXPERT},

    {"mlp.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", IN_BLOCK_SPARSE_MOE_GATE_BIAS},

    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},
    {"gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_EXPERT},
    {"gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_EXPERT},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},
    {"up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_EXPERT},
    {"up_proj.weight_scale", IN_MLP_GATEUP_SCALE_EXPERT},

    {"down_proj.weight", IN_MLP_DOWN_WEIGHT_EXPERT},
    {"down_proj.weight_offset", IN_MLP_DOWN_OFFSET_EXPERT},
    {"down_proj.weight_scale", IN_MLP_DOWN_SCALE_EXPERT},
};

static const std::map<int, int> WEIGHT_SHARD = {};

static const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_Q_PROJ_B_WEIGHT, 0},
    {IN_Q_PROJ_B_BIAS, 0},
    {IN_Q_PROJ_B_DESCALE, 0},
    {IN_K_PROJ_B_FOR_Q_WEIGHT, 0},
    {IN_V_PROJ_B_FOR_O_WEIGHT, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

static std::vector<int> SQUEEZE_WEIGHT_VEC = {
    IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
    IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
    IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
    IN_MLP_DOWN_SCALE_SHARED_EXPERT};

static std::vector<std::string> LINEAR_FOR_ROPE = {
    "self_attn.q_b_proj.weight",
    "self_attn.q_b_proj.quant_bias",
    "self_attn.q_b_proj.deq_scale",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_proj_with_mqa.quant_bias",
    "self_attn.kv_a_proj_with_mqa.deq_scale",
};

NpuDeepseekV2DecoderLayerImpl::NpuDeepseekV2DecoderLayerImpl(
    const Context& context,
    const int32_t layer_id,
    const float sm_scale)
    : NpuBaseLayer(context),
      device_id_(context.get_tensor_options().device().index()),
      layer_id_(layer_id),
      sm_scale_(sm_scale),
      num_speculative_tokens_(
          context.get_model_args().num_speculative_tokens()) {
  auto parallel_args = context.get_parallel_args();
  auto model_args = context.get_model_args();
  auto options = context.get_tensor_options();

  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.n_routed_experts() / ep_size_;
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_param_, model_args, parallel_args, false);

  initialize_tensors(options);
}

void NpuDeepseekV2DecoderLayerImpl::initialize_tensors(
    const torch::TensorOptions& options) {
  // initializ placeholder
  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  int_tensor_placeholder_ = torch::ones({1}).to(torch::kInt32).to(device_);
  slot_tensor_placeholder_ = torch::full({1}, 0).to(torch::kInt32).to(device_);
  block_tables_placeholder_ =
      torch::zeros({1, 1}).to(torch::kInt32).to(device_);
  tensor_placeholder_ = torch::zeros({1}).to(options);
  resize_experts_weights(prefill_param_.numOfDeviceExperts);
  expert_group_ = torch::arange(1024, torch::kInt32).to(device_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  at_start_expert_id_ =
      torch::tensor({start_expert_id_}, torch::kInt64).to(device_);
  at_in_device_expert_count_ =
      torch::tensor({num_experts_per_partition_ - 1}, torch::kInt64)
          .to(device_);
  initialize_weight_tensors(options);
}

void NpuDeepseekV2DecoderLayerImpl::param_from_args(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  initialize_basic_parameters(param, args, parallel_args, is_prefill);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
  initialize_kimi_k2_parameters(param, args, is_prefill);
}

void NpuDeepseekV2DecoderLayerImpl::resize_experts_weights(
    int num_of_device_experts) {
  experts_weights_["gate_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  if (quantize_type_ == "w8a8_dynamic") {
    experts_weights_["gate_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["up_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["down_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["gate_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["up_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["down_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
  }
}

void NpuDeepseekV2DecoderLayerImpl::initialize_weight_tensors(
    const torch::TensorOptions& options) {
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void NpuDeepseekV2DecoderLayerImpl::initialize_basic_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  param.isFA = false;
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.enableLcoc = false;

  param.attnLinearTransposeType = {1, 1, 1, 1, 1, 1};
  param.mlpLinearTransposeType = {1, -1, 1, -1};

  param.moeLinearTransposeType = (layer_id_ < args.first_k_dense_replace())
                                     ? std::vector<int>{-1, -1, -1, -1}
                                     : std::vector<int>{1, 0, -1, 1};

  param.worldSize = parallel_args.world_size();
  param.normEps = args.rms_norm_eps();
  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;
  param.hiddenSizePerAttentionHead = args.hidden_size() / args.n_heads();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank = 1;
  // static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = FLAGS_communication_backend;
  param.rankTableFile = FLAGS_rank_tablefile;

  param.layerId = layer_id_;
  param.numHiddenLayers = args.n_layers();
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  if (quantize_type_ == "") {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }
  param.enableDpOut = false;  // TODO
  if (num_speculative_tokens_ == 0) {
    param.enableSpeculate = false;  // MTP
  } else {
    param.enableSpeculate = true;
  }
  param.maskfree = true;                            // TODO
  param.enableSwiGLUQuantForSharedExperts = false;  // TODO

  num_key_value_heads_ = static_cast<int>(args.n_kv_heads().value());
  qk_nope_head_dim_ = args.qk_nope_head_dim();
  v_head_dim_ = args.v_head_dim();
  kv_lora_rank_ = args.kv_lora_rank();
  qk_rope_head_dim_ = args.qk_rope_head_dim();
}

void NpuDeepseekV2DecoderLayerImpl::initialize_attention_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.qLoraRank = args.q_lora_rank();
  // NOTE: The operation in this conditional is theoretically compatible with
  // DeepSeek, but we add this specific check to ensure DeepSeek behavior
  // remains unchanged
  if (args.model_type() != "kimi_k2") {
    param.headNum = args.n_heads();
  }
  param.qkNopeHeadDim = args.qk_nope_head_dim();
  param.qkRopeHeadDim = args.qk_rope_head_dim();
  param.kvLoraRank = args.kv_lora_rank();

  // sm_scale_ shows approximately 9 decimal places difference when compared
  // across different engines, which may cause minimal diff during the decode
  // phase
  param.softmaxScale = sm_scale_;
  if (quantize_type_ == "w8a8_dynamic" && num_speculative_tokens_ == 0) {
    param.enableMlaPreprocess = param.isBF16 ? false : true;
  } else {
    param.enableMlaPreprocess = false;
  }

  param.enableFA3 = false;           // TODO
  param.isNzCache = false;           // TODO
  param.enableKvQuantLayer = false;  // TODO
}

void NpuDeepseekV2DecoderLayerImpl::initialize_mlp_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  param.processLogits = "normScaling";
  param.routedScalingFactor = args.routed_scaling_factor();
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  if (ep_size_ > 1) {
    param.expertParallelDegree = std::max(FLAGS_expert_parallel_degree, 1);
  } else {
    param.expertParallelDegree = 0;
  }

  param.deviceExpert.resize(num_experts_per_partition_);
  // param.deviceExpert.resize(args.n_routed_experts());
  std::iota(
      param.deviceExpert.begin(), param.deviceExpert.end(), start_expert_id_);
  param.numOfExperts = args.n_routed_experts();
  param.numOfDeviceExperts = num_experts_per_partition_;
  param.maskStartIdx = 0;
  param.firstKDenseReplace = args.first_k_dense_replace();
  // param.numOfSharedExperts = args.n_shared_experts();
  param.numOfSharedExperts = 2;
  param.routingMethod = "noAuxTc";
  param.numOfGroups = args.n_group();
  param.topkGroups = atb::SVector<int>{args.topk_group()};
  param.isDynamicEp = param.expertParallelDegree == 2 ? true : false;

  param.quantGroupSize = 0;
  if (quantize_type_ == "") {
    param.enableInitQuant = false;
    param.enableSwigluQuant = false;
  } else {
    param.enableInitQuant = true;
    param.enableSwigluQuant = param.isPrefill && !param.enableGMMSwigluQuant;
  }
  param.enableFusedTopk = true;

  param.enableCVOverlap = false;           // TODO
  param.enableExpertCumSumOutput = false;  // TODO
  param.enableLoadBalance = false;         // TODO
  param.enableEPWB = false;                // TODO
  param.numOfRedundantExpert = 0;          // TODO
  param.enableInfNan = param.isPrefill;    // TODO

  param.dispatchAndCombineHcclComm = parallel_args.dispatchAndCombineHcclComm();
  param.dispatchAndCombinecommDomain =
      parallel_args.dispatchAndCombinecommDomain();

  if (layer_id_ >= param.firstKDenseReplace) {
    // param.enableQkvdownDp = (param.expertParallelDegree==1 &&
    // param.isPrefill) ? true:false;
    param.enableQkvdownDp = false;
    param.enableSharedExpertDp = false;  // TODO
    param.enableGatingDp = false;        // TODO
  }
  if (layer_id_ < param.firstKDenseReplace) {
    param.isDenseLayer = true;
  }
}

void NpuDeepseekV2DecoderLayerImpl::initialize_kimi_k2_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    bool is_prefill) {
  if (args.model_type() != "kimi_k2") {
    return;
  }
  // NOTE: These operations are theoretically applicable to DeepSeek as well,
  // but we only apply them to kimi_k2 to ensure DeepSeek behavior remains
  // unchanged
  param.enableInfNan = true;
  param.enableFusedTopk = (args.topk_method() == "noaux_tc" &&
                           args.n_group() * 32 >= args.n_routed_experts());
  param.maskfree = is_prefill;

  // TODO: Pending confirmation whether kimi_k2 model supports
  // enable_gmmswigluquant set to true
  bool enable_gmmswigluquant = false;
  param.enableSwigluQuant =
      quantize_type_ == "w8a8_dynamic" && !enable_gmmswigluquant;
  param.enableGMMSwigluQuant = enable_gmmswigluquant;
}

void NpuDeepseekV2DecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.lmHeadLocalTp = dp_local_tp_size_;
  param.enableSharedExpertOverlap = false;  // TODO

  param.enableAllToAllMC2 = (param.expertParallelDegree == 2);
  param.enableGatherPreNorm = true;
  param.enableExtraOprojTp = false;  // TODO
  param.isMlpFullTP = false;         // TODO
  param.mapping = parallel_args.mapping();
  param.maxDecodeDpTokenSize = 0;  // TODO
}

void NpuDeepseekV2DecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param) {
  if (quantize_type_ == "") {
    param.moePackQuantType = static_cast<int>(PackType::ALL_FP);
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID)};
    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::FP)};
    }
  } else {
    param.moePackQuantType = static_cast<int>(PackType::ALL_W8A8_DYNAMIC);
    param.packQuantType = {static_cast<int>(PackType::MIX_W8A8),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INT)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID)};
    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INT)};
    }
  }
}

void NpuDeepseekV2DecoderLayerImpl::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [name, tensor] : state_dict) {
    bool is_sharded = false;
    int index = 0;

    if (absl::EndsWith(name, "self_attn.kv_b_proj.weight")) {
      index = WEIGHT_MAPPING_W8A8.at(name);
      set_kv_weight(state_dict, name, index, WEIGHT_SHARD_W8A8.at(index));
      continue;
    }

    if (absl::StartsWith(name, "mlp.experts")) {
      process_expert_weights(state_dict, name, tensor);
      continue;
    }

    if (absl::StartsWith(name, "mlp.shared_experts")) {
      process_shared_expert_weights(state_dict, name, tensor);
      continue;
    }

    if (absl::StartsWith(name, "mlp") && !absl::StrContains(name, "gate.")) {
      process_mlp_common_weights(state_dict, name, tensor);
      continue;
    }

    process_general_weights(state_dict, name, tensor);
  }
}

int NpuDeepseekV2DecoderLayerImpl::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(WARNING) << "Parameter '" << name
                 << "' not found in mapping and will not be used.";
    return -1;
  }
  return it->second;
}

void NpuDeepseekV2DecoderLayerImpl::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  int expert_index = extract_expert_index(name);
  if (expert_index < start_expert_id_ || expert_index > end_expert_id_) {
    return;
  }

  const std::string suffix = extract_endswith(name);
  const int index = get_mapped_index(suffix, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  const int local_index = expert_index % num_experts_per_partition_;
  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);

  std::lock_guard<std::mutex> lock(experts_mutex_);
  torch::Tensor tmp_tensor =
      is_sharded ? get_sharded_tensor(state_dict,
                                      name,
                                      WEIGHT_SHARD_W8A8.at(index),
                                      ep_local_tp_rank_,
                                      ep_local_tp_size_)
                 : tensor;

  experts_weights_[suffix][local_index] = tmp_tensor.clone();
}

void NpuDeepseekV2DecoderLayerImpl::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  torch::Tensor tmp_tensor;
  std::lock_guard<std::mutex> lock(shared_experts_mutex_);
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  if (FLAGS_expert_parallel_degree == 2) {
    tmp_tensor = tensor.to(device_);
  } else {
    const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
    tmp_tensor = is_sharded ? get_sharded_tensor(
                                  state_dict, name, WEIGHT_SHARD_W8A8.at(index))
                                  .to(device_)
                            : tensor.to(device_);
  }
  if (absl::StrContains(name, "down_proj")) {
    at_weight_tensors_[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void NpuDeepseekV2DecoderLayerImpl::process_mlp_common_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
  std::lock_guard<std::mutex> lock(shared_experts_mutex_);

  torch::Tensor tmp_tensor =
      is_sharded ? get_sharded_tensor(state_dict,
                                      name,
                                      WEIGHT_SHARD_W8A8.at(index),
                                      dp_local_tp_rank_,
                                      dp_local_tp_size_)
                       .to(device_)
                 : tensor.to(device_);
  if (absl::StrContains(name, "down_proj")) {
    at_weight_tensors_[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void NpuDeepseekV2DecoderLayerImpl::process_general_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
  torch::Tensor tmp_tensor;

  tmp_tensor = is_sharded ? get_sharded_tensor(state_dict,
                                               name,
                                               WEIGHT_SHARD_W8A8.at(index),
                                               dp_local_tp_rank_,
                                               dp_local_tp_size_)
                                .to(device_)
                          : tensor.to(device_);

  correct_tensor_dtype(tmp_tensor, name);
  at_weight_tensors_[index] = tmp_tensor;
}

void NpuDeepseekV2DecoderLayerImpl::set_kv_weight(
    const StateDict& state_dict,
    const std::string& tensor_name,
    int weight_position,
    int dim) {
  torch::Tensor mutable_tensor;
  if (parallel_args_.world_size() <= 1) {
    mutable_tensor = state_dict.get_tensor(tensor_name).to(device_);
    correct_tensor_dtype(mutable_tensor, tensor_name);
  } else {
    mutable_tensor =
        get_sharded_tensor(
            state_dict, tensor_name, dim, dp_local_tp_rank_, dp_local_tp_size_)
            .to(device_);
    // mutable_tensor = get_sharded_tensor(state_dict, tensor_name, dim);
    correct_tensor_dtype(mutable_tensor, tensor_name);
  }

  torch::Tensor kv_b_proj_weight =
      mutable_tensor.reshape({num_key_value_heads_ / dp_local_tp_size_,
                              qk_nope_head_dim_ + v_head_dim_,
                              kv_lora_rank_});
  torch::Tensor k_b_proj_preprocessed =
      kv_b_proj_weight.slice(1, 0, qk_nope_head_dim_).contiguous();
  torch::Tensor v_b_proj_preprocessed =
      kv_b_proj_weight
          .slice(1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_)
          .transpose(1, 2)
          .contiguous();
  at_weight_tensors_[weight_position] = k_b_proj_preprocessed.to(device_);
  at_weight_tensors_[weight_position + 6] = v_b_proj_preprocessed.to(device_);
}

void NpuDeepseekV2DecoderLayerImpl::preprocess_linear_for_rope() {
  for (const auto& name : LINEAR_FOR_ROPE) {
    if (quantize_type_ == "") {
      if (!absl::EndsWith(name, "weight")) {
        continue;
      }
    }
    int index = WEIGHT_MAPPING_W8A8.at(name);
    at_weight_tensors_[index] =
        view_tensor(at_weight_tensors_[index], name, true);
    at_weight_tensors_[index] = trans_rope_weight(at_weight_tensors_[index]);
    at_weight_tensors_[index] =
        (!absl::EndsWith(name, "weight"))
            ? view_tensor(at_weight_tensors_[index], name, false).flatten()
            : view_tensor(at_weight_tensors_[index], name, false);
  }
}

torch::Tensor NpuDeepseekV2DecoderLayerImpl::view_tensor(
    torch::Tensor weight,
    const std::string& name,
    bool pre_view) {
  if (absl::StrContains(name, "q_b_proj")) {
    if (pre_view) {
      return weight
          .view({prefill_param_.numAttentionHeadsPerRank,
                 qk_nope_head_dim_ + prefill_param_.qkRopeHeadDim,
                 -1})
          .contiguous();
    } else {
      return weight
          .view({prefill_param_.numAttentionHeadsPerRank *
                     (qk_nope_head_dim_ + prefill_param_.qkRopeHeadDim),
                 -1})
          .contiguous();
    }
  } else if (absl::StrContains(name, "kv_a_proj_with_mqa")) {
    return weight.view({kv_lora_rank_ + prefill_param_.qkRopeHeadDim, -1})
        .contiguous();
  }
  return weight;
}

torch::Tensor NpuDeepseekV2DecoderLayerImpl::trans_rope_weight(
    torch::Tensor weight) {
  int64_t d = weight.size(-2);
  int64_t rope_dim = prefill_param_.qkRopeHeadDim;
  torch::Tensor weight_1 =
      weight.slice(-2, d - rope_dim, torch::indexing::None, 2).contiguous();

  torch::Tensor weight_2 =
      weight.slice(-2, d - rope_dim + 1, torch::indexing::None, 2).contiguous();

  torch::Tensor combined = torch::cat({weight_1, weight_2}, -2);

  weight.slice(-2, d - rope_dim, d).copy_(combined);

  return weight.contiguous();
}

torch::Tensor NpuDeepseekV2DecoderLayerImpl::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim) {
  if (parallel_args_.world_size() > 1) {
    return state_dict.get_sharded_tensor(
        name, dim, parallel_args_.rank(), parallel_args_.world_size());
  } else {
    return state_dict.get_tensor(name);
  }
}

torch::Tensor NpuDeepseekV2DecoderLayerImpl::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim,
    int loacal_tp_rank,
    int local_tp_size) {
  if (local_tp_size > 1) {
    return state_dict.get_sharded_tensor(
        name, dim, loacal_tp_rank, local_tp_size);
  } else {
    return state_dict.get_tensor(name);
  }
}

std::string NpuDeepseekV2DecoderLayerImpl::extract_endswith(
    const std::string& input) {
  std::vector<std::string> parts;
  std::stringstream ss(input);
  std::string part;
  while (std::getline(ss, part, '.')) {
    parts.push_back(part);
  }
  if (parts.size() < 2) {
    return "";
  }
  std::string result = parts[parts.size() - 2] + "." + parts[parts.size() - 1];
  return result;
}

int NpuDeepseekV2DecoderLayerImpl::extract_expert_index(
    const std::string& name) {
  std::string prefix = "experts.";
  size_t pos = name.find(prefix);
  if (pos != std::string::npos) {
    pos += prefix.length();
    size_t end_pos = pos;
    while (end_pos < name.length() && std::isdigit(name[end_pos])) {
      ++end_pos;
    }
    if (end_pos > pos) {
      return std::stoi(name.substr(pos, end_pos - pos));
    }
  }
  return -1;
}

void NpuDeepseekV2DecoderLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << prefix + name;
  }
}

void NpuDeepseekV2DecoderLayerImpl::merge_loaded_weights() {
  if (quantize_type_ == "w8a8_dynamic") {
    if (prefill_param_.isBF16) {
      convert_descaled_weights_to_float();
    }
    convert_offsets_to_int8();
    handle_device_specific_bias();
  }

  merge_shared_experts_weights();
  if (layer_id_ >= prefill_param_.firstKDenseReplace) {
    merge_experts_weights();
  }
  squeeze_experts_weights();
  preprocess_linear_for_rope();

  at_weight_tensors_[IN_Q_PROJ_A_WEIGHT] =
      torch::cat({at_weight_tensors_[IN_KV_PROJ_WITH_MQA_WEIGHT],
                  at_weight_tensors_[IN_Q_PROJ_A_WEIGHT]},
                 0)
          .contiguous();
  if (quantize_type_ == "w8a8_dynamic") {
    at_weight_tensors_[IN_Q_PROJ_A_BIAS] =
        torch::cat({at_weight_tensors_[IN_KV_PROJ_WITH_MQA_BIAS],
                    at_weight_tensors_[IN_Q_PROJ_A_BIAS]},
                   0)
            .contiguous();
    at_weight_tensors_[IN_Q_PROJ_A_DESCALE] =
        torch::cat({at_weight_tensors_[IN_KV_PROJ_WITH_MQA_DESCALE],
                    at_weight_tensors_[IN_Q_PROJ_A_DESCALE]},
                   0)
            .contiguous();
  }

  at_weight_tensors_[IN_Q_PROJ_A_WEIGHT] = at_npu::native::npu_format_cast(
      at_weight_tensors_[IN_Q_PROJ_A_WEIGHT], 29);
  at_weight_tensors_[IN_Q_PROJ_B_WEIGHT] = at_npu::native::npu_format_cast(
      at_weight_tensors_[IN_Q_PROJ_B_WEIGHT], 29);

  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_WEIGHT] = tensor_placeholder_;
  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_BIAS] = tensor_placeholder_;
  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_DESCALE] = tensor_placeholder_;
  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_OFFSET] = tensor_placeholder_;
  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_SCALE] = tensor_placeholder_;
  if (FLAGS_expert_parallel_degree != 2) {
    at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT] =
        torch::roll(at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT],
                    {-1 * ep_rank_ * num_experts_per_partition_},
                    {0})
            .contiguous();
    at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_BIAS] =
        torch::roll(at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_BIAS],
                    {-1 * ep_rank_ * num_experts_per_partition_},
                    {0})
            .contiguous();
  }
  // at_weight_tensors_[IN_MLP_DOWN_WEIGHT_SHARED_EXPERT] =
  // at_weight_tensors_[IN_MLP_DOWN_WEIGHT_SHARED_EXPERT].transpose(0, 1);
  if (!prefill_param_.isBF16 && quantize_type_ == "w8a8_dynamic") {
    at_weight_tensors_[IN_Q_PROJ_A_DESCALE] =
        convert_fp16_to_int64(at_weight_tensors_[IN_Q_PROJ_A_DESCALE]);
    at_weight_tensors_[IN_Q_PROJ_B_DESCALE] =
        convert_fp16_to_int64(at_weight_tensors_[IN_Q_PROJ_B_DESCALE]);
    at_weight_tensors_[IN_ATTENTION_OUT_DESCALE] =
        convert_fp16_to_int64(at_weight_tensors_[IN_ATTENTION_OUT_DESCALE]);

    at_weight_tensors_[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT] =
        at_weight_tensors_[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT].to(
            torch::kFloat16);
    at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
        at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].to(
            torch::kFloat32);
    at_weight_tensors_[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
        at_weight_tensors_[IN_MLP_DOWN_SCALE_SHARED_EXPERT].to(torch::kFloat32);
    at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT] =
        at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT].to(torch::kFloat32);
    at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT] =
        at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT].to(torch::kFloat16);
    at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT] =
        at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT].to(torch::kFloat32);
    at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT] =
        at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT].to(torch::kFloat16);
    at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] =
        at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT].to(torch::kFloat32);
  }
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }
  init_layer();
}

torch::Tensor NpuDeepseekV2DecoderLayerImpl::convert_fp16_to_int64(
    const torch::Tensor& fp16_tensor) {
  auto float_tensor = fp16_tensor.to(torch::kFloat32);
  auto int32_tensor = float_tensor.view(torch::kInt32);
  auto int64_tensor = int32_tensor.to(torch::kInt64);
  return int64_tensor;
}

void NpuDeepseekV2DecoderLayerImpl::convert_descaled_weights_to_float() {
  auto convert_to_float = [this](int index) {
    at_weight_tensors_[index] = at_weight_tensors_[index].to(torch::kFloat32);
  };
  convert_to_float(IN_Q_PROJ_A_DESCALE);
  convert_to_float(IN_Q_PROJ_B_DESCALE);
  convert_to_float(IN_KV_PROJ_WITH_MQA_DESCALE);
  convert_to_float(IN_ATTENTION_OUT_DESCALE);
}

void NpuDeepseekV2DecoderLayerImpl::convert_offsets_to_int8() {
  auto convert_to_int8 = [this](int index) {
    at_weight_tensors_[index] =
        at_weight_tensors_[index].to(torch::kInt8).to(device_);
  };
  convert_to_int8(IN_Q_PROJ_A_OFFSET);
  convert_to_int8(IN_Q_PROJ_B_OFFSET);
  convert_to_int8(IN_KV_PROJ_WITH_MQA_OFFSET);
  convert_to_int8(IN_ATTENTION_OUT_OFFSET);
}

void NpuDeepseekV2DecoderLayerImpl::handle_device_specific_bias() {
  if (dp_local_tp_rank_ != 0) {
    torch::Tensor original_tensor = at_weight_tensors_[IN_ATTENTION_OUT_BIAS];
    at_weight_tensors_[IN_ATTENTION_OUT_BIAS] =
        torch::zeros(original_tensor.sizes(),
                     torch::TensorOptions()
                         .dtype(original_tensor.dtype())
                         .device(original_tensor.device()));
  }
}

void NpuDeepseekV2DecoderLayerImpl::merge_shared_experts_weights() {
  auto merge_and_clear = [this](int index,
                                torch::Tensor& shared_experts_gate,
                                torch::Tensor& shared_experts_up) {
    at_weight_tensors_[index] =
        torch::cat({shared_experts_gate, shared_experts_up}, 0)
            .to(device_)
            .contiguous();
    shared_experts_gate = tensor_placeholder_;
    shared_experts_up = tensor_placeholder_;
  };

  if (layer_id_ >= prefill_param_.firstKDenseReplace) {
    merge_and_clear(
        IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
        shared_experts_weights_["mlp.shared_experts.gate_proj.weight"],
        shared_experts_weights_["mlp.shared_experts.up_proj.weight"]);
    if (quantize_type_ == "w8a8_dynamic") {
      merge_and_clear(
          IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
          shared_experts_weights_["mlp.shared_experts.gate_proj.weight_offset"],
          shared_experts_weights_["mlp.shared_experts.up_proj.weight_offset"]);
      merge_and_clear(
          IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
          shared_experts_weights_["mlp.shared_experts.gate_proj.weight_scale"],
          shared_experts_weights_["mlp.shared_experts.up_proj.weight_scale"]);
    }
  } else {
    merge_and_clear(IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
                    shared_experts_weights_["mlp.gate_proj.weight"],
                    shared_experts_weights_["mlp.up_proj.weight"]);
    if (quantize_type_ == "w8a8_dynamic") {
      merge_and_clear(IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
                      shared_experts_weights_["mlp.gate_proj.weight_offset"],
                      shared_experts_weights_["mlp.up_proj.weight_offset"]);
      merge_and_clear(IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
                      shared_experts_weights_["mlp.gate_proj.weight_scale"],
                      shared_experts_weights_["mlp.up_proj.weight_scale"]);
    }
  }
}

void NpuDeepseekV2DecoderLayerImpl::merge_experts_weights() {
  torch::Tensor mlp_gateup_weight =
      merge_experts_weights(experts_weights_["gate_proj.weight"],
                            experts_weights_["up_proj.weight"],
                            /*transpose=*/true);
  at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT] =
      at_npu::native::npu_format_cast(mlp_gateup_weight, 29);
  // at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT] =
  //     at_npu::native::npu_format_cast(mlp_gateup_weight, 2).contiguous();
  if (quantize_type_ == "w8a8_dynamic") {
    at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                              experts_weights_["up_proj.weight_offset"]);
    at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                              experts_weights_["up_proj.weight_scale"]);
  }

  torch::Tensor mlp_down_weight =
      merge_experts_weights(experts_weights_["down_proj.weight"],
                            /*transpose=*/false);
  // at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
  //     at_npu::native::npu_format_cast(mlp_down_weight, 29);
  at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
      at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();
  if (quantize_type_ == "w8a8_dynamic") {
    at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT] =
        merge_experts_weights(experts_weights_["down_proj.weight_offset"]);
    at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] =
        merge_experts_weights(experts_weights_["down_proj.weight_scale"]);
  }
}

torch::Tensor NpuDeepseekV2DecoderLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    bool transpose) {
  torch::Tensor merged_tensor = torch::stack(experts, 0).to(device_);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts.clear();
  return merged_tensor;
}

torch::Tensor NpuDeepseekV2DecoderLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose) {
  for (size_t i = 0; i < experts_up.size(); ++i) {
    experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
  }
  torch::Tensor merged_tensor = torch::stack(experts_gate, 0).to(device_);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts_gate.clear();
  experts_up.clear();
  return merged_tensor;
}

void NpuDeepseekV2DecoderLayerImpl::squeeze_experts_weights() {
  for (const auto& index : SQUEEZE_WEIGHT_VEC) {
    if (at_weight_tensors_[index].dim() > 1) {
      at_weight_tensors_[index] = at_weight_tensors_[index].squeeze();
    }
  }
}

int64_t NpuDeepseekV2DecoderLayerImpl::init_layer() {
  name_ = "deepseek_v2_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "DeepSeek_V2";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));
  return atb::NO_ERROR;
}

int64_t NpuDeepseekV2DecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::deepseekV2::DecoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::deepseekV2::DecoderLayer(param, &operation);
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
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  return atb::NO_ERROR;
}

torch::Tensor NpuDeepseekV2DecoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    atb::Context* context,
    AtbWorkspace& workspace,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    int node_id) {
  atb::Status st;
  if (input_params.global_empty_kv_cache) {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            true);
    st = execute_node(
        prefill_node_, context, workspace, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  } else {
    build_node_variant_pack(decode_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            /*attn_mask*/ tensor_placeholder_,
                            kv_cache,
                            input_params,
                            false);
    st = execute_node(
        decode_node_, context, workspace, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }
  return tensor_placeholder_;
}

void NpuDeepseekV2DecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensor_ = atb_speed::Utils::AtTensor2Tensor(x);
  // final_hidden_states_ = torch::zeros_like(x);
  int32_t input_idx = 0;
  auto& dp_ep_padding = input_params.dp_ep_padding_data;

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensor_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.expert_array());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());

  if (!input_params.block_tables.defined() ||
      input_params.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        const_cast<int32_t*>(placeholder_vec_.data());
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        const_cast<int32_t*>(input_params.kv_seq_lens_vec.data());
  }

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  if (!input_params.block_tables.defined() ||
      input_params.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(block_tables_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
  }
  if (!input_params.block_tables.defined() ||
      input_params.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 16) =
        atb_speed::Utils::AtTensor2Tensor(slot_tensor_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 16) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  }
  if (num_speculative_tokens_ > 0 && !is_prefill) {
    if (!input_params.block_tables.defined() ||
        input_params.block_tables.storage().data() == nullptr) {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
          atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17).hostData =
          const_cast<int32_t*>(placeholder_vec_.data());
    } else {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
          atb_speed::Utils::AtTensor2Tensor(input_params.q_seq_lens);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17).hostData =
          const_cast<int32_t*>(input_params.q_seq_lens_vec.data());
    }
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
        atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  }

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 18) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.attn_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 19) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.attn_unpadding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 20) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.ffn_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 21) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.ffn_unpadding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 22) =
      atb_speed::Utils::AtTensor2Tensor(
          dp_ep_padding.lm_head_skip_padding_token_indices());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 23) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.gather_prenorm_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 24) =
      atb_speed::Utils::AtTensor2Tensor(at_start_expert_id_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 25) =
      atb_speed::Utils::AtTensor2Tensor(at_in_device_expert_count_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 26) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 27) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.un_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 28) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.dynamic_ep_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 29) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.moe_idx());

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensor_;
}

}  // namespace layer
}  // namespace xllm
