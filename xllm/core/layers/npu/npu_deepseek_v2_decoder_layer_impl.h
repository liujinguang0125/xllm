#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>

#include "framework/context.h"
#include "framework/model/model_input_params.h"
#include "framework/model/npu_dp_ep_padding.h"
#include "framework/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "npu_base_layer.h"
#include "xllm_kernels/models/deepseekv2/layer/decoder_layer.h"

namespace xllm {
namespace layer {

class NpuDeepseekV2DecoderLayerImpl : public NpuBaseLayer {
 public:
  explicit NpuDeepseekV2DecoderLayerImpl(const Context& context,
                                         const int32_t layer_id,
                                         const float sm_scale);

  ~NpuDeepseekV2DecoderLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& prefix) const;

  virtual void merge_loaded_weights() override;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);

 private:
  struct ShardingConfig {
    bool is_sharded;
    int index;
    bool use_dp_sharding = false;
  };

  void initialize_tensors(const torch::TensorOptions& options);

  void initialize_weight_tensors(const torch::TensorOptions& options);

  void param_from_args(atb_speed::deepseekV2::DecoderLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool is_prefill);

  void resize_experts_weights(int num_of_device_experts);

  void initialize_basic_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args,
      bool is_prefill);

  void initialize_attention_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args);

  void initialize_mlp_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args);

  void initialize_parallel_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ParallelArgs& parallel_args);

  void initialize_quantization_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param);

  void initialize_kimi_k2_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      bool is_prefill);

  torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                   const std::string& name,
                                   int dim);

  torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                   const std::string& name,
                                   int dim,
                                   int local_tp_rank,
                                   int local_tp_size);

  std::string extract_endswith(const std::string& input);

  void set_kv_weight(const StateDict& state_dict,
                     const std::string& tensor_name,
                     int weight_position,
                     int dim);

  int extract_expert_index(const std::string& name);

  void convert_descaled_weights_to_float();

  void convert_offsets_to_int8();

  torch::Tensor convert_fp16_to_int64(const torch::Tensor& fp16_tensor);

  void handle_device_specific_bias();

  void merge_shared_experts_weights();

  void merge_experts_weights();

  void squeeze_experts_weights();

  void preprocess_linear_for_rope();

  void process_expert_weights(const StateDict& state_dict,
                              const std::string& name,
                              const torch::Tensor& tensor);

  void process_shared_expert_weights(const StateDict& state_dict,
                                     const std::string& name,
                                     const torch::Tensor& tensor);

  void process_mlp_common_weights(const StateDict& state_dict,
                                  const std::string& name,
                                  const torch::Tensor& tensor);

  void process_general_weights(const StateDict& state_dict,
                               const std::string& name,
                               const torch::Tensor& tensor);

  int get_mapped_index(const std::string& name,
                       const std::unordered_map<std::string, int>& mapping);

  torch::Tensor view_tensor(torch::Tensor weight,
                            const std::string& name,
                            bool pre_view);

  torch::Tensor trans_rope_weight(torch::Tensor weight);

  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts,
                                      bool transpose = false);

  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts_up,
                                      std::vector<torch::Tensor>& experts_gate,
                                      bool transpose = false);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::deepseekV2::DecoderLayerParam& param);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               const ModelInputParams& input_params,
                               bool is_prefill);

  torch::Tensor block_tables_placeholder_;
  std::string model_name_;

  int32_t device_id_;
  int32_t layer_id_;
  int32_t num_key_value_heads_;
  int32_t qk_nope_head_dim_;
  int32_t v_head_dim_;
  int32_t kv_lora_rank_;
  int32_t qk_rope_head_dim_;

  int32_t ep_size_;
  int32_t num_experts_;
  int32_t num_experts_per_partition_;
  int32_t ep_local_tp_size_;
  int32_t ep_local_tp_rank_;
  int32_t start_expert_id_;
  int32_t end_expert_id_;
  int32_t ep_rank_;

  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t dp_rank_;
  int32_t dp_local_tp_rank_;

  float sm_scale_;
  int32_t num_speculative_tokens_ = 0;
  atb_speed::deepseekV2::DecoderLayerParam prefill_param_;
  atb_speed::deepseekV2::DecoderLayerParam decode_param_;

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;

  atb::Tensor internal_tensor_;

  torch::Tensor tensor_placeholder_;
  torch::Tensor slot_tensor_placeholder_;
  torch::Tensor int_tensor_placeholder_;
  torch::Tensor decode_attn_mask_;
  torch::Tensor expert_group_;
  torch::Tensor one_hot_;
  torch::Tensor zero_hot_;
  torch::Tensor final_hidden_states_;
  torch::Tensor at_start_expert_id_;
  torch::Tensor at_in_device_expert_count_;

  std::vector<int32_t> int_placeholder_;

  std::unordered_map<std::string, torch::Tensor> shared_experts_weights_;
  std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;

  std::mutex shared_experts_mutex_;
  std::mutex experts_mutex_;
};

std::vector<torch::Tensor> get_dtp_inputs(torch::Tensor token_size_per_dp_group,
                                          int32_t dp_local_tp_size,
                                          int32_t dp_rank,
                                          int32_t dp_size,
                                          int32_t rank,
                                          at::Device device);
}  // namespace layer
}  // namespace xllm
