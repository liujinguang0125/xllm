#include "attention_mask.h"

namespace xllm::layer {

AttentionMask::AttentionMask(at::Device device,
                             torch::Dtype dtype,
                             float mask_value) {
  int max_seq_len = 128;
  seq_len_cached_ = max_seq_len;
  auto bias_cache =
      torch::tril(torch::ones({max_seq_len, max_seq_len}, torch::kBool))
          .view({max_seq_len, max_seq_len});
  bias_cache = ~bias_cache;
  if (dtype == torch::kFloat16) {
    mask_value_ = -std::numeric_limits<float>::infinity();
  } else {
    mask_value_ = mask_value;
  }
  atten_mask_cache_ = torch::zeros({max_seq_len, max_seq_len})
                          .masked_fill(bias_cache, mask_value_)
                          .to(device);
}

torch::Tensor AttentionMask::get_decode_attn_mask(torch::Tensor input_lengths,
                                                  int64_t max_s,
                                                  torch::Dtype dtype,
                                                  torch::Device device) {
  update_attn_cache(dtype, device, max_s);
  return atten_mask_cache_.index_select(0, input_lengths).view({-1, 1, max_s});
}

torch::Tensor AttentionMask::get_attn_mask(int64_t max_s,
                                           torch::Dtype dtype,
                                           torch::Device device) {
  update_attn_cache(dtype, device, max_s);
  return atten_mask_cache_.slice(0, 0, max_s).slice(1, 0, max_s);
}

torch::Tensor AttentionMask::gen_free_mask(int32_t q_len,
                                           torch::Dtype dtype,
                                           torch::Device device) {
  float pre_mask_factor = -10000.0f;
  if (dtype == torch::kBFloat16) {
    pre_mask_factor = 1.0f;
  }

  auto mask_options = torch::TensorOptions().dtype(dtype).device(device);
  auto mask_free =
      torch::full({125 + 2 * q_len, 128}, pre_mask_factor, mask_options);
  mask_free = torch::triu(mask_free, 2 - q_len);
  return mask_free;
}

void AttentionMask::update_attn_cache(torch::Dtype dtype,
                                      torch::Device device,
                                      int64_t seqlen) {
  if (seqlen > seq_len_cached_ || atten_mask_cache_.dtype() != dtype) {
    seq_len_cached_ = seqlen;

    auto options = torch::TensorOptions().dtype(torch::kBool).device(device);
    auto bias_cache = torch::tril(torch::ones({seqlen, seqlen}, options));
    bias_cache = ~bias_cache;

    auto mask_options = torch::TensorOptions().dtype(dtype).device(device);
    auto mask_atten_cache = torch::zeros({seqlen, seqlen}, mask_options)
                                .masked_fill(bias_cache, mask_value_);
    atten_mask_cache_ = mask_atten_cache;
  }
}

}  // namespace xllm::layer