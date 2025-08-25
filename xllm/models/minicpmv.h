#pragma once

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <regex>
#include <unordered_map>

#include "core/framework/context.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/multi_head_attention.h"
#include "core/layers/siglip_encoder_layer.h"
#include "model_registry.h"
#include "processors/input_processor.h"
#include "processors/minicpmv_image_processor.h"
#include "processors/pywarpper_image_processor.h"
#include "qwen2.h"
#include "xllm_kernels/core/include/atb_speed/log.h"

namespace xllm::hf {

class MiniCPMInputProcessor : public InputProcessor {
 public:
  MiniCPMInputProcessor(const ModelArgs& args) {
    image_feature_size_ = args.mm_image_feature_size();
    max_slice_nums_ = args.vision_max_slice_nums();
    slice_mode_ = args.mm_slice_mode();
    use_image_id_ = args.mm_use_image_id();
    scale_resolution_ = args.mm_scale_resolution();
  }

  void process(std::string& prompt, const MMData& mm_data) override {
    auto image_sizes = mm_data.get_tensor_vec("image_sizes");

    const std::regex pattern(R"(\(<image>[\s\S]*?</image>\))");

    std::sregex_iterator image_tag_begin(prompt.begin(), prompt.end(), pattern);
    std::sregex_iterator image_tag_end;

    if (image_tag_begin == image_tag_end) {
      return;
    }

    std::vector<std::pair<int, int>> image_size_list;
    image_size_list.reserve(image_sizes.size());
    for (auto& image_size : image_sizes) {
      if (image_size.dim() != 1 || image_size.size(0) != 2) {
        const auto& sizes = image_size.sizes();
        LOG(FATAL) << "image_size must be a 1D tensor with 2 "
                      "elements representing height and width;"
                      "now sizes: "
                   << sizes;
      }
      image_size_list.emplace_back(
          std::make_pair(image_size[0].item<int>(), image_size[1].item<int>()));
    }

    std::vector<std::string> text_chunks;
    size_t last_pos = 0;

    for (auto it = image_tag_begin; it != image_tag_end; ++it) {
      auto match = *it;
      text_chunks.push_back(
          prompt.substr(last_pos, match.position() - last_pos));
      last_pos = match.position() + match.length();
    }

    text_chunks.push_back(prompt.substr(last_pos));

    std::string new_prompt;
    for (size_t i = 0; i < image_size_list.size(); ++i) {
      new_prompt += text_chunks[i];
      new_prompt += get_slice_image_placeholder(image_size_list[i], i);
    }

    new_prompt += text_chunks.back();
    prompt = new_prompt;
  }

 private:
  std::string get_image_id_placeholder(int idx) const {
    return im_id_start_ + std::to_string(idx) + im_id_end_;
  }

  std::string get_grid_placeholder(const std::pair<int, int>& grid) const {
    if (grid.first == 0 || grid.second == 0) {
      return "";
    }

    // Prepare the slice placeholder
    std::string slice_placeholder = slice_start_token_;

    // Append the repeated unk_token_
    for (int i = 0; i < image_feature_size_; ++i) {
      slice_placeholder += unk_token_;
    }

    slice_placeholder += slice_end_token_;

    // Use a string to accumulate the result
    std::string grid_placeholder;

    // Loop over the grid and append placeholders
    for (int i = 0; i < grid.second; ++i) {     // Iterate through rows
      for (int j = 0; j < grid.first; ++j) {    // Iterate through columns
        grid_placeholder += slice_placeholder;  // Append the placeholder
      }
      if (i < grid.second - 1) {
        grid_placeholder +=
            "\n";  // Add a newline after each row except the last one
      }
    }

    return grid_placeholder;
  }

  std::string get_slice_image_placeholder(
      const std::pair<int, int>& image_size,
      int image_idx = 0,
      int max_slice_nums = -1,
      std::optional<bool> use_image_id_opt = std::nullopt) const {
    if (max_slice_nums < 0) {
      max_slice_nums = max_slice_nums_;
    }

    bool use_image_id =
        use_image_id_opt.has_value() ? use_image_id_opt.value() : use_image_id_;

    assert(max_slice_nums > 0);

    auto grid = MiniCPMVImageProcessor::get_sliced_grid(
        image_size, max_slice_nums, scale_resolution_);

    std::string image_placeholder = im_start_token_;

    for (int i = 0; i < image_feature_size_; ++i) {
      image_placeholder += unk_token_;
    }

    image_placeholder += im_end_token_;

    std::string final_placeholder;

    if (use_image_id) {
      final_placeholder =
          get_image_id_placeholder(image_idx) + image_placeholder;
    } else {
      final_placeholder = image_placeholder;
    }

    if (slice_mode_) {
      final_placeholder += get_grid_placeholder(grid);
    }

    return final_placeholder;
  }

 private:
  const std::string im_start_token_ = "<image>";
  const std::string im_end_token_ = "</image>";
  const std::string slice_start_token_ = "<slice>";
  const std::string slice_end_token_ = "</slice>";
  const std::string unk_token_ = "<unk>";
  const std::string im_id_start_ = "<image_id>";
  const std::string im_id_end_ = "</image_id>";

  bool slice_mode_;
  bool use_image_id_;
  int max_slice_nums_;
  int image_feature_size_;
  int scale_resolution_;
};

class BaseResamplerImpl : public torch::nn::Module {
 public:
  BaseResamplerImpl(const Context& context)
      : num_queries_(context.get_model_args().query_num()),
        embed_dim_(context.get_model_args().hidden_size()),
        num_heads_(context.get_model_args().n_heads()),
        kv_dim_(context.get_model_args().mm_hidden_size()) {
    auto options = context.get_tensor_options();
    // Initialize learnable query parameter
    query_ =
        register_parameter("query", torch::zeros({num_queries_, embed_dim_}));
    trunc_normal(query_, 0.02);
    query_.set_data(query_.to(options));
    ln_q_ = register_module(
        "ln_q",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})
                                 .elementwise_affine(true)
                                 .eps(1e-6)));
    ln_q_->weight.set_data(ln_q_->weight.to(options));
    ln_q_->bias.set_data(ln_q_->bias.to(options));
    ln_kv_ = register_module(
        "ln_kv",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})
                                 .elementwise_affine(true)
                                 .eps(1e-6)));
    ln_kv_->weight.set_data(ln_kv_->weight.to(options));
    ln_kv_->bias.set_data(ln_kv_->bias.to(options));
    // Initialize attention module
    attn_ = layer::MultiheadAttention(context);
    options_ = options;
    // Optionally add post projection
    ln_post_ = register_module(
        "ln_post",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})
                                 .elementwise_affine(true)
                                 .eps(1e-6)));
    ln_post_->weight.set_data(ln_post_->weight.to(options));
    ln_post_->bias.set_data(ln_post_->bias.to(options));
    proj_ = register_parameter(
        "proj",
        torch::randn({embed_dim_, embed_dim_}) * std::sqrt(1.0 / embed_dim_));
    proj_.set_data(proj_.to(options));
  }

 protected:
  int num_queries_, num_heads_, embed_dim_, kv_dim_;
  torch::Tensor query_;
  layer::MultiheadAttention attn_{nullptr};
  torch::TensorOptions options_;
  torch::nn::LayerNorm ln_q_{nullptr};
  torch::nn::LayerNorm ln_kv_{nullptr};
  torch::nn::LayerNorm ln_post_{nullptr};
  torch::Tensor proj_;

  // Helper to initialize weights with truncated normal distribution
  void trunc_normal(torch::Tensor& tensor, float std) {
    auto mean = 0.0f;
    auto variance = std * std;
    torch::nn::init::normal_(tensor, mean, std::sqrt(variance));
  }

  torch::Tensor repeat(const torch::Tensor& query, int N) {
    // query shape: [64, 3584]
    // Step 1: Unsqueeze the tensor at dimension 1
    auto unsqueezed = query.unsqueeze(1);  // Shape: [64, 1, 3584]

    // Step 2: Repeat the tensor along the specified dimensions
    auto repeated = unsqueezed.repeat({1, N, 1});  // Shape: [64, N, 3584]
    return repeated;
  }
};

TORCH_MODULE(BaseResampler);

class KVProjectorLinearImpl : public torch::nn::Module {
 public:
  KVProjectorLinearImpl(const Context& context) {
    auto model_args = context.get_model_args();

    linear_ = register_module(
        "linear",
        torch::nn::Linear(torch::nn::LinearOptions(model_args.mm_hidden_size(),
                                                   model_args.hidden_size())
                              .bias(false)));
    linear_->weight.set_data(linear_->weight.to(context.get_tensor_options()));
  }

  torch::Tensor forward(torch::Tensor image_features) {
    return linear_(image_features);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      DCHECK_EQ(linear_->weight.sizes(), weight.sizes())
          << "kv_proj weight size mismatch for " << name();
      weight = weight.reshape({weight.size(0), -1});
      linear_->weight.data().copy_(weight);
      is_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
  }

 private:
  torch::nn::Linear linear_{nullptr};
  bool is_weight_loaded_{false};
};
TORCH_MODULE(KVProjectorLinear);

torch::Tensor get_1d_sincos_pos_embed_from_grid(int embed_dim,
                                                const torch::Tensor& pos,
                                                std::pair<int, int> version = {
                                                    2,
                                                    0}) {
  TORCH_CHECK(embed_dim % 2 == 0, "embed_dim must be even");

  // compute omega
  auto omega = torch::arange(embed_dim / 2, torch::kFloat32);
  omega = 1.0 / torch::pow(10000.0, omega / (embed_dim / 2.0));  // (D/2)

  if (version == std::make_pair(2, 0)) {
    auto pos_flat = pos.reshape({-1});                       // (M)
    auto out = torch::einsum("m,d->md", {pos_flat, omega});  // (M, D/2)

    auto emb_sin = torch::sin(out);            // (M, D/2)
    auto emb_cos = torch::cos(out);            // (M, D/2)
    return torch::cat({emb_sin, emb_cos}, 1);  // (M, D)
  } else {
    auto out = torch::einsum("hw,d->hwd", {pos, omega});  // (H, W, D/2)
    auto emb_sin = torch::sin(out);                       // (H, W, D/2)
    auto emb_cos = torch::cos(out);                       // (H, W, D/2)
    return torch::cat({emb_sin, emb_cos}, -1);            // (H, W, D)
  }
}

torch::Tensor get_2d_sincos_pos_embed_from_grid(int embed_dim,
                                                const torch::Tensor& grid,
                                                std::pair<int, int> version = {
                                                    2,
                                                    0}) {
  TORCH_CHECK(embed_dim % 2 == 0, "embed_dim must be even");

  auto emb_h =
      get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[0], version);
  auto emb_w =
      get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[1], version);

  if (version == std::make_pair(2, 0)) {
    return torch::cat({emb_h, emb_w}, 1);  // (H*W, D)
  } else {
    return torch::cat({emb_h, emb_w}, -1);  // (H, W, D)
  }
}

torch::Tensor get_2d_sincos_pos_embed(int embed_dim,
                                      const std::pair<int, int>& grid_size,
                                      bool cls_token = false,
                                      std::pair<int, int> version = {2, 0}) {
  int grid_h_size = grid_size.first;
  int grid_w_size = grid_size.second;

  auto grid_h = torch::arange(grid_h_size, torch::kFloat32);
  auto grid_w = torch::arange(grid_w_size, torch::kFloat32);
  auto grid = torch::meshgrid({grid_w, grid_h}, "xy");     // 注意：w 在前
  auto grid_tensor = torch::stack({grid[0], grid[1]}, 0);  // (2, H, W)

  if (version == std::make_pair(2, 0)) {
    grid_tensor = grid_tensor.unsqueeze(1);  // (2, 1, H, W)
    auto pos_embed =
        get_2d_sincos_pos_embed_from_grid(embed_dim, grid_tensor, version);

    if (cls_token) {
      auto cls_embed = torch::zeros({1, embed_dim}, torch::kFloat32);  // (1, D)
      pos_embed = torch::cat({cls_embed, pos_embed}, 0);  // (1+H*W, D)
    }
    return pos_embed;
  } else {
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid_tensor, version);
  }
}

class Resampler2_5Impl : public BaseResamplerImpl {
 public:
  Resampler2_5Impl(const Context& context) : BaseResamplerImpl(context) {
    set_2d_pos_cache(max_size_, context.get_tensor_options().device());
    kv_proj_ = register_module("kv_proj", KVProjectorLinear(context));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor tgt_sizes) {
    TORCH_CHECK(x.size(0) == tgt_sizes.size(0), "Batch size mismatch!");

    int64_t batch_size = x.size(0);
    auto device = x.device();
    auto dtype = x.dtype();

    auto patch_len = tgt_sizes.index({torch::indexing::Slice(), 0}) *
                     tgt_sizes.index({torch::indexing::Slice(), 1});

    adjust_pos_cache(tgt_sizes, device);

    int64_t max_patch_len = patch_len.max().item<int64_t>();
    auto key_padding_mask =
        torch::zeros({batch_size, max_patch_len},
                     torch::TensorOptions().dtype(torch::kBool).device(device));

    std::vector<torch::Tensor> pos_embeds;
    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t tgt_h = tgt_sizes[i][0].item<int64_t>();
      int64_t tgt_w = tgt_sizes[i][1].item<int64_t>();

      auto pos_embed = pos_embed_
                           .index({torch::indexing::Slice(0, tgt_h),
                                   torch::indexing::Slice(0, tgt_w),
                                   torch::indexing::Slice()})
                           .reshape({tgt_h * tgt_w, -1})
                           .to(dtype);
      pos_embeds.push_back(pos_embed);

      key_padding_mask.index_put_(
          {i,
           torch::indexing::Slice(patch_len[i].item<int64_t>(),
                                  torch::indexing::None)},
          true);
    }

    auto pos_embed = torch::nn::utils::rnn::pad_sequence(
                         pos_embeds, /*batch_first=*/true, 0.0)
                         .permute({1, 0, 2})
                         .to(options_);
    auto x_proj = kv_proj_(x);
    x_proj = ln_kv_->forward(x_proj).permute({1, 0, 2});
    auto q = ln_q_->forward(query_);  // Q * D
    auto q_repeated = repeat(q, batch_size);
    auto out = attn_->forward(q_repeated,          // Q * B * D
                              x_proj + pos_embed,  // L * B * D
                              x_proj,
                              key_padding_mask  // Mask
    );
    out = out.permute({1, 0, 2});  // B * Q * D
    out = ln_post_->forward(out);
    out = torch::matmul(out, proj_);

    return out;
  }
  void load_state_dict(const StateDict& state_dict) {
    const auto& ln_kv_dict = state_dict.get_dict_with_prefix("ln_kv.");
    const auto& ln_kv_weight = ln_kv_dict.get_tensor("weight");
    if (ln_kv_weight.defined()) {
      CHECK_EQ(ln_kv_->weight.sizes(), ln_kv_weight.sizes())
          << "weight size mismatch for " << name();
      ln_kv_->weight.data().copy_(ln_kv_weight);
      is_norm_weight_loaded.at("ln_kv_weight") = true;
    }
    const auto ln_kv_bias = ln_kv_dict.get_tensor("bias");
    if (ln_kv_bias.defined()) {
      CHECK_EQ(ln_kv_->bias.sizes(), ln_kv_bias.sizes())
          << "bias size mismatch for " << name();
      ln_kv_->bias.data().copy_(ln_kv_bias);
      is_norm_weight_loaded.at("ln_kv_bias") = true;
    }

    const auto& ln_post_dict = state_dict.get_dict_with_prefix("ln_post.");
    const auto& ln_post_weight = ln_post_dict.get_tensor("weight");
    if (ln_post_weight.defined()) {
      CHECK_EQ(ln_post_->weight.sizes(), ln_post_weight.sizes())
          << "weight size mismatch for " << name();
      ln_post_->weight.data().copy_(ln_post_weight);
      is_norm_weight_loaded.at("ln_post_weight") = true;
    }
    const auto ln_post_bias = ln_post_dict.get_tensor("bias");
    if (ln_post_bias.defined()) {
      CHECK_EQ(ln_post_->bias.sizes(), ln_post_bias.sizes())
          << "bias size mismatch for " << name();
      ln_post_->bias.data().copy_(ln_post_bias);
      is_norm_weight_loaded.at("ln_post_bias") = true;
    }

    const auto& ln_q_dict = state_dict.get_dict_with_prefix("ln_q.");
    const auto& ln_q_weight = ln_q_dict.get_tensor("weight");
    if (ln_q_weight.defined()) {
      CHECK_EQ(ln_q_->weight.sizes(), ln_q_weight.sizes())
          << "weight size mismatch for " << name();
      ln_q_->weight.data().copy_(ln_q_weight);
      is_norm_weight_loaded.at("ln_q_weight") = true;
    }
    const auto ln_q_bias = ln_q_dict.get_tensor("bias");
    if (ln_q_bias.defined()) {
      CHECK_EQ(ln_q_->bias.sizes(), ln_q_bias.sizes())
          << "bias size mismatch for " << name();
      ln_q_->bias.data().copy_(ln_q_bias);
      is_norm_weight_loaded.at("ln_q_bias") = true;
    }

    kv_proj_->load_state_dict(state_dict.get_dict_with_prefix("kv_proj."));
    const auto query = state_dict.get_tensor("query");
    if (query.defined()) {
      DCHECK_EQ(query.sizes(), query_.sizes())
          << "query size mismatch for " << name();
      query_.data().copy_(query);
      is_query_loaded_ = true;
    }

    const auto proj = state_dict.get_tensor("proj");
    if (proj.defined()) {
      DCHECK_EQ(proj.sizes(), proj_.sizes())
          << "proj size mismatch for " << name();
      proj_.data().copy_(proj);
      is_proj_loaded_ = true;
    }
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (auto& [name, is_loaded] : is_norm_weight_loaded) {
      CHECK(is_loaded) << name << " is not loaded for "
                       << prefix + to_standard_name(name);
    }
    CHECK(is_query_loaded_) << "query is not loaded for " << prefix + "query";
    CHECK(is_proj_loaded_) << "proj is not loaded for " << prefix + "proj";
    attn_->verify_loaded_weights(prefix + "attn.");
  }

 private:
  std::pair<int, int> max_size_ = {70, 70};
  torch::Tensor pos_embed_;
  KVProjectorLinear kv_proj_{nullptr};
  std::unordered_map<std::string, bool> is_norm_weight_loaded = {
      {"ln_kv_weight", false},
      {"ln_kv_bias", false},
      {"ln_post_weight", false},
      {"ln_post_bias", false},
      {"ln_q_weight", false},
      {"ln_q_bias", false},
  };
  bool is_query_loaded_ = false;
  bool is_proj_loaded_ = false;

  static std::string to_standard_name(const std::string& name) {
    size_t pos = name.find_last_of('_');
    if (pos == std::string::npos) return name;
    return name.substr(0, pos) + '.' + name.substr(pos + 1);
  }

  void set_2d_pos_cache(const std::pair<int, int>& max_size,
                        const torch::Device& device) {
    auto pos_embed_arr = get_2d_sincos_pos_embed(
        embed_dim_, max_size, false, std::make_pair(2, 5));
    pos_embed_ = pos_embed_arr.to(torch::kFloat).to(device);
  }

  void adjust_pos_cache(const torch::Tensor& tgt_sizes,
                        const torch::Device& device) {
    int max_h =
        tgt_sizes.index({torch::indexing::Slice(), 0}).max().item<int>();
    int max_w =
        tgt_sizes.index({torch::indexing::Slice(), 1}).max().item<int>();

    if (max_h > max_size_.first || max_w > max_size_.second) {
      max_size_.first = std::max(max_h, max_size_.first);
      max_size_.second = std::max(max_w, max_size_.second);
      set_2d_pos_cache(max_size_, device);
    }
  }
};
TORCH_MODULE(Resampler2_5);

class Idefics2VisionEmbeddingsImpl : public torch::nn::Module {
 public:
  Idefics2VisionEmbeddingsImpl(const Context& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    embed_dim_ = model_args.mm_hidden_size();
    patch_size_ = model_args.mm_patch_size();
    auto in_features = model_args.mm_num_channels() *
                       model_args.mm_patch_size() * model_args.mm_patch_size();
    auto out_features = embed_dim_;
    patch_embedding_ = register_module(
        "patch_embedding",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(true)));
    patch_embedding_->weight.set_data(patch_embedding_->weight.to(options));
    patch_embedding_->bias.set_data(patch_embedding_->bias.to(options));
    image_size_ = model_args.mm_image_size();
    num_patches_per_side_ = image_size_ / patch_size_;
    int num_patches = num_patches_per_side_ * num_patches_per_side_;
    position_embedding_ =
        register_module("position_embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(
                            num_patches, embed_dim_)));
    position_embedding_->weight.set_data(
        position_embedding_->weight.to(options));
  }

  torch::Tensor forward(
      torch::Tensor pixel_values,
      torch::Tensor patch_attention_mask,
      torch::optional<torch::Tensor> tgt_sizes = torch::nullopt) {
    auto batch_size = pixel_values.size(0);
    auto max_im_h = pixel_values.size(2);
    auto max_im_w = pixel_values.size(3);

    namespace F = torch::nn::functional;
    auto col = F::unfold(
        pixel_values,
        F::UnfoldFuncOptions({patch_size_, patch_size_}).stride(patch_size_));
    col = col.permute({0, 2, 1});
    auto embeddings = patch_embedding_(col);

    int64_t max_nb_patches_h = max_im_h / patch_size_;
    int64_t max_nb_patches_w = max_im_w / patch_size_;

    auto boundaries =
        torch::arange(1.0 / num_patches_per_side_,
                      1.0,
                      1.0 / num_patches_per_side_,
                      torch::kFloat32);  // [1/num_patches_per_side_, ..., 1]
    auto device = position_embedding_->weight.device();
    auto position_ids =
        torch::full(
            {batch_size, max_nb_patches_h * max_nb_patches_w}, 0, torch::kInt)
            .to(device);

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      auto p_attn_mask = patch_attention_mask[batch_idx];

      int64_t nb_patches_h, nb_patches_w;
      if (tgt_sizes.has_value()) {
        nb_patches_h = tgt_sizes.value()[batch_idx][0].item<int64_t>();
        nb_patches_w = tgt_sizes.value()[batch_idx][1].item<int64_t>();
      } else {
        nb_patches_h = p_attn_mask.index({torch::indexing::Slice(), 0})
                           .sum()
                           .item<int64_t>();
        nb_patches_w = p_attn_mask.index({0, torch::indexing::Slice()})
                           .sum()
                           .item<int64_t>();
      }

      auto fractional_coords_h =
          torch::arange(0, 1 - 1e-6, 1.0 / nb_patches_h, torch::kFloat32);
      auto fractional_coords_w =
          torch::arange(0, 1 - 1e-6, 1.0 / nb_patches_w, torch::kFloat32);
      // inline at::Tensor at::bucketize(const at::Scalar &self,
      //     const at::Tensor &boundaries, bool out_int32 = false, bool right
      // = false)
      auto bucket_coords_h =
          torch::bucketize(fractional_coords_h, boundaries, true, true);
      auto bucket_coords_w =
          torch::bucketize(fractional_coords_w, boundaries, true, true);

      auto pos_ids = (bucket_coords_h.unsqueeze(1) * num_patches_per_side_ +
                      bucket_coords_w)
                         .flatten()
                         .to(device);  // [H'*W']
      auto mask_indices =
          torch::nonzero(p_attn_mask.flatten()).squeeze(1);  // [N]
      position_ids.index_put_({batch_idx, mask_indices}, pos_ids);
    }

    embeddings = embeddings + position_embedding_->forward(position_ids);
    return embeddings;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("patch_embedding.weight");
    if (weight.defined()) {
      DCHECK_EQ(patch_embedding_->weight.sizes(), weight.sizes())
          << "patch_embedding weight size mismatch for " << name();
      weight = weight.reshape({weight.size(0), -1});
      patch_embedding_->weight.data().copy_(weight);
      is_patch_embedding_weight_loaded = true;
    }

    const auto bias = state_dict.get_tensor("patch_embedding.bias");
    if (bias.defined()) {
      DCHECK_EQ(patch_embedding_->bias.sizes(), bias.sizes())
          << "patch_embedding bias size mismatch for " << name();
      patch_embedding_->bias.data().copy_(bias);
      is_patch_embedding_bias_loaded = true;
    }

    const auto position_embedding =
        state_dict.get_tensor("position_embedding.weight");
    if (position_embedding.defined()) {
      DCHECK_EQ(position_embedding_->weight.sizes(), position_embedding.sizes())
          << "patch_embedding weight size mismatch for " << name();
      position_embedding_->weight.data().copy_(position_embedding);
      is_position_embedding_weight_loaded = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_patch_embedding_weight_loaded)
        << "weight is not loaded for " << prefix + "patch_embedding.weight";
    CHECK(is_patch_embedding_bias_loaded)
        << "bias is not loaded for " << prefix + "patch_embedding.bias";
    CHECK(is_position_embedding_weight_loaded)
        << "weight is not loaded for " << prefix + "position_embedding.weight";
  }

 private:
  int embed_dim_, patch_size_, image_size_, num_patches_per_side_;
  torch::nn::Linear patch_embedding_{nullptr};
  torch::nn::Embedding position_embedding_{nullptr};
  bool is_patch_embedding_weight_loaded = false;
  bool is_patch_embedding_bias_loaded = false;
  bool is_position_embedding_weight_loaded = false;
};
TORCH_MODULE(Idefics2VisionEmbeddings);

class Idefics2EncoderImpl : public torch::nn::Module {
 public:
  Idefics2EncoderImpl(const Context& context) {
    auto model_args = context.get_model_args();

    layers_.reserve(model_args.mm_num_hidden_layers());
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (int32_t i = 0; i < model_args.mm_num_hidden_layers(); i++) {
      int32_t sliding_window = -1;
      if (model_args.use_sliding_window() &&
          i >= model_args.max_window_layers()) {
        sliding_window = model_args.sliding_window();
      }
      auto block = layer::SiglipEncoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // Output hidden states for all intermediate layers
  torch::Tensor forward(const torch::Tensor& embeddings) {
    auto hidden_states = embeddings;
    for (size_t i = 0; i < layers_.size(); ++i) {
      auto& layer = layers_[i];
      hidden_states = layer(hidden_states);
    }
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  std::vector<layer::SiglipEncoderLayer> layers_;
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(Idefics2Encoder);

class VisionAdapterMLPImpl : public torch::nn::Module {
  typedef std::tuple<torch::nn::LayerNorm,
                     torch::nn::Linear,
                     torch::nn::GELU,
                     torch::nn::Linear>
      MLPDef;

 public:
  VisionAdapterMLPImpl(const Context& context) {
    auto options = context.get_tensor_options();

    auto embed_dim = context.get_model_args().hidden_size();
    int num_layers = 3;
    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int idx = 0; idx < num_layers; ++idx) {
      auto lni = register_module(
          "lni",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})
                                   .elementwise_affine(true)
                                   .eps(1e-5)));
      lni->weight.set_data(lni->weight.to(options));
      lni->bias.set_data(lni->bias.to(options));
      auto cpl = register_module(
          "cpl",
          torch::nn::Linear(
              torch::nn::LinearOptions(embed_dim, embed_dim).bias(true)));
      cpl->weight.set_data(cpl->weight.to(options));
      cpl->bias.set_data(cpl->bias.to(options));

      auto act = torch::nn::GELU();
      auto rpl = register_module(
          "rpl",
          torch::nn::Linear(
              torch::nn::LinearOptions(embed_dim, embed_dim).bias(true)));
      rpl->weight.set_data(rpl->weight.to(options));
      rpl->bias.set_data(rpl->bias.to(options));
      auto seq = torch::nn::Sequential(lni, cpl, act, rpl);

      layers_->push_back(seq);
      mlps_.push_back(std::make_tuple(lni, cpl, act, rpl));
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    for (int idx = 0; idx < mlps_.size(); ++idx) {
      auto& mlp = mlps_[idx];
      auto res = std::get<0>(mlp)(x);
      res = std::get<1>(mlp)(res);
      res = std::get<2>(mlp)(res);
      res = std::get<3>(mlp)(res);
      x += res;
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (int idx = 0; idx < mlps_.size(); ++idx) {
      auto& mlp = mlps_[idx];
      const auto& lni_weight =
          state_dict
              .get_dict_with_prefix("layers." + std::to_string(idx) + ".0.")
              .get_tensor("weight");
      if (lni_weight.defined()) {
        CHECK_EQ(std::get<0>(mlp)->weight.sizes(), lni_weight.sizes())
            << "weight size mismatch for " << name();
        std::get<0>(mlp)->weight.data().copy_(lni_weight);
      }
      const auto& lni_bias =
          state_dict
              .get_dict_with_prefix("layers." + std::to_string(idx) + ".0.")
              .get_tensor("bias");
      if (lni_bias.defined()) {
        CHECK_EQ(std::get<0>(mlp)->bias.sizes(), lni_bias.sizes())
            << "bias size mismatch for " << name();
        std::get<0>(mlp)->bias.data().copy_(lni_bias);
      }

      const auto& cpl_weight =
          state_dict
              .get_dict_with_prefix("layers." + std::to_string(idx) + ".1.")
              .get_tensor("weight");
      if (cpl_weight.defined()) {
        CHECK_EQ(std::get<1>(mlp)->weight.sizes(), cpl_weight.sizes())
            << "weight size mismatch for " << name();
        std::get<1>(mlp)->weight.data().copy_(cpl_weight);
      }

      const auto& cpl_bias =
          state_dict
              .get_dict_with_prefix("layers." + std::to_string(idx) + ".1.")
              .get_tensor("bias");
      if (cpl_bias.defined()) {
        CHECK_EQ(std::get<1>(mlp)->bias.sizes(), cpl_bias.sizes())
            << "bias size mismatch for " << name();
        std::get<1>(mlp)->bias.data().copy_(cpl_bias);
      }

      const auto& rpl_weight =
          state_dict
              .get_dict_with_prefix("layers." + std::to_string(idx) + ".3.")
              .get_tensor("weight");
      if (rpl_weight.defined()) {
        CHECK_EQ(std::get<3>(mlp)->weight.sizes(), rpl_weight.sizes())
            << "weight size mismatch for " << name();
        std::get<3>(mlp)->weight.data().copy_(rpl_weight);
      }

      const auto& rpl_bias =
          state_dict
              .get_dict_with_prefix("layers." + std::to_string(idx) + ".3.")
              .get_tensor("bias");
      if (rpl_bias.defined()) {
        CHECK_EQ(std::get<0>(mlp)->bias.sizes(), lni_bias.sizes())
            << "bias size mismatch for " << name();
        std::get<3>(mlp)->bias.data().copy_(rpl_bias);
      }
      is_mpls_loaded.at(idx) = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int idx = 0; idx < mlps_.size(); ++idx) {
      CHECK(is_mpls_loaded.at(idx)) << "weight is not loaded for "
                                    << prefix + "layer." + std::to_string(idx);
    }
  }

 private:
  torch::nn::ModuleList layers_{nullptr};
  std::vector<MLPDef> mlps_;
  std::vector<bool> is_mpls_loaded = std::vector<bool>(3, false);
  ;
};
TORCH_MODULE(VisionAdapterMLP);

class Idefics2VisionTransformerImpl : public torch::nn::Module {
 public:
  Idefics2VisionTransformerImpl(const Context& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    embeddings_ =
        register_module("embeddings", Idefics2VisionEmbeddings(context));
    encoder_ = register_module("encoder", Idefics2Encoder(context));
    post_layernorm_ = register_module(
        "post_layernorm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    post_layernorm_->weight.set_data(post_layernorm_->weight.to(options));
    post_layernorm_->bias.set_data(post_layernorm_->bias.to(options));
  }

  torch::Tensor forward(const torch::Tensor& pixel_values,
                        const torch::Tensor& patch_attention_mask,
                        const torch::Tensor& tgt_sizes) {
    auto hidden_states =
        embeddings_(pixel_values, patch_attention_mask, tgt_sizes);
    auto encoder_outputs = encoder_(hidden_states);
    auto last_hidden_state = post_layernorm_(encoder_outputs);
    return last_hidden_state;
  }

  void load_state_dict(const StateDict& state_dict) {
    embeddings_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));

    const auto& post_norm_weight =
        state_dict.get_tensor("post_layernorm.weight");
    if (post_norm_weight.defined()) {
      CHECK_EQ(post_layernorm_->weight.sizes(), post_norm_weight.sizes())
          << "weight size mismatch for " << name();
      post_layernorm_->weight.data().copy_(post_norm_weight);
    }
    const auto& post_norm_bias = state_dict.get_tensor("post_layernorm.bias");
    if (post_norm_bias.defined()) {
      CHECK_EQ(post_layernorm_->bias.sizes(), post_norm_bias.sizes())
          << "bias size mismatch for " << name();
      post_layernorm_->bias.data().copy_(post_norm_bias);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
  }

 private:
  Idefics2VisionEmbeddings embeddings_{nullptr};
  Idefics2Encoder encoder_{nullptr};
  torch::nn::LayerNorm post_layernorm_{nullptr};
};
TORCH_MODULE(Idefics2VisionTransformer);

struct MiniCPMVImageInputs {
  torch::Tensor image_bounds;
  std::vector<torch::Tensor> data;
  torch::Tensor tgt_sizes;
  std::string type;
};

class MiniCPMV2_6Impl : public torch::nn::Module {
 public:
  MiniCPMV2_6Impl(const Context& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    use_vision_adapter_ =
        context.get_model_args().vision_custom_adapter() == "mlp3";

    vpm_ = register_module("visual_", Idefics2VisionTransformer(context));

    resampler_ = register_module("resampler", Resampler2_5(context));

    if (!model_args_.image_embedding_mode()) {
      language_model_ = register_module("model", QWen2ForCausalLM(context));
      if (use_vision_adapter_)
        mlp_ = register_module("mlp", VisionAdapterMLP(context));
    }
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    const auto& mm_data = input_params.mm_data;

    torch::Tensor image_embeds;
    if (const auto& res = mm_data.get<torch::Tensor>("image_embeds"))
      image_embeds = res.value();

    auto pixel_values = mm_data.get_tensor_vec("pixel_values");

    torch::Tensor tgt_sizes;
    if (const auto& res = mm_data.get<torch::Tensor>("tgt_sizes"))
      tgt_sizes = res.value();

    torch::Tensor image_embedding;
    std::optional<MiniCPMVImageInputs> image_inputs;
    if (image_embeds.defined()) {
      image_inputs = generate_image_inputs({}, tokens, image_embeds, tgt_sizes);
    } else if (pixel_values.size() > 0) {
      image_inputs = generate_image_inputs(
          pixel_values, tokens, torch::Tensor(), tgt_sizes);
    }
    image_embedding = get_vision_embedding(image_inputs);

    if (model_args_.image_embedding_mode()) {
      return image_embedding;
    }

    if (use_vision_adapter_ && image_embedding.defined()) {
      image_embedding = mlp_(image_embedding);
    }

    input_params.input_embedding =
        merge_text_vision_embeddings(tokens, image_inputs, image_embedding);

    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor get_image_bounds(
      const torch::Tensor& input_ids,
      int64_t im_start_id,
      int64_t im_end_id,
      const std::optional<int64_t>& slice_start_id = std::nullopt,
      const std::optional<int64_t>& slice_end_id = std::nullopt) {
    auto start_cond = (input_ids == im_start_id);
    auto end_cond = (input_ids == im_end_id);

    if (slice_start_id.has_value() && slice_end_id.has_value()) {
      start_cond |= (input_ids == slice_start_id.value());
      end_cond |= (input_ids == slice_end_id.value());
    }

    auto image_start_tokens_vec = torch::where(start_cond);
    auto image_end_tokens_vec = torch::where(end_cond);

    auto image_start_tokens = image_start_tokens_vec[0];
    auto image_end_tokens = image_end_tokens_vec[0];

    if (image_start_tokens.numel() > 0) {
      image_start_tokens += 1;  // Adjust for start token offset
    }

    int64_t valid_image_nums =
        std::max(image_start_tokens.size(0), image_end_tokens.size(0));

    if (valid_image_nums == 0) {
      return torch::zeros({0, 2},
                          torch::TensorOptions().device(input_ids.device()));
    }

    return torch::hstack(
        {image_start_tokens.slice(0, 0, valid_image_nums).unsqueeze(-1),
         image_end_tokens.slice(0, 0, valid_image_nums).unsqueeze(-1)});
  }

  std::optional<MiniCPMVImageInputs> parse_and_validate_inputs(
      const torch::Tensor& input_ids,
      const std::vector<torch::Tensor>& pixel_values,
      const torch::Tensor& tgt_sizes,
      const std::optional<torch::Tensor>& image_embeds = std::nullopt,
      const std::optional<int64_t>& im_start_id = std::nullopt,
      const std::optional<int64_t>& im_end_id = std::nullopt,
      const std::optional<int64_t>& slice_start_id = std::nullopt,
      const std::optional<int64_t>& slice_end_id = std::nullopt) {
    // Case 1: Handle image_embeds if provided
    if (image_embeds.has_value()) {
      if (!im_start_id.has_value() || !im_end_id.has_value()) {
        LOG(FATAL) << "im_start_id and im_end_id must be "
                      "provided when using image_embeds.";
      }

      return MiniCPMVImageInputs{
          .image_bounds = get_image_bounds(input_ids,
                                           im_start_id.value(),
                                           im_end_id.value(),
                                           slice_start_id,
                                           slice_end_id),
          .data = {image_embeds.value()},  // Wrap in a vector
          .tgt_sizes = torch::Tensor(),    // Empty tensor
          .type = "image_embeds"};
    }

    std::vector<torch::Tensor> pixel_value_flat;
    constexpr const int channel = 3;
    for (const auto& item : pixel_values) {
      auto vec = item.split(channel);
      pixel_value_flat.insert(pixel_value_flat.end(), vec.begin(), vec.end());
    }

    if (pixel_value_flat.size() != tgt_sizes.size(0)) {
      LOG(INFO) << "pixel_value_flat size:" << pixel_value_flat.size()
                << " tgt_sizes shape:" << tgt_sizes.sizes();
      LOG(FATAL)
          << "Inconsistent batch lengths between pixel_values and tgt_sizes.";
    }

    return MiniCPMVImageInputs{
        .image_bounds = get_image_bounds(input_ids,
                                         im_start_id.value(),
                                         im_end_id.value(),
                                         slice_start_id,
                                         slice_end_id),
        .data = pixel_value_flat,
        .tgt_sizes = tgt_sizes,
        .type = "pixel_values"};
  }

  std::optional<MiniCPMVImageInputs> generate_image_inputs(
      const std::vector<torch::Tensor>& pixel_values,
      const torch::Tensor& tokens,
      const torch::Tensor& inputs_embeds,
      const torch::Tensor& tgt_sizes = torch::Tensor()) {
    torch::Device device = tokens.device();
    std::optional<torch::Tensor> optional_image_embeds = std::nullopt;
    if (inputs_embeds.defined()) {
      optional_image_embeds = inputs_embeds;
    }
    auto image_inputs = parse_and_validate_inputs(tokens,
                                                  pixel_values,
                                                  tgt_sizes,
                                                  optional_image_embeds,
                                                  im_start_id_val_,
                                                  im_end_id_val_,
                                                  slice_start_id_val_,
                                                  slice_end_id_val_);

    return image_inputs;
  }

  torch::Tensor get_vision_embedding(
      const std::optional<MiniCPMVImageInputs>& image_inputs) {
    if (!image_inputs.has_value()) {
      return torch::Tensor();
    }
    const auto& inputs = image_inputs.value();
    torch::Tensor vision_hidden_states;

    if (inputs.type == "image_embeds") {
      vision_hidden_states = inputs.data[0];
    } else {
      vision_hidden_states = get_vision_hidden_states(inputs);
    }
    return vision_hidden_states;
  }

  torch::Tensor merge_text_vision_embeddings(
      const torch::Tensor& tokens,
      const std::optional<MiniCPMVImageInputs>& image_inputs,
      torch::Tensor& vision_hidden_states) {
    torch::Tensor llm_embedding = language_model_->get_input_embeddings(tokens);
    if (!image_inputs.has_value()) {
      return llm_embedding;
    }
    const auto& inputs = image_inputs.value();
    if (inputs.type == "image_embeds") {
      vision_hidden_states = vision_hidden_states.to(llm_embedding.device())
                                 .to(llm_embedding.dtype());
    }
    if (inputs.image_bounds.size(0) > 0) {
      auto image_bounds = inputs.image_bounds.to(llm_embedding.device());
      std::vector<torch::Tensor> ranges;

      for (int64_t i = 0; i < image_bounds.size(0); ++i) {
        int64_t start = image_bounds[i][0].item<int64_t>();
        int64_t end = image_bounds[i][1].item<int64_t>();
        ranges.push_back(torch::arange(start, end, torch::kLong));
      }

      auto image_indices = torch::stack(ranges).to(llm_embedding.device());

      llm_embedding.scatter_(
          0,
          image_indices.view({-1, 1}).expand({-1, llm_embedding.size(-1)}),
          vision_hidden_states.view({-1, vision_hidden_states.size(-1)}));
    }
    return llm_embedding;
  }

  torch::Tensor get_vision_hidden_states(const MiniCPMVImageInputs& data) {
    const auto& pixel_values = data.data;
    auto tgt_sizes = data.tgt_sizes;

    auto device = tgt_sizes.device();
    // auto dtype = model_args_.dtype();

    std::vector<torch::Tensor> all_pixel_values_lst;
    for (const auto& tensor : pixel_values) {
      all_pixel_values_lst.push_back(tensor.flatten(0, 1).permute({1, 0}));
    }

    auto all_pixel_values = torch::nn::utils::rnn::pad_sequence(
        all_pixel_values_lst, /*batch_first=*/true, /*padding_value=*/0.0);

    auto sizes = all_pixel_values.sizes();
    int64_t B = sizes[0];  // Batch size
    int64_t L = sizes[1];  // Sequence length
    int64_t C = sizes[2];  // Channel size
    all_pixel_values = all_pixel_values.permute({0, 2, 1}).reshape(
        torch::IntArrayRef({B, 3, -1, L}));

    auto max_patches = (tgt_sizes.index({torch::arange(B), 0}) *
                        tgt_sizes.index({torch::arange(B), 1}))
                           .max()
                           .item<int64_t>();
    auto patch_attn_mask =
        torch::zeros({B, 1, max_patches},
                     torch::TensorOptions().dtype(torch::kBool).device(device));

    for (int64_t i = 0; i < B; ++i) {
      int64_t num_true =
          tgt_sizes[i][0].item<int64_t>() * tgt_sizes[i][1].item<int64_t>();
      patch_attn_mask.index_put_(
          {i,
           0,
           torch::arange(0, num_true, torch::TensorOptions().device(device))},
          true);
    }

    auto vision_embedding =
        vpm_(all_pixel_values.to(options_), patch_attn_mask, tgt_sizes);

    return resampler_(vision_embedding, tgt_sizes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    // load weight
    for (const auto& state_dict : loader->get_state_dicts()) {
      if (!model_args_.image_embedding_mode()) {
        if (use_vision_adapter_)
          mlp_->load_state_dict(state_dict->get_dict_with_prefix("mlp."));
      }
      resampler_->load_state_dict(
          state_dict->get_dict_with_prefix("resampler."));
      vpm_->load_state_dict(state_dict->get_dict_with_prefix("vpm."));
    }
    language_model_->load_model(std::move(loader),
                                "llm.");  // llm. weight name prefix

    // verify
    if (!model_args_.image_embedding_mode()) {
      if (use_vision_adapter_) mlp_->verify_loaded_weights("mlp.");
    }
    resampler_->verify_loaded_weights("resampler.");
    vpm_->verify_loaded_weights("vpm.");
  }

  LlmHead get_lm_head() { return language_model_->get_lm_head(); }

  void set_lm_head(LlmHead& head) { language_model_->set_lm_head(head); }

  AtbWordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(AtbWordEmbedding& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  QWen2ForCausalLM language_model_{nullptr};
  ModelArgs model_args_;
  int64_t im_start_id_val_ = 151646;
  int64_t im_end_id_val_ = 151647;
  int64_t slice_start_id_val_ = 151656;
  int64_t slice_end_id_val_ = 151657;
  Idefics2VisionTransformer vpm_{nullptr};
  Resampler2_5 resampler_{nullptr};
  bool use_vision_adapter_ = false;
  VisionAdapterMLP mlp_{nullptr};
  torch::TensorOptions options_;
  int device_id = 0;
};
TORCH_MODULE(MiniCPMV2_6);

REGISTER_CAUSAL_VLM_MODEL(minicpmv, MiniCPMV2_6);
REGISTER_INPUT_PROCESSOR(minicpmv, MiniCPMInputProcessor);
REGISTER_IMAGE_PROCESSOR(minicpmv, MiniCPMVImageProcessor);

REGISTER_MODEL_ARGS(minicpmv, [&] {
  // text config
  LOAD_ARG_OR(model_type, "model_type", "minicpmv");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vision_custom_adapter, "vision_adapter_type", "");
  LOAD_ARG_OR(vision_max_slice_nums, "slice_config.max_slice_nums", 9);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(rope_scaling_factor, "rope_scaling_factor", 1.0f);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 131072);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(query_num, "query_num", 64);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(vocab_size, "vocab_size", 151666);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");

  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 980);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_attention_heads", 16);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.num_hidden_layers", 27);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_num_channels, "vision_config.num_channels", 3);
  LOAD_ARG_OR(mm_dropout, "attention_dropout", 0.0);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.layer_norm_eps", 1e-06);
  LOAD_ARG_OR_FUNC(mm_head_dim, "mm_head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });
});
}  // namespace xllm::hf