# Layer 层多后端抽象设计文档

> **状态**: 提案  
> **作者**: xLLM Team  
> **创建日期**: 2026-03-17  
> **涉及模块**: `xllm/core/layers/`, `xllm/models/`

---

## 1. 背景与动机

xLLM 当前需要支持多种芯片后端（华为 NPU / NVIDIA CUDA / 寒武纪 MLU / 摩尔线程 MUSA / ILU 等）。随着支持的模型和后端不断增多，现有架构中 Layer 层与具体芯片实现的强耦合问题日益突出，主要体现在：

- **模型代码按后端分裂**：同一个模型（如 DeepSeek V2）在不同后端有完全独立的模型文件（`llm/deepseek_v2.h` vs `llm/npu/deepseek_v2.h`），代码无法共享
- **新增后端成本极高**：每适配一个新芯片，需要 fork 整套模型文件并修改 Layer 类型
- **维护负担沉重**：模型逻辑的任何改动都需要在多个后端文件中同步

本文档提出基于 **Layer 工厂 + 依赖注入** 的抽象方案，目标是将「模型拓扑/逻辑」与「硬件计算实现」解耦。

---

## 2. 现状分析

### 2.1 目录结构

```
xllm/core/layers/
├── common/                  # 通用层实现（Attention、MLP、Norm、Linear 等）
├── npu/                     # 华为 NPU（ATB 算子）
│   ├── loader/              # NPU 权重加载器
│   ├── npu_base_layer.h     # NPU 层基类
│   ├── npu_deepseek_v2_decoder_layer_impl.h/cpp
│   ├── npu_qwen3_decoder_layer_impl.h/cpp
│   └── ...
├── cuda/                    # CUDA（xattention, flashinfer）
├── mlu/                     # 寒武纪 MLU
├── musa/                    # 摩尔线程 MUSA
├── ilu/                     # ILU
├── qwen2_decoder_layer.h    # 通用 decoder layer（CUDA/MLU/ILU 共用）
├── qwen3_decoder_layer.h
└── ...

xllm/models/
├── models.h                 # 编译期后端选择入口（#ifdef 驱动）
├── model_registry.h         # 模型注册中心
├── llm/
│   ├── llm_model_base.h     # 通用模型模板基类
│   ├── qwen3.h              # CUDA/MLU 版 Qwen3
│   ├── deepseek_v2.h        # MLU 版 DeepSeek V2
│   └── npu/
│       ├── llm_model_base.h # NPU 模型模板基类（独立一套）
│       ├── qwen3.h           # NPU 版 Qwen3
│       ├── deepseek_v2.h     # NPU 版 DeepSeek V2
│       └── ...
```

### 2.2 两套独立体系

当前存在两套完全独立的模型体系，无统一抽象接口。

#### 通用后端体系（CUDA / MLU / ILU / MUSA）

```cpp
// xllm/core/layers/qwen2_decoder_layer.h
class Qwen2DecoderLayerImpl : public torch::nn::Module {
 public:
  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);
 private:
  Qwen2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
};
```

- 基类：直接继承 `torch::nn::Module`
- 配套组件：`WordEmbedding` / `RMSNorm` / `LmHead`（通用实现）
- 模型基类：`LlmModelImplBase<DecoderLayerType>`（`llm/llm_model_base.h`）

#### NPU 后端体系

```cpp
// xllm/core/layers/npu/npu_deepseek_v2_decoder_layer_impl.h
class NpuDeepseekV2DecoderLayerImpl : public BaseLayer {
 public:
  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);
};
```

- 基类：继承 `BaseLayer`（封装了 ATB 执行引擎、权重生命周期管理）
- 配套组件：`NpuWordEmbedding` / `NpuRMSNorm` / `NpuLmHead` / `NpuPosEmbedding`
- 模型基类：`LlmModelImplBase<DecoderLayerType>`（`llm/npu/llm_model_base.h`，与通用版同名但完全不同的类）

### 2.3 关键差异对比

| 维度 | 通用后端 | NPU 后端 |
|------|---------|----------|
| **Layer 基类** | `torch::nn::Module` | `BaseLayer` → `torch::nn::Module` |
| **forward 签名** | `(x, residual, positions, attn_metadata, kv_cache, input_params)` | `(x, cos_pos, sin_pos, attn_mask, kv_cache, input_params, event, event_flag, node_id)` |
| **位置编码** | 在 `AttentionMetadata` 内部管理 | Model 层预算 cos/sin 后显式传入 |
| **Attention Mask** | `AttentionMetadata` 内部管理 | Model 层预算后显式传入 |
| **异步控制** | 无 | `aclrtEvent*` + `event_flag` |
| **权重生命周期** | 标准 `load_state_dict` | `load → verify → merge → free → reload` |
| **Embedding** | `WordEmbedding` | `NpuWordEmbedding` |
| **Norm** | `RMSNorm` | `NpuRMSNorm` |
| **LmHead** | `LmHead` | `NpuLmHead` |
| **Model 基类** | `llm/llm_model_base.h` | `llm/npu/llm_model_base.h`（独立一套） |

### 2.4 当前初始化链路

以 DeepSeek V2 模型为例，对比两个后端从入口到 Layer 创建的完整路径：

```
NPU 后端                                    MLU/通用后端
--------                                    -----------
create_llm_model("deepseek_v2")             create_llm_model("deepseek_v2")
  │ (ModelRegistry 查表)                       │ (ModelRegistry 查表)
  ▼                                            ▼
DeepseekV2ForCausalLMImpl                   DeepseekV2ForCausalLMImpl
  : npu::LlmForCausalLMImplBase<              : LlmForCausalLMImplBase<
      DeepseekV2Model>                            DeepseekV2Model>
  │                                            │
  ▼                                            ▼
DeepseekV2ModelImpl (npu 版)                DeepseekV2ModelImpl (通用版)
  ├─ NpuWordEmbedding                        ├─ WordEmbedding
  ├─ NpuPosEmbedding                         ├─ (无，在 attn 内部)
  ├─ NpuRMSNorm                              ├─ RMSNorm
  ├─ AttentionMask                           │
  └─ for i in n_layers:                      └─ for i in n_layers:
       DeepseekV2DecoderLayer(ctx, i)             layer::DeepseekV2DecoderLayer(ctx, i)
            │                                          │
            ▼                                          ▼
  NpuDeepseekV2DecoderLayer  ← 写死       layer::DeepseekV2DecoderLayerImpl ← 写死
     └─ BaseLayer                              └─ DeepseekV2Attention
     └─ ATB Model::Node                       └─ FusedMoE / DenseMLP
     └─ DeepseekV2DecoderLoader                └─ RMSNorm × 2
```

### 2.5 后端选择机制

当前使用**编译期宏隔离**作为唯一的后端选择机制：

```cpp
// models.h — 同一模型名称在不同后端 include 不同文件
#if defined(USE_NPU)
#include "llm/npu/deepseek_v2.h"    // NPU 版
#include "llm/npu/qwen3.h"
#elif defined(USE_MLU)
#include "llm/deepseek_v2.h"       // MLU 版（用通用 layer）
#include "llm/qwen3.h"
#elif defined(USE_CUDA)
#include "llm/qwen3.h"             // CUDA 版（用通用 layer）
#endif
```

两个后端的模型文件中都使用相同的注册宏：

```cpp
// llm/npu/deepseek_v2.h
REGISTER_CAUSAL_MODEL(deepseek_v2, DeepseekV2ForCausalLM);

// llm/deepseek_v2.h（MLU 版）
REGISTER_CAUSAL_MODEL(deepseek_v2, DeepseekV2ForCausalLM);
```

由于 `#ifdef` 保证同一次编译只 include 一个，所以不会冲突——但也意味着**无法在同一二进制中支持多后端**。

### 2.6 现状问题总结

| 问题 | 影响 |
|------|------|
| 同一模型需为每个后端维护独立的模型文件 | 代码量 = N(模型) × M(后端)，增长迅速 |
| Layer 类型在模型构造函数中硬编码 | 模型逻辑与后端实现无法解耦 |
| 配套组件（Embedding/Norm/LmHead）也分后端实现 | 进一步放大代码分裂 |
| forward 签名不统一 | 模型基类也被迫分裂为两套 |
| 编译期宏是唯一选择机制 | 无法在运行时选择后端，测试困难 |
| 模型逻辑改动需多文件同步 | 容易遗漏，维护成本高 |

---

## 3. 设计目标

1. **模型逻辑只写一份** — 消除同一模型在多后端间的代码重复
2. **新增后端成本最小化** — 只需实现 Layer 接口 + 注册，无需 fork 模型文件
3. **渐进式改造** — 不破坏现有实现，通过 Adapter 兼容已有代码
4. **保留后端特化的性能优化空间** — 不强制统一底层实现细节
5. **支持未来扩展** — 可选支持运行时后端选择

---

## 4. 总体方案：Layer 工厂 + 依赖注入

### 4.1 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                 UnifiedLlmModelImplBase                      │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │ IBackendStrategy  │  │ vector<shared_ptr<IDecoderLayer>>│  │
│  │ (prepare context) │  │ (unified forward)                │  │
│  └────────┬─────────┘  └──────────────┬───────────────────┘  │
│           │                           │                      │
└───────────┼───────────────────────────┼──────────────────────┘
            │                           │
      ┌─────┴─────┐             ┌───────┴────────┐
      │           │             │                │
  NpuStrategy  CudaStrategy  NpuAdapter     CommonAdapter
                              │                │
                      NpuDecoderLayerImpl  Qwen2DecoderLayerImpl
                      (现有代码不变)        (现有代码不变)
```

核心思路是引入五层抽象：

1. **`LayerInitContext`** — 统一 Layer 构造参数
2. **`LayerForwardContext`** — 统一 forward 调用参数
3. **`IDecoderLayer`** — 统一 Layer 抽象接口
4. **`IBackendStrategy`** — 后端策略接口（位置编码、mask、异步等）
5. **`LayerRegistry`** — Layer 工厂注册中心

### 4.2 详细设计

#### 4.2.1 统一 Layer 构造上下文（LayerInitContext）

**问题**：不同后端 Layer 的构造函数参数不一致：

| Layer | 构造函数签名 |
|-------|------------|
| `Qwen2DecoderLayerImpl` (通用) | `(const ModelContext& context)` |
| `Qwen3MoeDecoderLayerImpl` (通用) | `(const ModelContext& context, int32_t layer_id)` |
| `DeepseekV2DecoderLayerImpl` (MLU) | `(const ModelContext& context, int32_t layer_id)` |
| `NpuDeepseekV2DecoderLayerImpl` (NPU) | `(const ModelContext& context, const int32_t layer_id)` |
| `NpuGlm4DecoderLayerImpl` (NPU) | `(const ModelContext& context)` |
| `NpuEagle3DecoderLayerImpl` (NPU) | `(const ModelContext& context)` |

`ModelContext` 是所有后端都需要的（内含 `ModelArgs`、`ParallelArgs`、`TensorOptions`、`QuantArgs` 等），但 `layer_id` 等参数只有部分 Layer 需要。随着后端和模型的增多，未来还可能出现更多构造期参数（如特殊的优化配置、算子选择 hint 等）。

**方案**：引入 `LayerInitContext`，封装 Layer 构造所需的全部参数：

```cpp
// xllm/core/layers/layer_init_context.h

#pragma once
#include <any>
#include <string>
#include <unordered_map>
#include "framework/model_context.h"

namespace xllm {
namespace layer {

struct LayerInitContext {
  // 核心参数（所有 Layer 必需）
  const ModelContext& model_context;

  // 层索引（大多数 Layer 需要）
  int32_t layer_id = -1;

  // 总层数（便于 Layer 内部判断是否为首尾层等）
  int32_t total_layers = -1;

  // 后端标识（"npu", "cuda", "mlu", "musa", "ilu" 等）
  std::string backend;

  // 可扩展的属性包 —— 用于传递后端/模型特有的构造参数，
  // 避免为每个新参数修改接口
  std::unordered_map<std::string, std::any> extra_params;

  // 便捷的类型安全取值方法
  template <typename T>
  T get_extra(const std::string& key, const T& default_value = T{}) const {
    auto it = extra_params.find(key);
    if (it != extra_params.end()) {
      return std::any_cast<T>(it->second);
    }
    return default_value;
  }

  template <typename T>
  void set_extra(const std::string& key, T&& value) {
    extra_params[key] = std::forward<T>(value);
  }

  // 从 ModelContext 提取的常用快捷方法
  const ModelArgs& model_args() const {
    return model_context.get_model_args();
  }
  const ParallelArgs& parallel_args() const {
    return model_context.get_parallel_args();
  }
  const QuantArgs& quant_args() const {
    return model_context.get_quant_args();
  }
  torch::TensorOptions tensor_options() const {
    return model_context.get_tensor_options();
  }
};

}  // namespace layer
}  // namespace xllm
```

**设计要点**：

- **核心字段固定**：`model_context`、`layer_id`、`total_layers`、`backend` 是高频使用的参数，直接作为命名字段
- **扩展字段动态**：通过 `extra_params`（`std::unordered_map<string, std::any>`）支持任意后端特有的构造参数，无需每次修改接口
- **类型安全**：`get_extra<T>()` 提供带默认值的类型安全访问，`set_extra()` 支持任意类型
- **快捷方法**：`model_args()`、`parallel_args()` 等避免 `ctx.model_context.get_model_args()` 的链式调用

**使用示例**：

```cpp
// 模型基类在循环中构造 LayerInitContext
for (int32_t i = 0; i < model_args.n_layers(); ++i) {
  LayerInitContext init_ctx{
      .model_context = context,
      .layer_id = i,
      .total_layers = model_args.n_layers(),
      .backend = context.get_backend(),
  };
  // 设置后端特有参数
  init_ctx.set_extra("enable_eplb", FLAGS_enable_eplb);
  init_ctx.set_extra("redundant_experts_num", FLAGS_redundant_experts_num);

  layers_.push_back(factory(init_ctx));
}
```

```cpp
// NPU Layer 构造函数从 LayerInitContext 读取参数
NpuDeepseekV2DecoderLayerImpl::NpuDeepseekV2DecoderLayerImpl(
    const LayerInitContext& init_ctx)
    : BaseLayer(init_ctx.model_context),
      device_id_(init_ctx.tensor_options().device().index()),
      layer_id_(init_ctx.layer_id),
      num_speculative_tokens_(init_ctx.model_args().num_speculative_tokens()) {
  // ...
  bool enable_eplb = init_ctx.get_extra<bool>("enable_eplb", false);
  redundant_experts_num_ = init_ctx.get_extra<int32_t>(
      "redundant_experts_num", 0);
}
```

```cpp
// 通用 Layer 构造函数只用核心字段
Qwen2DecoderLayerImpl::Qwen2DecoderLayerImpl(
    const LayerInitContext& init_ctx)
    : parallel_args_(init_ctx.parallel_args()) {
  const auto& model_args = init_ctx.model_args();
  const auto& options = init_ctx.tensor_options();
  // layer_id 和 extra_params 不用的话直接忽略
}
```

#### 4.2.2 统一 Forward 上下文

将不同后端 forward 签名中的参数统一封装为一个上下文对象：

```cpp
// xllm/core/layers/layer_forward_context.h

#pragma once
#include <torch/torch.h>
#include <atomic>
#include <memory>
#include <optional>
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "common/attention_metadata.h"

namespace xllm {
namespace layer {

struct LayerForwardContext {
  // 所有后端都需要的核心字段
  torch::Tensor& x;
  KVCache& kv_cache;
  const ModelInputParams& input_params;

  // 通用后端使用的字段
  std::optional<torch::Tensor> residual;
  torch::Tensor positions;
  std::shared_ptr<AttentionMetadata> attn_metadata;

  // NPU 后端使用的字段
  std::optional<torch::Tensor> cos_pos;
  std::optional<torch::Tensor> sin_pos;
  std::optional<torch::Tensor> attn_mask;

  // 异步控制（NPU 需要，其他后端可忽略）
  void* async_event = nullptr;
  std::atomic<bool>* event_flag = nullptr;
  int layer_id = 0;
};

}  // namespace layer
}  // namespace xllm
```

设计要点：
- 后端特化字段使用 `std::optional` 或指针，不同后端按需填充
- 避免在上下文中引入任何后端专用的头文件依赖（如 `aclrtEvent`），使用 `void*` 擦除类型
- `LayerForwardContext` 是值语义的轻量对象，按栈分配

#### 4.2.3 统一 DecoderLayer 抽象接口

```cpp
// xllm/core/layers/i_decoder_layer.h

#pragma once
#include <torch/torch.h>
#include "layer_forward_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

class IDecoderLayer : public torch::nn::Module {
 public:
  virtual ~IDecoderLayer() = default;

  // 统一的 forward 入口
  virtual torch::Tensor forward(LayerForwardContext& ctx) = 0;

  // 权重加载（必须实现）
  virtual void load_state_dict(const StateDict& state_dict) = 0;

  // 权重生命周期管理（带默认空实现，NPU 等需要的后端 override）
  virtual void verify_loaded_weights() const {}
  virtual void verify_loaded_weights(const std::string& prefix) const {}
  virtual void merge_loaded_weights() {}
  virtual void free_weights() {}
  virtual void reload_weights() {}
  virtual void reload_weights_from_device() {}
  virtual void merge_and_move_pinned_host() {}

  // Layer 初始化回调（可选）
  virtual int64_t init_layer() { return 0; }
};

}  // namespace layer
}  // namespace xllm
```

设计要点：
- 继承 `torch::nn::Module` 保持与 PyTorch 生态兼容
- 权重生命周期方法提供默认空实现，通用后端不需要 override
- 不引入任何后端特化类型

#### 4.2.4 后端适配器（Adapter）

为现有实现编写适配器，**不修改已有代码**，只在外层包装：

```cpp
// xllm/core/layers/npu/npu_decoder_layer_adapter.h

#pragma once
#include "core/layers/i_decoder_layer.h"
#include "core/layers/layer_init_context.h"
#include "npu_base_layer.h"

namespace xllm {
namespace layer {

// 泛型适配器：支持接受 (ModelContext, layer_id) 构造的 NPU Layer
template <typename NpuLayerImpl>
class NpuDecoderLayerAdapter : public IDecoderLayer {
 public:
  explicit NpuDecoderLayerAdapter(const LayerInitContext& init_ctx)
      : impl_(register_module("impl",
            NpuLayerImpl(init_ctx.model_context, init_ctx.layer_id))) {}

  torch::Tensor forward(LayerForwardContext& ctx) override {
    return impl_->forward(
        ctx.x,
        ctx.cos_pos.value(),
        ctx.sin_pos.value(),
        ctx.attn_mask.value(),
        ctx.kv_cache,
        ctx.input_params,
        static_cast<aclrtEvent*>(ctx.async_event),
        ctx.event_flag,
        ctx.layer_id);
  }

  void load_state_dict(const StateDict& state_dict) override {
    impl_->load_state_dict(state_dict);
  }
  void verify_loaded_weights() const override {
    impl_->verify_loaded_weights();
  }
  void merge_loaded_weights() override {
    impl_->merge_loaded_weights();
  }
  void free_weights() override { impl_->free_weights(); }
  void reload_weights() override { impl_->reload_weights(); }
  void reload_weights_from_device() override {
    impl_->reload_weights_from_device();
  }
  void merge_and_move_pinned_host() override {
    impl_->merge_and_move_pinned_host();
  }

 private:
  NpuLayerImpl impl_{nullptr};
};

// 特化适配器：支持只接受 (ModelContext) 构造的 NPU Layer（如 GLM4、Eagle3）
template <typename NpuLayerImpl>
class NpuDecoderLayerAdapterNoLayerId : public IDecoderLayer {
 public:
  explicit NpuDecoderLayerAdapterNoLayerId(const LayerInitContext& init_ctx)
      : impl_(register_module("impl",
            NpuLayerImpl(init_ctx.model_context))) {}

  // forward / load_state_dict / 权重生命周期方法同上，代理给 impl_
  // ...

 private:
  NpuLayerImpl impl_{nullptr};
};

}  // namespace layer
}  // namespace xllm
```

```cpp
// xllm/core/layers/common/common_decoder_layer_adapter.h

#pragma once
#include "core/layers/i_decoder_layer.h"
#include "core/layers/layer_init_context.h"

namespace xllm {
namespace layer {

// 适配器：支持 (ModelContext) 构造的通用 Layer
template <typename CommonLayerImpl>
class CommonDecoderLayerAdapter : public IDecoderLayer {
 public:
  explicit CommonDecoderLayerAdapter(const LayerInitContext& init_ctx)
      : impl_(register_module("impl",
            CommonLayerImpl(init_ctx.model_context))) {}

  torch::Tensor forward(LayerForwardContext& ctx) override {
    return impl_->forward(
        ctx.x, ctx.residual, ctx.positions,
        *ctx.attn_metadata, ctx.kv_cache, ctx.input_params);
  }

  void load_state_dict(const StateDict& state_dict) override {
    impl_->load_state_dict(state_dict);
  }

 private:
  CommonLayerImpl impl_{nullptr};
};

// 适配器：支持 (ModelContext, layer_id) 构造的通用 Layer（如 MoE 层）
template <typename CommonLayerImpl>
class CommonDecoderLayerAdapterWithLayerId : public IDecoderLayer {
 public:
  explicit CommonDecoderLayerAdapterWithLayerId(
      const LayerInitContext& init_ctx)
      : impl_(register_module("impl",
            CommonLayerImpl(init_ctx.model_context, init_ctx.layer_id))) {}

  torch::Tensor forward(LayerForwardContext& ctx) override {
    return impl_->forward(
        ctx.x, ctx.residual, ctx.positions,
        *ctx.attn_metadata, ctx.kv_cache, ctx.input_params);
  }

  void load_state_dict(const StateDict& state_dict) override {
    impl_->load_state_dict(state_dict);
  }

 private:
  CommonLayerImpl impl_{nullptr};
};

}  // namespace layer
}  // namespace xllm
```

#### 4.2.5 后端无关 Layer 的处理策略

当前 `common/` 下的层实际可分为三类，在注册机制下需要不同的处理方式：

**类型 A：真正的后端无关层**（如 `DenseMLP`、`RMSNorm`、`WordEmbedding`）

纯 `torch::Tensor` 操作，不含任何 `#ifdef`。这些层作为 DecoderLayer 的**内部子组件**使用，不需要直接注册到 `LayerRegistry`。它们在各后端的 DecoderLayer 实现中自然被组合引用。

**类型 B：子组件内部含后端分发的"伪通用"层**（如 `common/attention.h`）

当前的 `common/attention.h` 实际是一个 `#ifdef` 分发器：

```cpp
// 现状：common/attention.h 实际按编译宏选择不同实现
#if defined(USE_MLU)
#include "layers/mlu/attention.h"
#elif defined(USE_CUDA)
#include "layers/cuda/attention.h"
// ...
#endif
```

由此组装的通用 DecoderLayer（如 `Qwen2DecoderLayerImpl`）虽然代码"看起来"后端无关，
但编译后已经绑定了特定后端的 Attention 实现。
这类层可以直接注册为 `"common"` 后端，在 fallback 时使用：

```cpp
// 注册为 "common"，多个后端共享
REGISTER_DECODER_LAYER(qwen2, common,
    [](const LayerInitContext& init_ctx) {
      return std::make_shared<
          CommonDecoderLayerAdapter<layer::Qwen2DecoderLayer>>(init_ctx);
    });
```

任何没有注册专用 Layer 的后端（如 CUDA、MLU、ILU）都会自动回退到这个 `"common"` 注册。

**类型 C：DecoderLayer 自身含 `#ifdef` 的混合层**（如 `Qwen3MoeDecoderLayerImpl`）

这类层内部有 `#ifdef` 选择不同的子组件（如不同后端的 FusedMoE），处理方式有两个选择：

- **短期（兼容现有代码）**：仍然注册为 `"common"`，编译期由 `#ifdef` 决定实际使用哪个 FusedMoE。这不需要改动现有代码。
- **长期（彻底解耦）**：将子组件（FusedMoE 等）也纳入注册机制（见后续"子组件注册"扩展设计），从 DecoderLayer 中消除 `#ifdef`。

**完整的注册查找优先级**：

```
查找 "qwen3::npu"  →  找到？用它
       ↓ 未找到
查找 "qwen3::common"  →  找到？用它（通用 layer 走这里）
       ↓ 未找到
报错：No decoder layer registered
```

**典型注册模式汇总**：

```cpp
// 1. 后端无关的通用 DecoderLayer —— 注册为 "common"，多后端共享
REGISTER_DECODER_LAYER(qwen2, common, [](const LayerInitContext& ctx) {
    return std::make_shared<CommonDecoderLayerAdapter<
        layer::Qwen2DecoderLayer>>(ctx);
});
// → CUDA、MLU、ILU 都会 fallback 到这里

// 2. NPU 有专用实现 —— 按后端注册，优先级高于 common
REGISTER_DECODER_LAYER(qwen2, npu, [](const LayerInitContext& ctx) {
    return std::make_shared<NpuDecoderLayerAdapter<
        layer::NpuQwen2DecoderLayer>>(ctx);
});
// → NPU 优先用这个，不会 fallback 到 common

// 3. 含 #ifdef 的混合层 —— 短期注册为 common，长期拆分
REGISTER_DECODER_LAYER(qwen3_moe, common, [](const LayerInitContext& ctx) {
    return std::make_shared<CommonDecoderLayerAdapterWithLayerId<
        layer::Qwen3MoeDecoderLayer>>(ctx);
});
// → 编译期 #ifdef 仍然生效，各后端拿到的 FusedMoE 实现不同

// 4. 某后端需要覆盖 common 实现（如 MLU 的 DeepSeek V2 有专用优化）
REGISTER_DECODER_LAYER(deepseek_v2, mlu, [](const LayerInitContext& ctx) {
    return std::make_shared<CommonDecoderLayerAdapterWithLayerId<
        layer::DeepseekV2DecoderLayer>>(ctx);  // MLU 专用版
});
```

#### 4.2.6 Layer 工厂注册中心

工厂函数统一接受 `LayerInitContext`：

```cpp
// xllm/core/layers/layer_registry.h

#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include "i_decoder_layer.h"
#include "layer_init_context.h"

namespace xllm {
namespace layer {

using DecoderLayerFactory = std::function<
    std::shared_ptr<IDecoderLayer>(const LayerInitContext& init_ctx)>;

class LayerRegistry {
 public:
  static LayerRegistry& instance() {
    static LayerRegistry inst;
    return inst;
  }

  void register_decoder_layer(const std::string& model_type,
                              const std::string& backend,
                              DecoderLayerFactory factory) {
    auto key = make_key(model_type, backend);
    factories_[key] = std::move(factory);
  }

  DecoderLayerFactory get_factory(const std::string& model_type,
                                  const std::string& backend) const {
    // 优先级：精确匹配 > common 回退
    auto key = make_key(model_type, backend);
    auto it = factories_.find(key);
    if (it == factories_.end()) {
      auto fallback_key = make_key(model_type, "common");
      it = factories_.find(fallback_key);
    }
    CHECK(it != factories_.end())
        << "No decoder layer registered for model=" << model_type
        << " backend=" << backend
        << " (also tried fallback to 'common')";
    return it->second;
  }

  bool has_factory(const std::string& model_type,
                   const std::string& backend) const {
    return factories_.count(make_key(model_type, backend)) > 0 ||
           factories_.count(make_key(model_type, "common")) > 0;
  }

 private:
  static std::string make_key(const std::string& model_type,
                              const std::string& backend) {
    return model_type + "::" + backend;
  }

  std::unordered_map<std::string, DecoderLayerFactory> factories_;
};

// 注册宏
#define REGISTER_DECODER_LAYER(model_type, backend, factory_fn)   \
  static const bool model_type##_##backend##_layer_registered =   \
      []() {                                                      \
        ::xllm::layer::LayerRegistry::instance()                  \
            .register_decoder_layer(#model_type, #backend,        \
                                    factory_fn);                  \
        return true;                                              \
      }()

}  // namespace layer
}  // namespace xllm
```

各后端的注册代码放在各自的 .cpp 文件中，工厂 lambda 统一接受 `LayerInitContext`：

```cpp
// xllm/core/layers/npu/npu_deepseek_v2_decoder_layer_impl.cpp（文件末尾追加）

#include "core/layers/layer_registry.h"
#include "core/layers/npu/npu_decoder_layer_adapter.h"

REGISTER_DECODER_LAYER(deepseek_v2, npu,
    [](const LayerInitContext& init_ctx) {
      return std::make_shared<
          layer::NpuDecoderLayerAdapter<layer::NpuDeepseekV2DecoderLayer>>(
          init_ctx);
    });
```

```cpp
// xllm/core/layers/mlu/deepseek_v2_decoder_layer_impl.cpp（文件末尾追加）

#include "core/layers/layer_registry.h"
#include "core/layers/common/common_decoder_layer_adapter.h"

REGISTER_DECODER_LAYER(deepseek_v2, mlu,
    [](const LayerInitContext& init_ctx) {
      return std::make_shared<
          layer::CommonDecoderLayerAdapterWithLayerId<
              layer::DeepseekV2DecoderLayer>>(init_ctx);
    });
```

```cpp
// xllm/core/layers/npu/npu_glm4_decoder_layer_impl.cpp（使用无 layer_id 适配器）

REGISTER_DECODER_LAYER(glm4, npu,
    [](const LayerInitContext& init_ctx) {
      return std::make_shared<
          layer::NpuDecoderLayerAdapterNoLayerId<
              layer::NpuGlm4DecoderLayer>>(init_ctx);
    });
```

#### 4.2.7 后端策略接口

将 Model forward 中的后端特化预处理逻辑（位置编码计算、attention mask 生成、异步事件管理）抽象为策略接口：

```cpp
// xllm/core/layers/i_backend_strategy.h

#pragma once
#include "layer_forward_context.h"

namespace xllm {
namespace layer {

class IBackendStrategy {
 public:
  virtual ~IBackendStrategy() = default;

  // ====== 构造期回调 ======

  // 创建配套组件（Embedding、Norm 等）
  virtual void init_components(const ModelContext& context) = 0;

  // 在每层 Layer 创建前调用，向 LayerInitContext 注入后端特有的参数
  virtual void populate_layer_init_context(
      LayerInitContext& init_ctx, int layer_id) {}

  // ====== Forward 期回调 ======

  // Embedding lookup
  virtual torch::Tensor embed(torch::Tensor tokens) = 0;

  // 在 forward 循环之前调用，准备公共上下文（cos/sin、attn_mask 等）
  virtual void prepare_forward(torch::Tensor& h,
                               torch::Tensor& positions,
                               const ModelInputParams& input_params,
                               LayerForwardContext& ctx_template) = 0;

  // 每层 forward 前的回调（设置层级参数，如 async event）
  virtual void before_layer(int layer_id,
                            const ModelInputParams& input_params,
                            LayerForwardContext& ctx) {}

  // 每层 forward 后的回调
  virtual void after_layer(int layer_id,
                           LayerForwardContext& ctx) {}

  // 最终 norm 处理
  virtual torch::Tensor finalize(torch::Tensor& h,
                                 std::optional<torch::Tensor>& residual) = 0;

  // ====== 权重生命周期（统一代理给 embedding、norm 等组件）======

  virtual void load_state_dict_components(const StateDict& state_dict) = 0;
  virtual void merge_loaded_weights_components() {}
  virtual void free_weights_components() {}
  virtual void reload_weights_components() {}
};

}  // namespace layer
}  // namespace xllm
```

NPU 策略实现示例：

```cpp
class NpuBackendStrategy : public IBackendStrategy {
 public:
  void init_components(const ModelContext& context) override {
    embed_ = layer::NpuWordEmbedding(context);
    norm_ = layer::NpuRMSNorm(context);
    pos_emb_ = layer::NpuPosEmbedding(context);
    // 初始化 cos_sin_, attn_mask_ 等
  }

  void prepare_forward(torch::Tensor& h,
                       torch::Tensor& positions,
                       const ModelInputParams& input_params,
                       LayerForwardContext& ctx_template) override {
    auto cos_sin = pos_emb_(cos_sin_, positions, 0);
    auto chunks = cos_sin.chunk(2, -1);
    ctx_template.cos_pos = chunks[0].contiguous();
    ctx_template.sin_pos = chunks[1].contiguous();
    ctx_template.attn_mask = compute_attn_mask(input_params);
  }

  void before_layer(int layer_id,
                    const ModelInputParams& input_params,
                    LayerForwardContext& ctx) override {
    ctx.layer_id = layer_id;
    if (input_params.layer_synchronizer) {
      ctx.async_event = input_params.layer_synchronizer->get_event(layer_id);
      ctx.event_flag = input_params.layer_synchronizer->get_event_flag(layer_id);
    }
  }

  torch::Tensor finalize(torch::Tensor& h,
                         std::optional<torch::Tensor>&) override {
    return norm_(h, 0);
  }

  torch::Tensor embed(torch::Tensor tokens) override {
    return embed_(tokens, 0);
  }

 private:
  layer::NpuWordEmbedding embed_{nullptr};
  layer::NpuRMSNorm norm_{nullptr};
  layer::NpuPosEmbedding pos_emb_{nullptr};
  torch::Tensor cos_sin_;
  layer::AttentionMask attn_mask_;
};
```

#### 4.2.8 统一模型基类

```cpp
// xllm/models/llm/unified_llm_model_base.h

template <typename BackendStrategyType>
class UnifiedLlmModelImplBase : public torch::nn::Module {
 public:
  UnifiedLlmModelImplBase(const std::string& model_type,
                          const ModelContext& context) {
    auto backend_name = context.get_backend();  // "npu" / "cuda" / "mlu"
    auto factory = layer::LayerRegistry::instance()
                       .get_factory(model_type, backend_name);

    strategy_ = std::make_unique<BackendStrategyType>();
    strategy_->init_components(context);

    auto model_args = context.get_model_args();
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      // 构造 LayerInitContext，传给工厂
      layer::LayerInitContext init_ctx{
          .model_context = context,
          .layer_id = i,
          .total_layers = model_args.n_layers(),
          .backend = backend_name,
      };
      // 后端策略可在此注入 extra 参数
      strategy_->populate_layer_init_context(init_ctx, i);
      layers_.push_back(factory(init_ctx));
    }
  }

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    auto h = strategy_->embed(tokens);

    // 构造 forward 上下文模板
    layer::LayerForwardContext ctx_template{
        .x = h,
        .kv_cache = kv_caches[0],
        .input_params = input_params,
    };
    strategy_->prepare_forward(h, positions, input_params, ctx_template);

    for (size_t i = 0; i < layers_.size(); i++) {
      ctx_template.kv_cache = kv_caches[i];
      strategy_->before_layer(i, input_params, ctx_template);
      h = layers_[i]->forward(ctx_template);
      strategy_->after_layer(i, ctx_template);
    }

    return ModelOutput(
        strategy_->finalize(h, ctx_template.residual));
  }

  void load_state_dict(const StateDict& state_dict) {
    strategy_->load_state_dict_components(state_dict);
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

 protected:
  std::unique_ptr<BackendStrategyType> strategy_;
  std::vector<std::shared_ptr<layer::IDecoderLayer>> layers_;
};
```

---

## 5. 新增后端的工作流

以新增 **XPU** 后端为例，适配 DeepSeek V2 模型只需要：

### 步骤 1：实现 Layer

可以直接实现 `IDecoderLayer`（推荐新后端），或用 Adapter 包装已有实现：

```cpp
// 方式 A：直接实现接口（推荐新后端）
// xllm/core/layers/xpu/xpu_deepseek_v2_decoder_layer_impl.h
class XpuDeepseekV2DecoderLayerImpl : public IDecoderLayer {
 public:
  explicit XpuDeepseekV2DecoderLayerImpl(const LayerInitContext& init_ctx) {
    // 从 init_ctx 获取所有需要的参数
    auto& args = init_ctx.model_args();
    int32_t layer_id = init_ctx.layer_id;
    // 可通过 extra_params 获取后端特有参数
    bool use_xpu_flash_attn = init_ctx.get_extra<bool>(
        "use_xpu_flash_attn", true);
    // ...
  }
  torch::Tensor forward(LayerForwardContext& ctx) override { /* ... */ }
  void load_state_dict(const StateDict& state_dict) override { /* ... */ }
};
```

```cpp
// 方式 B：用 Adapter 包装已有实现（快速适配）
// 如果 XPU 已有 Layer 但签名不同，写一个 Adapter 即可
```

### 步骤 2：注册 Layer

```cpp
// xllm/core/layers/xpu/xpu_deepseek_v2_decoder_layer_impl.cpp
REGISTER_DECODER_LAYER(deepseek_v2, xpu,
    [](const LayerInitContext& init_ctx) {
      return std::make_shared<XpuDeepseekV2DecoderLayerImpl>(init_ctx);
    });
```

### 步骤 3：实现 BackendStrategy（如果不能复用通用策略）

```cpp
class XpuBackendStrategy : public IBackendStrategy {
  void populate_layer_init_context(
      LayerInitContext& init_ctx, int layer_id) override {
    // 注入 XPU 特有的构造参数
    init_ctx.set_extra("use_xpu_flash_attn", true);
  }
  // ...
};
```

**无需新增或修改任何模型文件**。

---

## 6. 与现有架构的对比

| 维度 | 当前方式 | Layer 工厂 + 依赖注入 |
|------|---------|---------------------|
| **新增后端** | 复制整套模型文件，改 Layer 类型 | 只实现 Layer + 注册，无需改模型 |
| **新增模型** | 每个后端写一份模型文件 | 写一份统一模型 + 各后端注册 Layer |
| **代码量** | O(N × M)，N=模型数，M=后端数 | O(N + N×M) 但每个后端只写 Layer |
| **模型逻辑改动** | 需在 M 个文件同步修改 | 只改一处统一模型 |
| **现有代码侵入** | - | 零侵入（通过 Adapter 兼容） |
| **后端选择时机** | 编译期（`#ifdef`） | 可支持运行时选择 |
| **性能开销** | 无（直接调用） | 虚函数调用开销（可忽略） |

---

## 7. 渐进式实施计划

改造范围大、风险高，建议分阶段推进：

### Phase 1：基础设施（预计 1-2 周）

- [ ] 定义 `LayerForwardContext` 结构体
- [ ] 定义 `IDecoderLayer` 接口
- [ ] 实现 `LayerRegistry` 工厂注册中心
- [ ] 编写单元测试验证接口设计

**产出**：新增文件，不影响现有代码编译和运行。

### Phase 2：适配器验证（预计 2-3 周）

- [ ] 实现 `NpuDecoderLayerAdapter` 模板
- [ ] 实现 `CommonDecoderLayerAdapter` 模板
- [ ] 选择一个模型（如 Qwen3）作为试点，为其 NPU 和 CUDA 实现编写适配器并注册
- [ ] 验证通过适配器调用的正确性和性能

**产出**：Qwen3 模型可通过统一接口在 NPU 和 CUDA 上运行。

### Phase 3：统一模型基类（预计 2-3 周）

- [ ] 定义 `IBackendStrategy` 接口
- [ ] 实现 `NpuBackendStrategy` 和 `CommonBackendStrategy`
- [ ] 实现 `UnifiedLlmModelImplBase`
- [ ] 将 Qwen3 迁移到统一模型基类
- [ ] 对比迁移前后的性能（确保无回归）

**产出**：Qwen3 模型只有一份模型文件，通过策略模式适配不同后端。

### Phase 4：全面推广（预计 4-6 周）

- [ ] 将其他模型（DeepSeek V2/V3、Llama、GLM4 等）逐步迁移
- [ ] 为 MLU、MUSA、ILU 后端编写适配器
- [ ] 清理废弃的后端特化模型文件
- [ ] 更新 `models.h` 的编译期选择逻辑

**产出**：所有模型使用统一架构，`models.h` 大幅简化。

### Phase 5：高级特性（可选）

- [ ] 支持运行时后端选择（`ModelContext` 携带 backend 标识，运行时查表）
- [ ] 支持同一二进制多后端（移除 `#ifdef` 隔离，转为运行时注册）
- [ ] Layer 级别的性能 profiling 框架

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 虚函数调用引入性能开销 | Layer forward 是热路径 | 经验证虚函数调用开销 < 1μs/call，相对 Layer 计算时间（~ms 级）可忽略；必要时可用 CRTP 零成本多态 |
| `LayerForwardContext` 膨胀 | 随后端增多字段越来越多 | 使用 `std::any` 或 `void*` 的扩展字段 map，仅核心字段放主体 |
| Adapter 层增加代码理解成本 | 间接调用链变长 | 适配器是模板生成的薄层，IDE 可直接跳转到底层实现 |
| NPU forward 签名与通用差异大 | 适配器需要处理参数映射 | 已在 `LayerForwardContext` 设计中充分考虑，通过 optional 字段覆盖所有场景 |
| 全面迁移周期长 | 过渡期两套代码并存 | 分阶段推进，旧代码在新架构验证通过后再清理 |

---

## 9. 附录

### A. 关键文件索引

| 文件 | 说明 |
|------|------|
| `xllm/core/layers/qwen2_decoder_layer.h` | 通用 decoder layer 实现示例 |
| `xllm/core/layers/npu/npu_base_layer.h` | NPU layer 基类（ATB 执行引擎） |
| `xllm/core/layers/npu/npu_deepseek_v2_decoder_layer_impl.h` | NPU DeepSeek V2 layer 实现 |
| `xllm/core/layers/npu/loader/base_loader.h` | NPU 权重加载基类 |
| `xllm/core/layers/mlu/deepseek_v2_decoder_layer_impl.h` | MLU DeepSeek V2 layer 实现 |
| `xllm/models/llm/llm_model_base.h` | 通用模型模板基类 |
| `xllm/models/llm/npu/llm_model_base.h` | NPU 模型模板基类 |
| `xllm/models/model_registry.h` | 模型注册中心（现有） |
| `xllm/models/models.h` | 编译期后端选择入口 |

### B. 新增文件清单（Phase 1-3）

| 文件 | 说明 |
|------|------|
| `xllm/core/layers/layer_init_context.h` | 统一 Layer 构造上下文 |
| `xllm/core/layers/layer_forward_context.h` | 统一 forward 上下文 |
| `xllm/core/layers/i_decoder_layer.h` | Layer 抽象接口 |
| `xllm/core/layers/layer_registry.h` | Layer 工厂注册中心 |
| `xllm/core/layers/i_backend_strategy.h` | 后端策略接口 |
| `xllm/core/layers/npu/npu_decoder_layer_adapter.h` | NPU 适配器模板 |
| `xllm/core/layers/common/common_decoder_layer_adapter.h` | 通用适配器模板 |
| `xllm/models/llm/unified_llm_model_base.h` | 统一模型基类 |
