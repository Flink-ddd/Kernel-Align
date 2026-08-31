# WS2 Attention Transformer Engine 复用方案

Status: #235 设计补充

## 设计结论

Transformer Engine（TE）在 #235 中只能是显式 opt-in 的 validation oracle
或 backend candidate，不是 RL-Kernel attention 语义的可信源。可信源仍然是
RL-Kernel 自己的 `AttentionContract`、RoPE/cache metadata、
attention-domain `lse`、固定 `global_block_index` merge 顺序、
deterministic reference 和 drift report。

TE 复用分为三层：

| 层级 | TE 角色 | 允许范围 |
| --- | --- | --- |
| Merge oracle | 复用 TE context-parallel correction helpers 校验 `(out, lse)` online-softmax merge | PR2、PR3、PR5、PR6 |
| Fused forward candidate | 评估 `DotProductAttention` 作为 opt-in 生产后端候选 | 仅 PR7 |
| Backward oracle | 仅在 TE 暴露兼容 saved forward state 时，通过 autograd/backward 对比 `dq/dk/dv` | 仅 PR8 |

## Merge Oracle Contract

对任意 Q row，RL-Kernel 先按逻辑 KV block 生成 partial states：

```text
state_i = (out_i, lse_i, global_block_index_i)
```

其中 `out_i` 是本地 KV block 内已经归一化的 attention output，`lse_i`
是 attention-domain LSE，shape 为 `[B, Hq, Sq]`。所有 state 必须按
`global_block_index` 排序后再合并：

```text
lse_new = logaddexp(lse_prev, lse_i)
out_new = exp(lse_prev - lse_new) * out_prev
        + exp(lse_i    - lse_new) * out_i
```

TE helper 可以负责 correction arithmetic，但语义输入必须由 RL-Kernel 提供：

```text
TE_merge(sorted(RL-Kernel partial states)) == RL-Kernel_merge(sorted(partial states))
```

调用 TE 前，RL-Kernel 必须保证：

- merge accumulation 使用 FP32，只有 `final_write` 才 downcast；
- merge 顺序来自逻辑 `global_block_index`，不是通信 arrival order；
- all-masked / empty-KV row 保持 `lse = -inf`、`out = 0`，不能产生 NaN；
- TE adapter 启用前必须完成 capability probe：module/symbol 存在、helper
  signature 兼容、tiny numeric merge smoke 通过；
- RoPE state、causal/padding mask、packed/varlen boundary、cache position 已经对齐。

## PR-level TE Plan

| PR | RL-Kernel 核心功能 | TE 复用方式 | 精确 TE API | RL-Kernel 必须准备 | Gate / fallback |
| --- | --- | --- | --- | --- | --- |
| PR1 / #236 | 定义 attention contract、sharding/reduction metadata、RoPE/cache 字段 | 不调用 TE；只预留 `transformer_engine` 作为未来显式 backend 名称 | 无 | backend、reduction、`lse_domain`、`merge_order`、RoPE/cache identity 字段 | 不依赖 TE；metadata 缺失仍由 RL-Kernel contract fail |
| PR2 / #253 | 单 GPU full/chunked/paged-KV attention comparison harness | optional paged-KV merge oracle | `transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py`；`transformer_engine.pytorch.attention.dot_product_attention.context_parallel`；`flash_attn_fwd_softmax_lse_correction`；`flash_attn_fwd_out_correction_init`；`flash_attn_fwd_out_correction` | 相同 Q/K/V、相同 causal/padding metadata、相同 KV page order、RL-Kernel partial states `(out_i, lse_i)` | 对比 `TE_merge(partials)` 和 `RL-Kernel_merge(partials)` 的 `out/lse`；TE 不可用时 report `unavailable` |
| PR3 / #238 | post-RoPE Q/K 上的 deterministic CP attention reference | optional CP merge oracle test | 同 PR2 的 `context_parallel.py` module/functions | post-RoPE Q/K boundary、CP partial states、不重叠 global KV block ranges、固定 merge order | TE 不可用时 skip；TE 不定义 reference path |
| PR4 | Qwen3-8B TP=2 CP=2 BF16 cross-config 集成和 backend provenance | policy/provenance only | 不新增 TE 调用 | runtime descriptor 可 request `transformer_engine`，但默认执行仍是 deterministic reference | 记录 requested backend、actual backend、fallback reason、TE availability；禁止 silent fallback |
| PR5 | 分布式 prefill/chunked-prefill drift benchmark 和 report artifacts | benchmark merge oracle | 通过 `TEContextParallelMergeAdapter` 调用同 PR2 的 `context_parallel.py` module/functions | 与 RL-Kernel merge 完全相同的 gathered CP partial states、per-rank block metadata hash、FP32 merge dtype | 报告 `merge_drift = drift(TE_merge(partials), RL-Kernel_merge(partials))`；benchmark 可 provenance fallback |
| PR6 | decode-stage KV-cache CP attention replay | decode / paged-KV merge oracle only | 通过 decode TE merge adapter 调用同 PR2 的 `context_parallel.py` module/functions | `cache_position`、`kv_seq_lens`、page table、prefix-cache identity、global token positions、RoPE cache state、sorted logical page/block order | TE 只验证 `(out, lse)` merge；cache/page identity 不一致时，在调用 TE 前 fail |
| PR7 | deterministic reference 稳定后的 fused prefill/decode backend alignment | full fused forward backend candidate | `transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py`；`transformer_engine.pytorch.DotProductAttention`；actual backend 可观测时记录为 `FlashAttention` / `FusedAttention` / `UnfusedDotProductAttention` | 精确 layout / `qkv_format`、mask mode、RoPE fusion boundary、dtype、scale placement、dropout=0 correctness mode、deterministic controls、LSE export capability、actual-backend 观测方式 | 只有 TE output、attention-domain LSE、actual backend provenance 都能对齐/记录时才可作为 production candidate；如果不能导出 LSE 或不能观测 actual backend，只能算 exploratory，并记录原因 |
| PR8 | training backward CP attention reference 和 gradient drift validation | optional backward oracle | `DotProductAttention` autograd/backward path，仅当 compatible saved forward state 暴露时使用 | 与 RL-Kernel reference 相同的 forward inputs/metadata：`out`、attention-domain `lse`、masks、RoPE state、sequence/cache metadata、CP block ownership | 对比 `dq/dk/dv`；没有兼容 TE backward state 时明确写 `not used`，不能宣称复用 TE backward |

## Capability / Provenance Checklist

任何 PR 只要提到 TE，都必须写清：

```text
te_available, te_version, te_module, te_symbols
te_capability_probe, te_signature_checked, te_numeric_selftest
requested_backend, actual_backend, actual_backend_source
fallback, fallback_reason
attention_mode, dtype, layout/qkv_format, mask_alignment
lse_domain, lse_exported, merge_order, accum_dtype, downcast_at
split_kv_policy, paged_kv_policy, cp_block_metadata_hash
scale_placement, deterministic_controls, dropout_policy, te_env_controls
```

fallback 策略：

| 场景 | TE 不可用 / capability 不匹配时 |
| --- | --- |
| optional oracle test | skip / report unavailable |
| benchmark exploration | provenance fallback 到 deterministic reference |
| correctness gate | fail closed |
| production backend | fail closed 或显式 provenance fallback；禁止 silent fallback |

## 不宣称的事

- 不把 TE 设为 #235 的硬依赖。
- 不用 TE API 反向定义 RL-Kernel contract。
- 不在 metadata 不完整时 silent fallback 到 TE。
- 不用 NCCL / TE arrival order 决定 attention merge 数值顺序。
- 不在 PR7 前把 TE fused path 宣称为默认生产路径。
- PR7 如果拿不到 attention-domain LSE，不宣称完整 correctness closure。
- PR8 如果拿不到兼容 backward state，不宣称复用 TE backward。

## 最终判断标准

TE 可以帮助验证和加速，但 #235 的正确性仍由 RL-Kernel 自己的 contract、
metadata、deterministic reference 和 drift report 保证。当前最值得复用的是
TE context-parallel correction helper；完整 `DotProductAttention` 路径只有在
显式声明 capability 并满足 RL-Kernel 语义契约后，才允许作为生产候选后端。
