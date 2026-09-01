# Archived v1 /goal — FastWAM Policy Adapter with Paired Intervention Supervision

> **ARCHIVED / SUPERSEDED（2026-08-17）**：以下正文是旧三场景 smoke 的原始目标，为保持历史可追溯性不做改写。它不再是正式 Policy 执行规范；当前唯一权威协议是 [`policy_protocol_v2.md`](policy_protocol_v2.md)，方法解释和实施顺序分别见 [`policy_method.md`](policy_method.md) 与 [`实验规划.md`](实验规划.md)。旧 smoke 的 PASS 不能解释为 v2 原生 50 Hz 数据、C2、正式训练或在线评测已经完成。

在 `/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM` 中，基于当前已经完成的 E0–E3 representation experiments，实现一个最小的 policy-level prototype。

目标不是重新设计 FastWAM，而是验证：

> E0–E3 学到的 background-invariant content representation，能否真正提升 RoboTwin Random rollout success。

请先完整检查现有 FastWAM action path、E0/E1/E2/E3 content head、训练入口、checkpoint loading 和 RoboTwin rollout/evaluation pipeline，再开始修改。

不要直接大规模训练。先完成实现、静态检查、gradient audit 和小规模 smoke，最后停下来给出结果与正式运行命令，等待人工审阅。

---

## 1. Hard Constraints

禁止破坏现有 E0–E3。

禁止直接修改原 FastWAM 模型语义或覆盖 release checkpoint。

第一版必须保持：

```text
VAE            Frozen
Video Backbone Frozen
```

不要做 full fine-tuning。

所有新增代码优先放在独立目录，例如：

```text
experiments/robotwin/policy_content_adapter/
```

如必须在 `src/fastwam/**` 增加最小 hook，请：

1. 保持默认行为完全不变；
2. 只有显式 config enable 时才激活；
3. 写数值等价 / identity test；
4. 清楚列出修改位置。

---

## 2. Reuse Existing E0–E3 Content Head

先找到当前 E0–E3 已验证通过的 content head 实现。

保持其核心结构：

```text
Layer-16 tokens: 120 × 3072
→ Linear(3072, 384)
→ 8 learnable content queries
→ 1-layer 8-head cross-attention
→ Zc: 8 × 384
```

Policy 路径保留：

```text
Zc ∈ R^(8×384)
```

不要先 mean pool。

Contrastive 路径：

```text
Zc
→ mean pool
→ Linear(384,384)
→ SiLU
→ Linear(384,384)
→ L2 normalize
→ z ∈ R^384
```

要求：

- 尽量直接复用 E1–E3 权重兼容的代码；
- 不重新设计 head；
- 输出 trainable parameter count。

---

## 3. Implement Gated Cross-Attention Action Adapter

找到 FastWAM ActionDiT action tokens 的合理注入位置。

第一版只增加一个 minimal gated cross-attention adapter：

\[
\Delta X_a = CrossAttn(X_a,Z_c,Z_c)
\]

\[
X_a' = X_a + \tanh(g)\Delta X_a
\]

要求：

```text
gate g initialized to exactly 0
```

因此初始化时：

```text
adapter_enabled model output ≈ original model output
```

请实现 identity/numerical check，确认 zero-init 时不会显著改变原 policy 输出。

第一版只插 1 个 GCA，不实现 every-layer / every-4-block。

---

## 4. Implement Two Policy Modes

### P-v1

```text
VAE            Frozen
Video Backbone Frozen
Content Head   Train
Action Adapter Train
ActionDiT      Frozen
```

注意：

ActionDiT frozen 只意味着参数不更新。

`L_action` 的梯度必须仍可经过 frozen ActionDiT 回传到：

```text
Action Adapter
Content Head
```

实现 gradient audit，确认：

```text
grad(ContentHead) != 0
grad(Adapter) != 0
grad(ActionDiT params) == None / 0
grad(Video Backbone params) == None / 0
```

### P-v2

```text
VAE            Frozen
Video Backbone Frozen
Content Head   Train
Action Adapter Train
ActionDiT      Train
```

默认学习率：

```text
Content Head / Adapter LR = 1e-4
ActionDiT LR              = 1e-5
```

要求 param group 清晰可审计。

---

## 5. Dual-Stream Training

不要简单把两套数据 concat 成一个 dataset。

每个 optimizer step 同时取：

```python
official_batch = next(official_loader)
paired_batch   = next(paired_loader)
```

### Official stream

使用已有 RoboTwin official policy dataset / loader。

根据 setting：

```text
Clean
or
Clean + Random
```

只计算：

\[
L_{\rm action}(D_{\rm official})
\]

### Paired stream

复用 E2/E3 数据：

```text
Clean + R1 + R2
```

同一个 physical state 的：

```text
C_i, R1_i, R2_i
```

互为 positives。

同 task 不同 physical state 为 negatives。

只计算：

\[
L_{\rm ctr}(D_{\rm paired})
\]

不要让 R3 进入 paired training。

总 loss：

\[
L=
L_{\rm action}
+
\lambda_{\rm ctr}L_{\rm ctr}
\]

默认：

```text
lambda_ctr = 0.1
temperature tau = 0.07
```

第一版 Video Backbone frozen，因此不要重新引入 video reconstruction / video flow-matching loss。

---

## 6. Data Provenance / Distribution Safety

我们自己生成的 C/R1/R2 可能与 RoboTwin official policy dataset 存在 distribution gap。

因此主方法必须遵守：

```text
Official data -> action supervision
Our paired C/R1/R2 -> contrastive supervision
```

不要在主方法中用 Our C/R1/R2 替代 official policy data。

先增加一个轻量 audit：

- official Clean 的 camera/resolution/frame convention；
- our C 的 camera/resolution/frame convention；
- proprio/action/task prompt convention；
- 必要时提取同一 Layer-16 feature，给出简单分布统计。

只报告，不因为 audit 自动改变数据。

---

## 7. Required Policy Controls

实现配置，不一定在第一轮全部正式跑完。

### C0 Original / continuation baseline

无 Content Head，无 Adapter，无 contrastive。

### C1 Architecture-only

与 Ours 使用相同 Content Head + Adapter，但：

```text
lambda_ctr = 0
```

只计算：

```text
L_action
```

用于排除：

> 只是增加参数 / adapter 就提升。

### C2 Naive Augmentation

作为 ablation：

```text
Our C/R1/R2 直接加入 action supervision
No contrastive loss
```

这组只是为了回答：

> 单纯多看背景是否已经足够？

不要把它作为主训练方式。

### C3 Ours

```text
Official data -> L_action
Paired C/R1/R2 -> L_ctr
```

---

## 8. Main Four Dataset Settings

先准备 config，不要在 smoke 通过前自动启动全部正式训练。

### Clean

起点：

```text
Clean-only FastWAM checkpoint
```

继续用 official Clean 做 matched-step action-only training。

### Clean + Aug

从同一个 Clean-only checkpoint 开始：

```text
official Clean -> L_action
our C/R1/R2 -> L_ctr
```

### Clean + Random

起点：

```text
Clean+Random FastWAM checkpoint
```

继续用 official Clean+Random 做 matched-step action-only training。

### Clean + Random + Aug

从同一个 Clean+Random checkpoint 开始：

```text
official Clean+Random -> L_action
our C/R1/R2 -> L_ctr
```

确保：

```text
Clean vs Clean+Aug
```

以及：

```text
Clean+Random vs Clean+Random+Aug
```

分别具有：

- 相同初始化 checkpoint；
- 相同 continuation steps；
- 相同 rollout protocol；
- 只有 paired intervention supervision 不同。

如果 Clean-only checkpoint 路径无法从 repo/config 中确定，不要猜；在最终报告中明确列出需要人工提供的 checkpoint path。

---

## 9. Three-Task Smoke

先只使用已有 E0–E3 的三个任务：

```text
move_stapler_pad
open_microwave
place_a2b_left
```

Smoke 目标不是得到正式科学数字，而是确认：

1. checkpoint 正常加载；
2. Layer-16 hook 正确；
3. Content Head 输出 shape 正确；
4. GCA zero-init identity 成立；
5. P-v1 gradient flow 正确；
6. P-v2 param groups 正确；
7. dual-loader 正常；
8. loss finite；
9. 一个极短训练后 gate / head 参数确实更新；
10. rollout pipeline 能加载新 checkpoint 并执行。

不要自动启动长时间正式训练。

---

## 10. Rollout Evaluation Interface

准备统一评估命令：

```text
Clean rollout
Official Random rollout
```

如果当前已有 background-only R3 rollout 环境，则同时支持：

```text
R3 background-only rollout
```

但不要为了这个目标大改 RoboTwin evaluator。

最终正式指标必须包括：

```text
Success Rate on Clean
Success Rate on Random
```

Representation metrics：

```text
Style distance
State distance
State/Style ratio
R3 -> Clean R@1
```

只作为辅助解释，不替代 rollout success。

---

## 11. Logging

每次 run 至少记录：

```text
total loss
action loss
contrastive loss
positive similarity
negative similarity
gate value
Content Head grad norm
Adapter grad norm
ActionDiT grad norm (P-v2)
trainable parameter count
learning rates
```

保存：

```text
run_config.json
train_log.csv
training_summary.json
checkpoint
```

---

## 12. Deliverables

Smoke 完成后停下来，提供：

1. 代码修改文件列表；
2. FastWAM 原 action path 的理解；
3. Content Head 复用方式；
4. GCA 插入位置与原因；
5. zero-init identity test 数值；
6. P-v1 gradient audit；
7. P-v2 gradient audit；
8. trainable parameter counts；
9. dual-stream data provenance；
10. official Clean vs our C audit；
11. smoke loss / gate / gradient 日志；
12. 新 checkpoint 是否能被 rollout evaluator 正常加载；
13. P-v1 完整运行命令；
14. P-v2 完整运行命令；
15. Architecture-only / Naive Aug / Ours 的 config；
16. 四组正式实验 config；
17. README_POLICY_CONTENT_ADAPTER.md。

最后给出：

```text
git status
git diff --stat
```

不要自动提交 git commit，除非当前 repo workflow 明确要求。

---

## 13. Explicit Non-Goals

本次不要实现：

```text
LOVO
router
SIVA full architecture
multi-layer adapter sweep
query-number sweep
full Video Backbone fine-tuning
teacher model
EMA
memory bank
new video loss
```

当前唯一核心问题：

> Can the E0–E3 background-invariant content representation improve actual RoboTwin Random rollout success when injected only into the action-facing policy branch?
