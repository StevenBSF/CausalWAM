# FastWAM Policy Content Adapter

> **Policy Protocol v2 状态（2026-08-19）**：当前权威规范是 [`docs/policy_protocol_v2.md`](../../../../docs/policy_protocol_v2.md)。主线已改为作者 RoboTwin 50-task release checkpoint 上的配对 C1/C3；C0 仅为可选外部 reference，方法结构与 loss 不变。原三任务 Stage 1 及其后续复验移动到扩展实验。下方 legacy 记录中的三场景 C/R1/R2、R3 holdout、E2 Head 初始化和旧正式四组不能作为 v2 正式协议。

> **P-v2 后续机制实验（2026-08-22）**：P-v1 的 36-cell 正式结果已经完成并继续作为主实验。新增的 ActionDiT 可训练实验属于查看主结果后的 post-hoc mechanism study，使用独立输出根，不能覆盖或替代 P-v1。完整协议见 [`docs/pv2_actiondit_followup.md`](../../../../docs/pv2_actiondit_followup.md)。

## Policy Protocol v2 摘要

### 当前主线基座与控制

当前固定基座 `B_release`：

```text
/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
sha256: 776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63

/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
sha256: 7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095
```

这是作者发布的全 RoboTwin release checkpoint。关联 release 数据包含 27,500 episodes，即 50 个任务块 × 550；每块前 50 Clean、后 500 Official Random 是本协议绑定的分区规则，不是 checkpoint payload 自带的 domain 字段。三个目标任务已参与预训练，因此当前实验是 pretrained-policy adaptation，不是从头训练或未见任务泛化。

所有主线分支从同一个 `B_release` 分叉：

| Control | Head/GCA | Official C+R | 新 Paired C/R1/R2/R3 |
|---|---|---|---|
| C0 Author Release | 无 | 不继续训练 | 无（可选 reference） |
| C1 Architecture-only（action-only continuation） | 有 | Action loss | `lambda_ctr=0` |
| C3 Ours | 与 C1 相同 | Action loss | `lambda_ctr>0` Contrastive loss |
| C2 Naive Aug | 与 C3 相同 | Action loss | Action loss（后续辅助） |

C1/C3 在 Stage 2 seeds `[1,2,3]` 内逐一配对，使用完全相同的 Head/GCA 初始 tensor、official/paired 数据顺序、训练步数和公共随机流；C1 将 paired loss 乘零，唯一额外梯度来自 C3 contrastive supervision。唯一主结论是 `C3-C1`。strict aggregate 使用 schema v4/profile `c1_c3_primary`，强制 36 条主 records。C0 可省略；若附加必须是相同协议/seed bank 的完整六格。E2 Head 和 C2 都保留为主结果之后的辅助实验。

原三任务 `B_CR` Stage 1 不取消，但后移为基座敏感性扩展。其 seed/OOM/batch8 amendment 和所有历史产物继续保留，不进入当前 release-base 主表。

### 新 Policy paired 数据

旧 30 Hz C/R1/R2/R3 数据不插值到 50 Hz，也不用于正式 Policy C2/C3。新数据必须原生包含：

```text
first-person camera + left-wrist camera + right-wrist camera
+ robot state
+ native 50 Hz state/action trajectory
+ future 32-step × 14-D action target
```

采集端固定使用 RoboTwin 原生 `Large_D435`，三个相机均直接渲染为 `640×480`；禁止把普通 `D435` 的 `320×240` 输出事后上采样后冒充官方分辨率。

C、R1、R2、R3 是同一物理状态的四种**场景版本，不是四个相机**；每种场景内部仍有三个同步相机。专家只执行一次，其他版本从相同物理状态重渲染，并共享完全相同的 robot state 和 action target。

协议数量为：

| Split | Clean | Random（R1+R2+R3） |
|---|---:|---:|
| Train | 30 | 90 |
| Val | 10 | 30 |
| Test | 10 | 30 |
| 总计 | 50 | 150 |

采集遵循每任务 `1 physical trajectory × C/R1/R2/R3` pilot gate，审计三相机同步、原生 50 Hz、`[32,14]` action windows、物理状态/动作逐项一致和完整 provenance，随后才扩展完整 split。

当前完整 artifact 已存在于 `outputs/policy_content_adapter/native50hz_three_task_rgb640x480_v1/full_lerobot_v21`。其 export audit 已 PASS：600 scene episodes、50 contents/task、50 Hz、32-step × 14D、`all_pairs_exact=true`、无插值。新主线仍须将它与 `B_release` lineage、state bank 和 split manifest 重新 SHA-lock；数据 PASS 不等于 cache/训练/rollout 完成。

Train split 每条 trajectory 固定选 8 个无末端 padding 的 states，形成 240 groups/task、720 groups/三任务、2,880 scene views；C2/C3 绑定同一个 state-bank SHA。

### Cache 与评测

必须从锁定 SHA 的 `B_release` 对新 native-50Hz paired 数据重提 Layer-16 cache，例如 `policy_release50tasks_native50hz_four_scene_v1`。不得复用、拼接或覆盖 E0–E3 cache，也不得使用未完成三任务 `B_CR` 的 cache。

P-v1/P-v2 需要基于 `B_release` 和 C1/action-only 目标（`lambda=0`）重新短训，并使用同一个 `dev_selection` seed bank 各执行 20 Clean/official Random episodes/task/domain。固定规则是：Clean macro 距最佳值不能低超过 0.05，再按 official Random macro 选高者，完全打平选 P-v1。不得用 C3 contrastive dev 结果选择共享 P 模式。选择结果写入 hash-bound manifest；正式 `final_test` seed bank 必须与 dev bank 不相交。旧 smoke 没有 simulator success rate，不能据其 loss 或梯度数字选择正式模式。

Policy 最终只评测：

- `demo_clean`；
- `demo_randomized`。

主表报告三任务 SR、macro average、C1/C3 三个 Stage 2 seeds 的 mean ± std，以及主比较 C3−C1。已有同协议完整 C0 时才附加 C3−C0 总体适配参考；C0 不阻塞当前实验。R3 success 和 Style/State/Ratio/R@1 只属于 E0–E3 representation 结果。

### v2 release-base 工程 smoke

当前 release lineage、600-scene binding、paired prompt cache 与从作者 release
重新提取的 Layer-16 cache 已具备真实 SHA 绑定。C1/C3 的三步工程 gate 使用同一个
seed、Head/GCA 初值、official sample 顺序和 paired physical-state 顺序；唯一处理差异
是 C1 `lambda_contrastive=0`，C3 `lambda_contrastive=0.1` 及对应梯度开关。

官方 68,704 条 prompt cache（约 72 GiB）已有一次完整 payload 重载审计，并绑定为
`outputs/policy_content_adapter/release_base_v1/official_text_cache_binding_manifest.json`。
每个 Stage-2 run 只复核 binding/audit/inventory SHA，并在读取时验证实际 prompt，禁止
每次重复扫描 72 GiB。

顺序运行单卡 C1/C3 三步训练、compact checkpoint 保存、每任务一个 action 的部署检查
以及严格 pair audit：

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
GPU_ID=0 REGIME=p_v1 \
  bash experiments/robotwin/policy_content_adapter/run_release_c1_c3_engineering_smoke.sh
```

该 gate 是非正式工程证据，不是 P-v1/P-v2 选择、在线 success rate 或正式 C3-C1 结果，
也不会自动启动后续 dev pilot/formal training。

### C0 author-release dev deployment gate（已完成的可选工程证据）

C0 首先使用一个独立 `development_analysis` seed bank 做部署门槛：真实
`B_release` 输入上的原生/零门控动作路径必须逐位一致，然后在三任务的
`demo_clean` 与 `demo_randomized` 各执行 1 个有效 episode，共 6 个。在线入口会把
CUDA inference 与 SAPIEN/Vulkan renderer 显式绑定到同一 NVIDIA PCI 设备。

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
GPU_ID=0 \
  bash experiments/robotwin/policy_content_adapter/run_release_c0_dev_gate.sh
```

该门槛只证明 C0 可按统一 evaluator 正确部署；6 个 outcome 明确标记为
`scientific_result=false`，不进入正式 SR 主表，也不会打开 `final_test` seed bank。
当前主线不再要求启动正式 C0；该 dev gate 仅保留为已有工程证据。

### v2 实施状态

方法和大部分组件可以复用：`full_550_per_task` loader、Stage 1 wrapper、原生三相机 paired 采集/export、state bank、Layer-16 cache extractor、C2 action stream、C3 contrastive stream、C0 transport、P 模式选择与 Clean/official Random 聚合器。

但当前 active formal configs/audits 仍强制绑定三任务 `B_CR` completion manifest。新规划正式运行前还必须实现：

- fail-closed `author_release` base-lineage manifest；
- release-base formal C0/C1/C3 configs；
- release ancestry 的 cache/training/C0/P-mode/evaluation audits；
- Stage 2 seed 与固定 C0 的新聚合语义；
- C1/C3 config difference allowlist 及完整测试。

因此，本页更新表示“实验规划已批准”，不表示旧正式命令已经可以直接用于新主线，也不表示 C0/C1/C3 正式结果已经产生。当前三任务 Stage 1 及其产物属于后移扩展实验。

---

# Legacy v1：三场景 Policy Content Adapter smoke

> **ARCHIVED / SUPERSEDED**：以下内容记录 2026-08-16 至 2026-08-17 已完成的旧 C/R1/R2 smoke、代码路径和产物。它仍是有效的工程历史证据，但不是 Policy Protocol v2 的完成证据。

本目录实现一个独立的、action-facing 的最小原型，用来验证 E0–E3 的背景稳定 content representation 能否转化为 RoboTwin Random rollout success 提升。当前实现不修改 `src/fastwam/**`，不覆盖 release checkpoint，也不会自动启动正式长训练。

## Legacy v1 结论与状态快照

状态含义：

- `PASS`：已有代码或可重复的静态/单元测试证据。
- `PENDING`：实现已准备，但仍需真实 GPU、数据或 checkpoint 运行证据。
- `BLOCKED`：缺少不能安全猜测的输入或审计转换，因此配置主动 fail closed。
- `NOT_STARTED`：尚未执行对应实验，不能报告结果。

截至 2026-08-17 的证据快照：

| 范围 | 状态 | 证据/限制 |
|---|---|---|
| 独立实现与静态测试 | PASS | `111 passed`；仅有一个沙箱中 CUDA 初始化 warning，不是 GPU smoke |
| 三任务 official task subset | PASS | hash-bound manifest 给出每任务 550 个候选 episode；与原生 seed-42 train split 交集后实际选择 546/543/549，共 1,638 个 episode；task-balanced smoke sampler |
| official 数据的 Clean/Random 身份 | PENDING | release metadata 未提供可信 domain label，当前只能称 `unverified_release_distribution` |
| P-v1 三步 GPU smoke | PASS | `gpu1_smoke_20260816_retry1/p_v1_smoke/strict_smoke_audit.json`；10 项 smoke gate 全部通过 |
| P-v2 三步 GPU smoke | PASS | `gpu1_smoke_20260816_retry1/p_v2_smoke/strict_smoke_audit.json`；含 ActionDiT FP32/BF16-visible update audit |
| 新 checkpoint 的真实模型 rollout load/execute | PASS | P-v1/P-v2 均加载 compact checkpoint；三个任务各执行 1 个 finite 14-D action，部署 `Zc=[1,8,384]` |
| Clean / official Random success rate | NOT_STARTED | 不能用离线 representation 指标代替 rollout success |
| 正式四组、3 seeds 长训练 | BLOCKED | smoke 已通过，但正式 checkpoint、domain map、text cache 等必需输入仍是占位符；本次没有启动长训练 |

静态测试命令：

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
PYTHONPATH=./src:. /root/anaconda3/envs/fastwam-robotwin-bw/bin/python \
  -m pytest -q experiments/robotwin/policy_content_adapter/tests

PYTHONPATH=./src:. /root/anaconda3/envs/fastwam-robotwin-bw/bin/python \
  -m experiments.robotwin.policy_content_adapter.config_audit \
  --config-dir experiments/robotwin/policy_content_adapter/configs
```

## 方法与原始 FastWAM action path

原生训练/rollout 的关键 action 路径保持不变：官方 observation 预处理与 normalization 生成视觉、文本、proprio 和 action 输入；Video branch 预填充 MoT KV cache；`ActionDiT.pre_dit` 中的 `action_encoder` 生成 action tokens；共享 MoT 通过 video KV cache 预测 action tokens；`ActionDiT.post_dit`、scheduler 和原生 action queue 生成并执行动作。

本原型只在 `model.action_expert.action_encoder` 的输出处安装一个显式、可撤销的 forward hook。这个位置是 native training 与 cached rollout 共用的最早 action-token 表示，因此可以让 action token 读取 content，同时不改变 release 模型 state-dict、Video branch 或原生 action queue 语义。未显式安装 runtime 时，FastWAM 行为完全不变。

Content Head 严格复用 E1–E3 的参数命名与 shape：

```text
Layer-16 [B, 120, 3072]
  -> Linear(3072, 384)
  -> 8 learnable queries
  -> one 8-head cross-attention
  -> Zc [B, 8, 384]
```

Policy 路径保留全部 8 个 query token，不做 mean pooling。只有 contrastive 路径执行 `mean -> Linear -> SiLU -> Linear -> L2 normalize`。E1/E2/E3 checkpoint 的 `payload["head"]` 使用 strict load。

唯一的 GCA 为：

```text
delta = CrossAttention(Xa, Zc, Zc)
Xa'   = Xa + tanh(g) * delta
g     = exactly 0 at initialization
```

初始化时 `tanh(0)=0`，tensor-level identity 单元测试为 bit-exact。真实 release 模型的 full action path audit 同时比较 mirrored/native video KV cache 与最终 action output；本次 P-v1/P-v2 GPU smoke 均验证 KV cache 与 action output bit-exact，`max_abs_error=0.0`、`max_rel_error=0.0`。

## P-v1、P-v2 与参数规模

| 模式 | VAE | Video Backbone | Content Head/GCA | ActionDiT | LR |
|---|---|---|---|---|---|
| P-v1 | Frozen | Frozen | Train | Frozen | Head/GCA `1e-4` |
| P-v2 | Frozen | Frozen | Train | Train | Head/GCA `1e-4`；ActionDiT `1e-5` |

精确参数数：

| 模块 | 参数数 |
|---|---:|
| Content Head | 2,070,144 |
| 单个 GCA（含 scalar gate） | 2,887,681 |
| P-v1 总 trainable | 4,957,825 |
| release ActionDiT | 1,020,900,366 |
| P-v2 总 trainable | 1,025,858,191 |

P-v1 中 ActionDiT 参数不更新，但不能把 action forward 包在 `no_grad` 中：`L_action` 必须穿过 frozen ActionDiT 回传到已经打开的 gate、GCA 和 Content Head。严格的 zero gate 有一个重要语义：第 1 步 action loss 只能给 gate 非零梯度；optimizer 打开 gate 后，从第 2 步起 action-only probe 才必须同时到达 GCA attention weights、官方 `Zc` 和 Content Head。paired contrastive loss 可以从第 1 步训练 Content Head。P-v2 另外要求 ActionDiT gradient/update 非零，并使用独立 `1e-5` param group。

## Legacy v1 真实 GPU 三任务 smoke 结果

成功运行目录：

```text
outputs/policy_content_adapter/gpu1_smoke_20260816_retry1/
```

两种模式均为单 seed、3 optimizer steps；随后加载各自真实 checkpoint，对 `place_a2b_left`、`open_microwave`、`move_stapler_pad` 各执行一次无 SAPIEN 的真实 FastWAM action inference。两份 `strict_smoke_audit.json` 均为 `PASS`。

| 指标 | P-v1 | P-v2 |
|---|---:|---:|
| zero-init action `max_abs_error` | 0.0 | 0.0 |
| zero-init action `max_rel_error` | 0.0 | 0.0 |
| final raw gate | `8.4461e-5` | `-5.1534e-5` |
| step-3 total loss | 0.071421 | 0.071266 |
| step-3 action loss | 0.002041 | 0.001886 |
| step-3 contrastive loss | 0.693799 | 0.693799 |
| step-3 action-only Head grad norm | `1.2813e-6` | `1.2004e-6` |
| step-3 action-only GCA grad norm | `9.9951e-7` | `8.9573e-7` |
| step-3 action-only official-Zc grad norm | `2.6215e-8` | `2.3088e-8` |
| step-3 ActionDiT grad norm | 0.0 | 0.375968 |
| Video Backbone / VAE grad norm | 0.0 / 0.0 | 0.0 / 0.0 |
| Head max parameter delta | `2.9991e-4` | `2.9991e-4` |
| Adapter max parameter delta | `8.4461e-5` | `7.2425e-5` |

P-v1 的 ActionDiT 明确记录为 `changed: false`。P-v2 对 ActionDiT 做确定性分层抽样：32 个参数张量、32,768 个元素中，32,529 个发生 FP32 更新（99.27%）；4,783 个在部署 BF16 量化后仍可见（14.60%），8/8 个必需的 early/mid/late/head strata 全部更新，Adam `exp_avg` 的 32,768 个样本全部非零且为 FP32。

两种模式的 rollout 均满足：

- checkpoint/base/runtime artifacts 与完整 47-file FastWAM Python source tree 校验通过；
- 三个任务各执行恰好 1 个 action；
- action shape 均为 `[14]` 且 finite；
- 部署 content tokens 均为 `[1,8,384]` 且 finite；
- `sapien_imported=false`；该检查证明加载和执行链路，不声称 simulator success rate。

`training_summary.json` 在 rollout 之前写入，因此其中的 `rollout_load_execute` 保留为 `PENDING_SEPARATE_SMOKE`；后续生成且重新复核通过的 `rollout_load_execute.json` 与 `strict_smoke_audit.json` 是该项的权威完成证据。

## Legacy v1 双流数据与 provenance

每个 optimizer step 使用两条独立 loader，不 concat dataset：

```text
official RoboTwin stream -> native action flow-matching MSE only
our paired C/R1/R2 stream -> multi-positive SupCon only
total = L_action + 0.1 * L_ctr, tau = 0.07
```

第一版没有 video reconstruction 或 video flow-matching loss；VAE、Video Backbone、text encoder 和 proprio encoder 都保持 frozen。R3 不进入 paired training，只在 evaluator 已支持时作为 background-only holdout。

Official 三任务 manifest 为 `configs/official_three_task_manifest.json`，其数据元信息按 size 和 SHA-256 绑定：

| task | official episode range | episodes |
|---|---:|---:|
| `place_a2b_left` | 11000–11549 | 550 |
| `open_microwave` | 9350–9899 | 550 |
| `move_stapler_pad` | 8250–8799 | 550 |

上述范围是 canonical candidate allowlist，不是绕过原生 split 的强制样本数。显式 loader 先复现原生 seed-42 train/val split，再取交集，实际选择如下：

| task | native train intersection | episodes |
|---|---:|---:|
| `place_a2b_left` | 11000–11549 与 train split 的交集 | 546 |
| `open_microwave` | 9350–9899 与 train split 的交集 | 543 |
| `move_stapler_pad` | 8250–8799 与 train split 的交集 | 549 |

一次真实 cold-cache loader 核验加载了 1,638 episodes / 461,851 frames，用时 58.76 秒；没有初始化整套 27,225 episodes。Smoke 使用 `episode_anchor` 与三任务 round-robin sampler，使最短 3 steps 覆盖三个任务。正式模板使用 `all_frames`。这个 manifest 只证明任务选择和 release 元数据身份，不证明数据是 Clean；在外部 domain evidence 提供前，代码会保留 `domain_verified: false`。

Paired stream 使用 `outputs/e2_e3/full/cache/e2_train.pt`，只接受严格顺序：

```text
clean, style_00_seed_0, style_01_seed_1
```

同 task、同 physical state 的 C/R1/R2 为 positives；同 task、不同 physical state 为 negatives。paired batch 如果包含 action target、R3、错误 variant 顺序或非 `[120,3072]` Layer-16 tokens，会直接报错。

本次 P-v1/P-v2 smoke 均已生成：

- `run_config.json`
- `artifact_identities.json`
- `official_subset_audit.json`
- `data_provenance_audit.json`
- `data_distribution_audit.json`
- `gradient_audit.json`
- `parameter_update_audit.json`
- `train_log.csv`
- `training_summary.json`
- `checkpoint.pt`
- `rollout_load_execute.json`
- `strict_smoke_audit.json`

其中分布 audit 比较 official stream 与 paired Clean 的 Layer-16 mean/std、pooled token norm 和 standardized mean gap，只作诊断，不会自动改变数据或把 release 数据重新标成 Clean。本次两种模式观察到同一组数据统计：official-unverified mean/std 为 `0.034613/1.096688`，paired-C 为 `0.001434/1.084306`；standardized mean gap 为 `0.030426`，std ratio 为 `1.011419`，token-L2 mean gap 为 `0.051780`。这说明本次采样的 Layer-16 一阶尺度偏移较小，但 release metadata 仍没有可信 Clean/Random 域标签，因此结果明确记录为 `DIAGNOSTIC_ONLY_UNVERIFIED_OFFICIAL_DOMAIN`，不能据此宣称完成了 official-Clean 科学比较。

## Legacy v1 Controls 与旧正式四组

> 下表是 2026-08-16 当时的 legacy snapshot，只描述旧 smoke schema 的状态；当前同名 active configs 已升级为 fail-closed `B_CR`/v2 schema，不能根据本表称其现在可运行。

| Control | 语义 | 配置状态 | 执行状态 |
|---|---|---|---|
| C0 Original | 原 policy matched continuation；无 Head/GCA/contrastive | 已建 `c0_original.yaml` | BLOCKED：需 smoke 后选定 P 模式并提供匹配的 native action-only runner |
| C1 Architecture-only | 与 ours 同 Head/GCA，`lambda_ctr=0`，official action-only | 当时已建 `c1_architecture_only.yaml` | PENDING：当时旧 smoke schema 可运行，尚未做 GPU control |
| C2 Naive Aug | C/R1/R2 直接 action supervision，无 contrastive | 已建 `c2_naive_aug.yaml` | BLOCKED：30 Hz paired current-frame 数据缺少经审计的 official 50 Hz、32-step action-window converter |
| C3 Ours | official -> action；paired C/R1/R2 -> contrastive | 当时已建 `c3_ours.yaml` | PENDING：当时旧 smoke schema 可运行，尚未做 GPU control |

当前 C1/C3 短控制模板共同锁定为 P-v1，便于在 smoke 前提供一对可审计配置。若 P-v2 smoke 更合适，必须把 C1 与 C3 的 `policy.regime`、ActionDiT freeze 及所有匹配训练字段同时切换，并重新通过内置 C1↔C3 fairness audit；不能只改其中一组。

四个正式模板已建立并通过结构与 matched-pair 公平性检查，因此“config 交付”本身为 PASS：

- `formal_clean.yaml`
- `formal_clean_aug.yaml`
- `formal_clean_random.yaml`
- `formal_clean_random_aug.yaml`

它们当前全部是 `runnable: false`、`fail_closed: true`。必须人工提供且校验：Clean-only checkpoint/SHA、Clean+Random checkpoint/SHA、各自 dataset root/stats、可信 domain evidence、task manifest、预计算 text cache、matched continuation steps/batch size/rollout protocol、P-v1 或 P-v2 选择及 matched adapter initialization seed。代码不会猜这些值。

正式比较固定为：

```text
Clean vs Clean+Aug                     # 同 checkpoint、steps、rollout protocol
Clean+Random vs Clean+Random+Aug       # 同 checkpoint、steps、rollout protocol
```

## Legacy v1 三任务 smoke 命令

推荐使用统一脚本。它依次运行 config/fairness audit、111 项单元测试、P-v1 三步训练及三任务一动作 rollout，然后运行 P-v2；任一严格检查失败都会停止，并且绝不会启动 controls 或正式长训练：

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
GPU_ID=0 \
MODEL_BASE=/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints \
bash experiments/robotwin/policy_content_adapter/run_three_task_smoke.sh
```

以下命令都是 1 seed、3 optimizer steps；它们只运行指定 config，不会级联启动 controls 或正式四组。建议在空闲 GPU 上逐条执行并在每条结束后审计输出。

P-v1：

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
CUDA_VISIBLE_DEVICES=0 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=./src:. \
/root/anaconda3/envs/fastwam-robotwin-bw/bin/python \
  -m experiments.robotwin.policy_content_adapter.train \
  --config experiments/robotwin/policy_content_adapter/configs/p_v1_smoke.yaml
```

P-v2（单独运行，不由 P-v1 自动启动）：

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
CUDA_VISIBLE_DEVICES=0 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=./src:. \
/root/anaconda3/envs/fastwam-robotwin-bw/bin/python \
  -m experiments.robotwin.policy_content_adapter.train \
  --config experiments/robotwin/policy_content_adapter/configs/p_v2_smoke.yaml
```

真实模型、无 SAPIEN 的一动作 checkpoint load/execute smoke（checkpoint 生成后）：

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
CUDA_VISIBLE_DEVICES=0 \
DIFFSYNTH_SKIP_DOWNLOAD=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=./src:. \
/root/anaconda3/envs/fastwam-robotwin-bw/bin/python \
  -m experiments.robotwin.policy_content_adapter.rollout_policy \
  --checkpoint outputs/policy_content_adapter/p_v1_smoke/checkpoint.pt \
  --dataset-stats /mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --model-base-path /mnt/cpfs-E/baoshifeng/FastWAM/checkpoints \
  --device cuda --mixed-precision bf16 \
  --action-horizon 32 --replan-steps 1 --num-inference-steps 1 --seed 0 \
  --output-json outputs/policy_content_adapter/p_v1_smoke/rollout_load_execute.json
```

该 smoke 对三个任务各执行一次真实 FastWAM action inference，但不创建 renderer。它会先验证 compact checkpoint schema v3、base checkpoint SHA-256、checkpoint 标为 rollout-required 的 dataset stats/VAE/text encoder/tokenizer identities，以及训练时绑定的整个 `src/fastwam/**/*.py` 源码树（当前 47 个文件）的绝对 source root、文件集合、size 和 SHA-256；缺字段、增删文件或任一源码字节漂移都会 fail closed。

RoboTwin Clean + official Random 小规模 rollout（需要 SAPIEN/Vulkan 正常）：

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
PYTHONPATH=./src:. /root/anaconda3/envs/fastwam-robotwin-bw/bin/python \
  experiments/robotwin/policy_content_adapter/eval_robotwin_single.py \
  ckpt=outputs/policy_content_adapter/p_v1_smoke/checkpoint.pt \
  gpu_id=0 \
  EVALUATION.dataset_stats_path=/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name='[place_a2b_left,open_microwave,move_stapler_pad]' \
  EVALUATION.task_config=both \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.output_dir=outputs/policy_content_adapter/p_v1_smoke/robotwin_rollout
```

在该 legacy v1 方案中，原计划正式结果报告 Clean SR、official Random SR 和 matched delta；R3 仅在旧 evaluator 支持时报告。此句不适用于当前 v2：当前 Policy 主表不报告 R3，Style/State/Ratio/R@1 仍只属于 representation 解释。

## 17 项交付/审计清单

| # | 交付项 | 状态 | 当前证据与下一步 |
|---:|---|---|---|
| 1 | 代码修改文件列表 | PASS | 见下节；新增均位于独立实验目录，无 `src/fastwam/**` 修改；最终 `git status`/`git diff --stat` 需与用户已有改动一起如实报告 |
| 2 | FastWAM 原 action path 理解 | PASS | 已在“方法与原始 FastWAM action path”说明 native prefill、ActionDiT、MoT、post-DiT/action queue |
| 3 | Content Head 复用方式 | PASS | E1–E3 参数命名/shape strict compatible；policy 保留 `[B,8,384]`，contrastive 才 mean pool |
| 4 | GCA 插入位置与原因 | PASS | 单 hook 位于 `ActionDiT.action_encoder` 输出，是 train/cached rollout 共用 action-token 点 |
| 5 | zero-init identity test 数值 | PASS | P-v1/P-v2 真实 release full action path 均 bit-exact，`max_abs_error=max_rel_error=0.0` |
| 6 | P-v1 gradient audit | PASS | step 3 action-only Head/GCA/Zc grad 均非零；ActionDiT、Video、VAE grad 均为 0；Head/GCA 更新且 gate 打开 |
| 7 | P-v2 gradient audit | PASS | 独立 LR group；ActionDiT grad=0.375968；99.27% sampled FP32 elements 更新，8/8 strata 与四类 BF16 visibility 均通过 |
| 8 | trainable parameter counts | PASS | Head 2,070,144；GCA 2,887,681；P-v1 4,957,825；P-v2 1,025,858,191 |
| 9 | dual-stream data provenance | PASS | official action-only、paired contrastive-only、R3 excluded；manifest/cache identities 受审计 |
| 10 | official Clean vs our C audit | PASS（诊断） | 真实 Layer-16 moments/shift 已输出；审计结论是 release 域无法证明为 Clean，故保持 unverified，禁止伪造 Clean claim |
| 11 | smoke loss/gate/gradient 日志 | PASS | 两种模式均有 3 行 `train_log.csv`、`gradient_audit.json`、`parameter_update_audit.json` 与真实数值 |
| 12 | 新 checkpoint 被 rollout evaluator 加载执行 | PASS | 两个 checkpoint 均严格加载；三个任务各执行 1 个 finite 14-D action，`Zc=[1,8,384]`，无 SAPIEN |
| 13 | P-v1 完整运行命令 | PASS | 见“三任务 smoke 命令” |
| 14 | P-v2 完整运行命令 | PASS | 见“三任务 smoke 命令” |
| 15 | Architecture-only / Naive Aug / Ours config | PASS（legacy snapshot） | 当时旧 smoke schema 下 C1/C3 可运行；这不描述当前同名 active configs；旧 C2 因 action converter 缺失而 BLOCKED |
| 16 | 四组正式实验 config | PASS（legacy 模板） | 当时四模板与 matched-pair audit 已就绪；当前 v2/release-base readiness 由本文顶部的新协议决定 |
| 17 | `README_POLICY_CONTENT_ADAPTER.md` | PASS | 本文件已写入 2026-08-17 GPU smoke 的真实 artifact 数值与限制 |

## 代码文件列表

核心实现：

- `model.py`：E1–E3-compatible head、single zero-init GCA、runtime hook、freeze/optimizer groups、compact checkpoint v3。
- `losses.py`：native action-only flow matching、paired-only SupCon、full-path zero-gate identity audit、feature moments。
- `train.py`：strict P-v1/P-v2 dual-stream trainer、三任务 coverage、日志与 checkpoint。
- `data.py`：paired frozen-token dataset、same-task physical-state batching、dual-stream provenance。
- `official_data.py`：hash-bound official 三任务 subset 与 round-robin sampler。
- `training_audit.py`：gradient、parameter-update 与 distribution audits。
- `runtime_utils.py`：release model、official dataset 与 dtype/runtime resolution。
- `rollout_policy.py`：checkpoint-v3-aware native RoboTwin policy bridge和无 SAPIEN 一动作 smoke。
- `eval_robotwin_single.py`：Clean/Random 顺序评估入口与 success-rate manifest。
- `deploy_policy.yml`：独立 RoboTwin policy module 配置。
- `config_audit.py`：config schema、execution readiness 与正式 matched-pair 审计。
- `smoke_audit.py`：从实际产物逐条验证 10 个 smoke 目标、任务覆盖、梯度/更新、identity、源码绑定与 rollout 执行。
- `run_three_task_smoke.sh`：顺序运行静态审计、P-v1/P-v2 三步训练、三任务无 SAPIEN rollout 和严格产物审计，任一步失败即停止。

配置：

- `configs/p_v1_smoke.yaml`、`configs/p_v2_smoke.yaml`
- `configs/c0_original.yaml`、`configs/c1_architecture_only.yaml`
- `configs/c2_naive_aug.yaml`、`configs/c3_ours.yaml`
- `configs/formal_clean.yaml`、`configs/formal_clean_aug.yaml`
- `configs/formal_clean_random.yaml`、`configs/formal_clean_random_aug.yaml`
- `configs/official_three_task_manifest.json`

测试：

- `tests/test_model.py`
- `tests/test_data.py`
- `tests/test_official_data.py`
- `tests/test_configs.py`
- `tests/test_training_audit.py`
- `tests/test_checkpoint_v3.py`
- `tests/test_rollout.py`
- `tests/test_runtime_explicit_loader.py`
- `tests/test_smoke_audit.py`

包文件：

- `__init__.py`

## Legacy v1 停止条件与非目标

`train.py` 只执行显式 config 的 `max_steps`，并把 `formal_training_auto_started` 记录为 `false`。Smoke 结束后必须停下审阅 identity、gradient、update、distribution 与 rollout load/execute 产物；它不会自动运行 C0–C3、正式四组或 3 seeds。

当前不实现 LOVO、router、SIVA full architecture、multi-layer/every-4-block adapter、query sweep、full Video Backbone fine-tuning、teacher、EMA、memory bank 或新 video loss。正式训练的唯一解锁条件是：P-v1/P-v2 smoke 和 rollout 链路通过、选择训练模式、补齐所有可信 provenance/路径、正式 config audit 通过，并由人工显式启动。

最终科学判断仍只有一个：与严格 matched baseline 相比，方法是否提高 official Random success rate，同时不显著损害 Clean success rate。
