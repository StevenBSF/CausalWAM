# Policy Method v2：Action-facing Causal Content Adapter

> 状态：本文件解释 Policy v2 的方法；唯一权威协议是 [`policy_protocol_v2.md`](policy_protocol_v2.md)。2026-08-19 起，当前主线改用作者 50-task RoboTwin release checkpoint 作为固定基座，方法结构和 loss 不变；自行训练三任务基座及其后续复验后移。旧 30 Hz、C/R1/R2 训练和 R3 holdout 方案仅属于 E0–E3 representation 实验及历史 smoke。

## 1. 研究问题与协议边界

E0–E3 已完成 representation-level 验证，用 Style distance、State distance、State/Style ratio 和 R3→Clean R@1 检查中间表示。Policy 阶段只回答：背景稳定的 content representation 能否提升 RoboTwin 在线执行成功率。

两类实验必须分开：

- E0–E3：保留原代码、30 Hz 数据、cache、checkpoint 和历史结果，不覆盖、不改写；
- Policy v2：使用已生成并审计的原生官方格式 paired 数据，并只用 `demo_clean` 和 `demo_randomized` 做在线 rollout 评测；
- R3 在 Policy v2 中是第三种随机场景训练版本，不是单独的 Policy 测试环境。

## 2. 当前共同基座：作者全任务预训练 FastWAM

当前主线固定使用作者发布的 RoboTwin 50-task checkpoint `B_release`：

```text
checkpoint: /mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
sha256:    776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63
stats:     /mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
sha256:    7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095
```

release 数据包含 27,500 episodes，对应 50 个任务块 × 550；本协议将每个任务块划分为前 50 Clean、后 500 Official Random。由于 release 元数据本身没有 domain 字段，论文中应把它描述为“作者全任务 release checkpoint + 本协议审计的 50/500 分区”，而不是声称 checkpoint payload 自己证明了域标签。

三个目标任务已参与作者预训练。因此研究问题是：

> 在固定作者全任务预训练策略上，加入 action-facing content adapter 和 paired contrastive supervision，是否提高三个已见任务的 official Random 在线鲁棒性，同时保持 Clean success rate？

这不是 zero-shot、未见任务泛化或从头训练结论。原三任务 `B_CR` 训练定义完全保留，但移动到第 11 节的后续基座敏感性实验。

## 3. Policy v2 原生 Paired 数据

旧 30 Hz C/R1/R2/R3 数据不做 30→50 Hz 插值，也不用于正式 Policy C2/C3。Policy v2 已在 `FastWAM/outputs/policy_content_adapter/native50hz_three_task_rgb640x480_v1/full_lerobot_v21` 生成原生数据；export audit 已 PASS：600 scene episodes、50 contents/task、50 Hz、32-step × 14D、`all_pairs_exact=true`、无插值。release-base 正式运行前仍需把该 artifact 与新的 lineage/state-bank/split manifest 重新 SHA-lock。

### 3.1 一个训练时刻包含什么

每个时刻必须同时包含：

- 第一视角相机；
- 左腕相机；
- 右腕相机；
- 当前机器人状态；
- 原生 50 Hz 状态与动作序列；
- 从当前时刻开始的 32 步、14 维动作目标。

即：

\[
O_t=\{\text{first-person camera},\text{left wrist camera},\text{right wrist camera},\text{robot state}\},
\]

\[
Y_t=[a_t,a_{t+1},\ldots,a_{t+31}],\qquad a_t\in\mathbb{R}^{14}.
\]

### 3.2 四种场景版本不是四个相机

同一物理状态生成四种场景版本：

- C：Clean；
- R1：随机场景版本 1；
- R2：随机场景版本 2；
- R3：随机场景版本 3。

每种场景版本内部仍有上述三个同步相机。因此一个物理状态对应的是：

```text
4 种场景版本 × 每种场景 3 个相机
```

不能把 C/R1/R2/R3 称为四个 camera views；文档和代码统一使用“scene variants / 场景版本”。

### 3.3 生成方式与共享动作标签

专家策略只执行一次。采集系统记录原生 50 Hz 物理状态、机器人状态和完整动作轨迹，再从相同物理状态重新渲染 C/R1/R2/R3。四种场景版本必须共享完全相同的 32-step 动作标签：

\[
(C_t,R1_t,R2_t,R3_t)\rightarrow[a_t,\ldots,a_{t+31}].
\]

允许优先改变：背景、纹理、材质、光照和不发生碰撞的装饰物。暂不改变目标物体位置、机器人状态、桌高、相机内外参或参与碰撞的物体，否则原动作标签可能失效。

不得让专家策略分别执行四次后把四条近似轨迹伪装成 paired data，也不得通过 30→50 Hz 插值构造正式标签。

### 3.4 数量与拆分

| Split | Clean/task | Random/task | Physical trajectories/task |
|---|---:|---:|---:|
| Train | 30 | 90 | 30 |
| Val | 10 | 30 | 10 |
| Test | 10 | 30 | 10 |
| 总计 | 50 | 150 | 50 |

每条 physical trajectory 产生 1 个 Clean 和 3 个 Random 场景版本。每条轨迹会进一步产生大量 50 Hz、32-step action windows；表中的 30 不是 30 张图片。

第一轮为每条 Train trajectory 确定性选取 8 个无需末端 padding 的状态，因此每任务是 240 个 physical-state groups，三个任务合计 720 groups、2,880 views。C2/C3 使用同一个带 SHA 的 state bank；如果数据不够，优先新增轨迹和 style seeds，而不是重复抽取同一轨迹的更多相邻帧。

### 3.5 Pilot gate 与当前状态

先为每个任务生成 `1 条物理轨迹 × C/R1/R2/R3`。三个任务必须全部通过以下审计，才能扩大到 30/10/10：

- 三相机同步且相机身份正确；
- 控制、状态和动作记录为原生 50 Hz；
- 动作为 14 维，维度顺序、单位和归一化与 official stream 一致；
- 每个有效时刻可构造未来 32 步窗口，末端 padding/mask 规则明确；
- C/R1/R2/R3 的物理状态、机器人状态和动作逐项一致；
- 图像、状态、动作与时间戳对齐；
- 场景随机化没有改变动作有效性；
- split、task、trajectory、scene variant 和 source state 均可追溯。

上述 pilot→full gate 已执行，完整 artifact 的 native action/export contract 为 PASS。它不等于 release-base cache、Stage 2 或在线评测已经完成。

## 4. Stage 2：C0–C3 实验矩阵

所有当前主线分支都从同一个 `B_release` 出发：

| 组别 | Head/GCA | Official C+R | 新 Paired C/R1/R2/R3 | 作用 |
|---|---|---|---|---|
| C0 Author Release | 无 | 不继续训练 | 无 | 可选外部参考，不阻塞主实验 |
| C1 Architecture-only（action-only continuation） | 有 | Action loss | `lambda_ctr=0` | 主 baseline；控制架构与 official continuation |
| C3 Ours | 与 C1 完全相同 | Action loss | `lambda_ctr>0` Contrastive loss | 主方法 |
| C2 Naive Aug | 与 C3 相同 | Action loss | Action loss | 后续辅助对照 |

对应目标：

\[
L(\lambda)=L_{\text{action}}(D_{\text{official}})
+\lambda L_{\text{contrastive}}(D_{\text{paired}}),
\]

C1 令 `lambda=0`，C3 令 `lambda=lambda_ctr>0`。C3 的 paired stream 只提供 contrastive supervision，不提供 action loss。两者必须具有完全相同的 `B_release`、Head/GCA 初始 tensor、official stream、优化预算和公共随机流；主结论只来自 `C3-C1`。

C2 的目标保持为：

\[
L_{\text{C2}}=L_{\text{action}}(D_{\text{official}})+L_{\text{action}}(D_{\text{paired}}).
\]

C2 使用同一套新数据的 32-step action target，不使用 contrastive loss；它后移为辅助 ablation，不阻塞 C1/C3 主线。

C2 的 action-supervised 数据路径和 paired 数据 gate 已具备。它当前不再被数据生成阻塞，而是后移为辅助实验，并等待 release-base lineage、formal config 与审计适配；旧 30 Hz 数据仍不进入 C2。

C0 本身不做 Stage 2 训练，也不是当前主实验的必需项。若把已有原始 FastWAM 结果作为可选 reference 纳入统一 evaluator，只允许完整提供三任务 × Clean/Random 六格，且必须与 C1/C3 使用相同 final seed bank 和 rollout protocol；部分或异协议 C0 数字会被严格拒绝。

三个关键比较：

- **C3−C1：paired contrastive supervision 的增量作用，唯一主结论；**
- C3−C0：仅在存在同协议完整 C0 时报告，且同时混入架构、action continuation 和 contrastive，只作辅助；
- C2−C1、C3−C2：普通 action augmentation 的后续辅助比较。

## 5. Content Head、GCA 与初始化

Content Head 保持 E1–E3 验证过的结构：

```text
Layer-16 [B,120,3072]
  -> Linear(3072,384)
  -> 8 learnable queries
  -> one-layer 8-head cross-attention
  -> Zc [B,8,384]
```

Policy 路径保留 8 个 query tokens；只有 contrastive 路径执行 `mean pool -> projection -> L2 normalize`。

ActionDiT 的 action tokens 通过单个 zero-init gated cross-attention 读取 `Zc`：

\[
X_a'=X_a+\tanh(g)\operatorname{CrossAttn}(X_a,Z_c,Z_c),\qquad g=0.
\]

正式主比较中：

- C1/C2/C3 使用完全相同、同随机种子的 Head/GCA 初始参数；
- 默认使用随机初始化，使 C1 真正是 Architecture-only；
- E0–E3 只锁定 layer、结构和超参数，不直接向主比较转移 contrastive-trained Head；
- E2 pretrained Head 只作为额外 warm-start ablation。

若使用 pretrained Head，必须另命名为 `C1-pretrained` 和 `C3-pretrained`，不能把 `C1-pretrained` 称为纯 Architecture-only。

## 6. Paired contrastive 定义与 cache

同一物理状态的 positive group 为：

\[
G_s=\{C_s,R1_s,R2_s,R3_s\}.
\]

每个 anchor 的另外三个场景版本都是 positives；同一任务、不同物理状态是 negatives。

正式 Policy cache 必须由锁定 SHA 的 `B_release` 对新 native-50Hz paired 数据重新提取。不得复用或拼接旧 `e0/e1/e2` cache，也不得使用未完成三任务 `B_CR` 的 cache。新 cache 使用独立协议名，例如：

```text
policy_release50tasks_native50hz_four_scene_v1
```

cache manifest 必须绑定 `B_release` checkpoint/stats SHA、数据 manifest/state-bank SHA、模型代码与预处理、camera/scene 顺序、layer、dtype、token shape 和 split，并验证同一真实输入下 cache extractor 与在线 native prefill 的 Layer-16 tokens bit-exact。旧 E0–E3 cache、checkpoint 和结果不覆盖、不删除。当前 P-v1/P-v2 都冻结 Video DiT，因此一份通过审计的 immutable cache 可供 C1/C3 和三个 Stage 2 seeds 共享；若未来解冻 Video DiT，则必须重新定义在线特征或按 checkpoint 重提 cache。

Train cache 的主协议形状是 `720 physical-state groups × 4 scene variants`；每个 view 的 Layer-16 token shape 为 `[120,3072]`。

## 7. P-v1/P-v2

| 模式 | Video Backbone | Head/GCA | ActionDiT | 特点 |
|---|---|---|---|---|
| P-v1 | Frozen | Train | Frozen | 显存低、稳定，但适应能力可能有限 |
| P-v2 | Frozen | Train | 小学习率更新 | 适应能力更强，但显存和破坏原能力的风险更高 |

历史 smoke 只证明两种模式的训练、梯度、保存和加载链路能够工作，没有产生 RoboTwin 在线成功率，不能据此选正式模式。

选择流程：

1. 先分别运行 P-v1/P-v2 工程 smoke，确认两条实现路径可训练和部署；
2. 从同一个固定 `B_release`、同一 Head/GCA 初始化和同一公共训练配方，使用 C1/action-only 目标（`lambda=0`）各短训一个 dev-pilot checkpoint；
3. 在同一个 `dev_selection` seed bank 上，对三个任务的 Clean/official Random 各运行 20 episodes/task/domain；
4. Clean macro SR 相比两者最佳值下降超过 0.05 的候选不合格；合格候选中选 official Random macro SR 更高者，完全打平时固定选 P-v1；
5. 用不可覆盖且 SHA 绑定的 selection manifest 一次性锁定胜者，C1/C2/C3 都必须引用它；
6. `final_test` seed bank 必须显式证明与 dev bank 不相交，且不得参与选择。

P 模式不得用 C3 contrastive dev 结果或 `C3-C1` dev gap 选择；否则完整 pipeline 会偏向方法组，不能再笼统声称只差 contrastive supervision。

## 8. 公平性约束

C1/C3 必须锁定：

- 相同 `B_release` checkpoint、dataset stats、模型配置、源码、预处理和 SHA；
- 每个 Stage 2 seed 内相同 Head/GCA 初始 tensor 及 tensor SHA；
- 相同 P-v1/P-v2 模式与 freeze contract；
- 相同 official 数据、batch 顺序、action noise、diffusion timesteps 和 augmentation；
- 相同 optimizer、参数组 LR、scheduler、weight decay、steps、precision、gradient accumulation/clipping 和 effective global batch；
- 相同公共随机流与 rollout simulator seeds、instruction 和 inference 配置。

正式 seeds `[1,2,3]` 现在只表示 Stage 2 adapter/优化随机性。每个 seed 内运行配对的 `C1_i/C3_i`；不能把它们写成三个独立预训练基座。C0 是单个固定 checkpoint，没有 training-seed 维度。

`lambda_ctr`、temperature、paired/official batch ratio、max steps、参数组 LR 和 config SHA 必须在任何 P-mode dev/final rollout 前锁定；之后的消融不能回写主实验配置。

C1/C3 的正式 config diff 必须通过 allowlist 审计：除 `lambda_ctr` 和由此必需的 contrastive gradient 开关外，不允许出现影响训练语义的差异。两者必须读取相同 official/paired 顺序、执行相同公共 forward/随机流，并令 C1 paired loss 为零，避免额外数据路径改变 official stream。

C2 仍与 C3 共享 paired 数据、Train/Val split 和 physical-state 顺序，但作为后续辅助对照运行。

## 9. Policy 在线评测与报告

Policy 最终只使用 RoboTwin 官方在线环境：

- `demo_clean`；
- `demo_randomized`。

所有方法使用完全相同的 simulator seeds、episodes/task、instruction 类型、action horizon、inference steps、replan 参数和 success definition。正式实验建议每 task、每环境 100 episodes，并报告：

- 三任务分别的 Clean SR；
- 三任务分别的 official Random SR；
- 三任务 macro average；
- C1/C3 三个 Stage 2 seeds `[1,2,3]` 的 mean ± std；
- 主比较 C3−C1；同协议完整 C0 若存在才辅助报告 C3−C0；C2 完成后另报 C2−C1/C3−C2，条件允许时附配对置信区间。

正式聚合 schema v4/profile `c1_c3_primary` 强制要求 C1/C3 的 36 条 records（2 controls × 3 seeds × 3 tasks × 2 domains）。C0 可不提供；若提供必须为相同 final-test seed bank 下的完整六格，且不得复制成三个 training seeds。`C3-C1` 是主因果对照，`C3-C0` 只能解释为总体适配差异。

Policy 主表不报告 R3 success、Style/State/Ratio 或 R3→Clean R@1；这些属于 E0–E3 representation 结果。`demo_randomized` 还包含杂物、光照、桌高等变化，因此结论只能表述为“提升 official Random 环境下的 Policy 鲁棒性”，不能只归因于背景。

## 10. 当前实施状态

截至 2026-08-19：

- E0–E3 与旧 P-v1/P-v2 smoke 作为 legacy 证据保留；
- 方法结构、paired 数据链路、`full_550_per_task` loader、Head/GCA、C1/C2/C3 loss、cache/P-mode/在线评测核心已有实现；
- 当前 active formal configs 和 fail-closed audits 仍绑定三任务 `B_CR` completion，尚未适配 `author_release` lineage；
- 因此本次只完成实验规划修订，不代表 release-base C0/C1/C3 已可用旧命令直接正式运行；
- 在 lineage、cache、训练和聚合审计适配并通过测试前，不启动新的正式 C1/C3，也不把旧 smoke 解释为 online success；
- 已启动或部分完成的三任务 Stage 1 不进入当前主结果。

## 11. 后续扩展：三任务 `B_CR` 基座

原方法不删除。资源允许后，仍按每任务 50 Clean + 500 Official Random、原生 `L_video+L_action`、5 epochs 训练三个 `B_CR^(1/2/3)`，再在每个基座内复刻 C1/C3。该实验用于检验 release-base 主结论是否依赖基座训练制度；它同时改变训练语料和任务规模，不是纯初始化消融。

这条扩展线必须重新为每个 `B_CR` 提取匹配 cache，并与当前 `B_release` 主表分开报告。Stage 1 已记录的执行修订保持为 8 GPUs、micro-batch 8/GPU、gradient accumulation 2、effective global batch 128；当前未完成 run 只保留为工程 provenance，不作为方法结果。

## 12. P-v2 可训练 ActionDiT 的机制解释边界

P-v2 不改变 Content Head/GCA 结构或 contrastive loss；它只把 ActionDiT 从冻结状态改为以 `1e-5` 学习率训练，Head/GCA 仍为 `1e-4`。因此 P-v2 的 `C3-C1` 回答的是：当 action backbone 也可适配时，contrastive supervision 的梯度能否通过 Head/GCA 与 action 路径转化为在线 SR 增益。

该实验是 P-v1 主结果之后的机制研究。即使通过，也只能作为“可训练 ActionDiT 条件下”的补充证据，不能回写 P-v1 的预注册主结论。seed53 pilot 的在线评测已在完整结果产生前按用户要求通过审计 amendment 从 20 提升为每任务/每域 100 episodes，以统一原始 FastWAM 的评测规模；训练与 gate 阈值不变。其锁定协议与条件式 seed59 测试见 `FastWAM/docs/pv2_actiondit_followup.md`。

seed53 pilot 的实际 macro 结果为：C1 Clean/Random `44.67%/59.33%`，C3 `46.67%/64.00%`，即 `C3-C1 = +2.00/+4.67 pp`。因此预设 Random gain 与 Clean guard 均通过，机制实验进入 seeds 2/3 + seed59 的三 seed 确认阶段；pilot 数值本身不作为最终三 seed 结论。
