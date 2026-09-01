# Motus Policy Content Adapter 适配规划

状态：CPU/data/model/train/checkpoint/evaluation implementation 已完成，CPU
suite 39 tests PASS；作者 Stage-3 checkpoint 已下载并完成 SHA lineage，GPU
artifact gate 正在执行 strict-load/zero-gate/cache。
当前真实数据 manifest、执行顺序与命令见
`experiments/robotwin/policy_content_adapter/README.md`。

## 1. 目标与结论边界

目标是在作者发布的 Motus RoboTwin policy 上复现我们在 FastWAM 中定义的核心方法：

1. 从当前观测提取与动作无关的视觉内容表示 `Zc`；
2. 通过零初始化门控交叉注意力（GCA）把 `Zc` 注入 Action Expert；
3. 用同一物理状态的 `C/R1/R2/R3` 作为正样本，学习背景不变的 Content Head；
4. 在 RoboTwin 三个任务上只用 Clean/Official Random 在线成功率评价 policy。

主因果对照必须是 `M3 - M1`：两组基座、初始化、数据顺序、动作噪声、timestep、训练步数与评测设置完全一致，唯一差异是 M3 的 paired contrastive gradient 打开。

这个实验能够回答：在作者 Motus RoboTwin policy 上，paired content-invariance supervision 是否提升三个已见任务的官方随机化鲁棒性。它不能证明未见任务泛化，也不能把 Official Random 的变化单独归因于背景。

## 2. 已审计的 Motus 基础

### 2.1 作者模型与训练协议

- 上游仓库版本：当前 Motus 仓库 HEAD `f771216`，工作树在开始适配前是 clean。
- 模型约 8B 参数：Wan video expert 约 5B、Qwen3-VL 约 2.1B、Action Expert 约 0.64B、Understanding Expert 约 0.25B。
- RoboTwin 原生训练输入是三相机 T-shaped composite，模型分辨率 `384x320`。
- 原生 action/state 都是 14 维，action chunk 为 16。
- 每个训练样本使用当前帧、8 个未来视频帧、状态、action chunk、T5 embedding 和 Qwen3-VL 输入。
- 原生目标为 `L_video + L_action`；VLM 默认冻结，WAN、Action Expert、Understanding Expert 参与 Stage-3 fine-tuning。
- 作者声明的 RoboTwin setting 为 50 tasks，每任务 50 Clean + 500 Random，平均成功率 87.02%。

### 2.2 当前本地资产

已存在：

- Stage-2 `Motus` checkpoint；
- Qwen3-VL 权重；
- Wan2.2 权重；
- FastWAM 已生成的三任务 native-50Hz paired `C/R1/R2/R3` 观测数据与 state bank。
- 三任务 1,650-episode official manifest，以及 720-state/2,880-view paired manifest；
- Wan/Qwen 配置和 tokenizer 元数据。

正式适配当前缺少：

- 基于完整 checkpoint 的 strict-load GPU audit；
- 三任务 UMT5 cache、frozen WAN token cache 和 zero-gate GPU audit。

Stage-3 checkpoint SHA256 已锁定为
`70bc1741b45db5e3eae86510ad3bb1bd1aef04f10b17d8c1ff4dbe099300eb65`。

上述 GPU artifacts 补齐前，CPU/data 实现可以验证，但不能称为正式 Motus policy 实验。

## 3. 为什么不能直接复制 FastWAM hook

FastWAM 有清晰的“当前帧 video prefill -> Layer-16 tokens -> ActionDiT”路径。Motus 不同：它在每个 denoising step 中让 video、action、understanding 三种 token 共同经过 30 层 joint attention。

如果直接从 Motus 正常 policy forward 的第 16 层抓 video token，它会同时依赖：

- 未来视频噪声；
- 当前 action noise；
- Understanding Expert token；
- 当前 diffusion timestep。

这样的 token 不是稳定的“当前观测内容表示”，也无法在十次 denoising step 间安全复用，因此不符合我们原方法的语义。

Motus 已提供 `WanVideoModel.get_layer_features(...)`。建议用它建立一个独立、确定性的 observation-only content branch：

```text
三相机当前观测
    -> Motus 原生 T-shaped composite / resize
    -> VAE encode current frame
    -> frozen WAN video-only forward (t=0, instruction T5 context)
    -> selected layer tokens
    -> Content Head
    -> Zc [B, 8, 384]

state + noisy action chunk
    -> ActionExpert.input_encoder
    -> Xa [B, 17(+registers), 1024]
    -> Xa + tanh(gate) * MHA(Xa, Zc, Zc)
    -> Motus 原生 30-layer trimodal joint path
    -> action velocity
```

`Zc` 每个当前观测只计算一次，在同一次 policy inference 的全部 denoising steps 中复用。

## 4. 新增模块

### 4.1 MotusContentHead

保持 FastWAM 方法定义：

- 输入：候选 WAN 中间层视觉 tokens，初始候选 layer 16；
- 输入维度：预计 3072，必须用真实 checkpoint smoke 审计；
- 8 个 learned queries；
- content dimension 384；
- policy path 输出 `Zc: [B,8,384]`；
- contrastive path 对 8 queries 聚合，经小 MLP 和 L2 normalization 得到 representation。

不能复用 FastWAM 的 Content Head 权重，因为 backbone、预处理和 token 分布均不同；只能复用结构与超参数定义。

### 4.2 MotusGatedCrossAttentionAdapter

- query：Action Expert input tokens，dimension 1024；
- key/value：`Zc`，dimension 384；
- 8 attention heads；
- residual injection；
- scalar gate 必须精确初始化为 0。

hook 位置固定在 `action_expert.input_encoder(...)` 之后、Action Expert 第 0 层之前。这样能覆盖 state/action tokens，又不修改作者 30 层 trimodal block 的定义。

### 4.3 零门控部署契约

当 `gate == 0` 时，加入 Head/GCA 的模型 action output 必须与原 Motus 在相同输入、相同 RNG、相同 precision 下相等。正式 gate 要至少验证：

- 3 tasks；
- 多个 action diffusion timesteps；
- inference 10 steps；
- 输出 shape、finite、最大绝对误差；
- checkpoint save/load 后仍成立。

如果原实现存在不可避免的非确定性，必须先定义数值容差，不能事后依据结果放宽。

## 5. 冻结与训练模式

先比较两个模式，再锁定一个用于正式 M1/M3：

| 模式 | 训练参数 | 冻结参数 | 用途 |
|---|---|---|---|
| M-P1 | Content Head + GCA | WAN、VAE、Qwen-VL、Understanding、Action Expert | 最低成本、最强单模块验证 |
| M-P2 | Content Head + GCA + Action Expert（较小 LR） | WAN、VAE、Qwen-VL、Understanding | 允许动作模型适应新内容信号 |

不建议第一轮解冻 WAN 或 Understanding Expert：

- 显存和通信成本显著增加；
- observation-only token cache 会随 backbone 更新而失效；
- 更难把改进归因于 content adapter；
- 本机相邻 Motus smoke 表明原生约 5.89B trainable 参数需要 8 卡 ZeRO 才能稳定运行。

P 模式只用 M1/`lambda_ctr=0` 的独立 dev rollout 选择，不按 M3-M1 gap 选择，避免为方法组单独调模式。

## 6. 双数据源训练

### 6.1 Official action stream

用途：提供 Motus 原生 action flow-matching supervision。

推荐数据：三个任务各 50 Clean + 500 Official Random，共 1,650 episodes，并严格保持作者 Motus temporal contract：

- T-shaped 三相机 composite；
- 384x320；
- state/action 14D；
- action chunk 16；
- `global_downsample_rate=3`；
- `video_action_freq_ratio=2`；
- action/state 使用作者训练与部署实际采用的 raw qpos；作者部署包中的
  min/max stats 会被 lineage 绑定，但当前发布路径没有使用它们。
- 在 LeRobot 适配中，当前 state 与未来 action 都取 `action` 列，以匹配
  Motus converter 的 `joint_action/vector -> qpos.pt`；不使用语义不同的
  FastWAM `observation.state` 列。

现有 Motus RoboTwin dataset 的 `__getitem__` 会内部随机选 task/episode/frame，不能直接用于严格 M1/M3 配对。适配实现改为 step-addressed deterministic sampler：现有 50 Hz release 轨迹以 stride 5 精确选帧，得到与作者 30 Hz/stride 3 相同的 10 Hz、16-step、1.6 秒物理时间窗口，不使用插值；每一步的 episode、frame、action noise 和 timestep 均可复现并写入 audit。

### 6.2 Paired content stream

用途：只计算 contrastive loss，不提供 action loss。

可复用 FastWAM 已生成的 600 scene episodes：

- 3 tasks；
- 每任务 50 physical trajectories；
- 每个 trajectory 有 C/R1/R2/R3 四个同状态 scene；
- 当前训练 state bank 为每任务 240 states，三任务 720 groups、2,880 views。

复用的是原始图像、task/state identity 和四视图分组，不复用 FastWAM Layer-16 token cache。必须重新执行 Motus 的相机合成、resize、VAE/WAN preprocessing，并生成 Motus-specific cache。

因为 paired stream 不做 action supervision，其 50Hz/32-step action contract 不限制本次 M1/M3 contrastive 使用。若以后做 M2 naive action augmentation，必须另建符合 Motus 16-step temporal contract 的数据，不能直接复用现有 action labels。

### 6.3 Paired loss

每个物理状态 group 为 `{C,R1,R2,R3}`：

- 同 group 其余 3 views 是 positives；
- 同 task、不同 physical state 是 negatives；
- 跨 task 默认不作为主 negatives，避免把 task identity 学成捷径；
- M1 也读取完全相同的 paired batch，但 `lambda_ctr=0`；
- M3 使用预注册的 `lambda_ctr>0`，初始候选沿用 0.1。

必须把 official RNG 与 paired RNG 分流，并证明同一 training seed 的 M1/M3 两条实际序列 SHA 完全一致。

## 7. 实验矩阵

| 组别 | 初始化 | Head/GCA | Action Expert | Paired stream | 目的 |
|---|---|---|---|---|---|
| M0 | 作者 Motus_robotwin2 | 无 | 不继续训练 | 无 | 作者基座参考 |
| M1 | 同一作者 checkpoint | 有 | 按锁定 P 模式 | 读取但 `lambda=0` | 架构 + action continuation control |
| M3 | 同一作者 checkpoint | 与 M1 同初值 | 与 M1 相同 | 同序、`lambda>0` | 主方法 |

主结果是 `M3-M1`。`M3-M0` 同时包含 adapter、continued action training 和 contrastive，只能作为总体适配增益。

建议顺序：

1. M0 deployment/dev gate；
2. M-P1/M-P2 各一个短训 checkpoint，使用 M1/`lambda=0` dev SR 选模式；
3. 锁定 LR、steps/global batch、lambda、data ratio、dev/final seed bank；
4. 正式训练 M1/M3，training seeds 建议 3 个；
5. 最终只评 Clean 和 Official Random，每任务每域 100 episodes；
6. 同一 training seed 内 M1/M3 使用相同 simulator 起始 seed 和作者 episode-selection 机制；如果未锁定 exact accepted episodes，只表述为 shared-start comparison，不声称逐 episode 完全配对。

报告：每任务 Clean/Random SR、三任务 macro、3 training seeds mean±std、M3-M1 paired-by-training-seed delta。Style/State/Ratio/R@1 只作为 representation 诊断，不进入 policy 主表。

## 8. 建议代码布局

尽量不直接改作者核心文件，先建立 experiment-owned wrapper：

```text
experiments/robotwin/policy_content_adapter/
  model.py                  # Head/GCA 与 Motus wrapper
  observation_content.py   # current-frame VAE + video-only WAN layer features
  data_official.py          # exact 3-task official manifest/sampler
  data_paired.py            # C/R1/R2/R3 grouping与Motus preprocess
  losses.py                 # multi-positive supervised contrastive
  train.py                  # dual-stream M1/M3 trainer
  extract_cache.py          # Motus-specific frozen content token cache
  checkpoint.py             # compact adapter/Action Expert checkpoint + resume
  lineage.py                # author checkpoint/config/stats SHA manifest
  config_audit.py           # fail-closed config/fairness audit
  rollout.py                # 严格 Motus/RoboTwin deployment wrapper
  evaluation.py             # 100-episode result/aggregate audit
  configs/
  tests/
```

可能需要对作者模型做的最小改动只有一个稳定 extension point：允许外部 wrapper 在 `ActionExpert.input_encoder` 输出后注入 tokens，或让 wrapper monkeypatch/forward-hook 实现。若 hook 无法安全处理 gradient checkpointing、DeepSpeed 和 inference 多步复用，再对 `models/motus.py` 增加可选 callback；默认 `None` 时原生行为必须不变。

原作者训练器已有定期 checkpoint 机制，但正式适配要扩展保存：Head/GCA、可选 Action Expert、optimizer、scheduler、global step、sampler/RNG states、配置/数据/lineage SHA，并真实做 interrupted-resume sequence equivalence smoke。

## 9. 分阶段实施与 gate

### Phase A：资产与 lineage

- 获取并 hash `Motus_robotwin2` checkpoint；
- 锁定 WAN/Qwen/VAE/config/stats/source revision；
- checkpoint strict-load 审计 missing/unexpected keys；
- 精确验证 task、camera、action chunk 和 raw-qpos contract。

通过标准：不可变 lineage manifest PASS，M0 单 action finite，原生 checkpoint 可部署。

### Phase B：数据

- 准备三任务 1,650 official episodes 的精确 manifest；
- 把 600 paired scenes 绑定为 Motus 四视图 manifest；
- 验证同状态图像配对、无 task/split 泄漏；
- 建立 dev/final simulator seed banks，互不重叠。

通过标准：逐域计数、所有文件 SHA、camera/action/raw-qpos audit PASS。

### Phase C：模型接入

- observation-only layer feature extractor；
- Content Head/GCA；
- zero-gate equivalence；
- Zc 在 denoising steps 间只计算一次；
- checkpoint round-trip。

通过标准：CPU unit tests + 单卡/8卡 forward smoke + action-path gradient audit PASS。

### Phase D：representation pilot

- 比较 layer 8/16/24；
- 只用 Train/Val paired split；
- 冻结一次 layer 和 Head 超参数；
- Test 只作历史/辅助诊断。

通过标准：预注册指标与选择规则输出一次，不用 policy final seeds 调 layer。

### Phase E：policy smoke 与 P-mode

- M1/M3 各 3-20 optimizer-step pair smoke；
- 证明 M1/M3 init/data/RNG/steps 一致；
- M-P1/M-P2 用 M1 dev rollout 选择；
- 真实中断恢复 smoke。

### Phase F：正式训练与评测

- 先跑一个 seed 的 M1/M3 完整 paired run；
- 在线 100-episode/cell 检查趋势；
- 协议不变后补齐另两个 seeds；
- 最后统一聚合 M1/M3，M0 只作参考。

## 10. 资源与时间预估

当前只有硬件量级，不应在 smoke 前承诺精确时长：

- 相邻 Motus 代码在本机 8x72GB、ZeRO-1、microbatch 1 下完成过全模型一步 smoke；单卡与 2/4 卡方案曾 OOM。
- M-P1/M-P2 冻结 WAN/Understanding 后应明显低于原生全模型训练，但仍需执行大部分 joint forward；真正峰值必须实测。
- observation-only cache extraction 是额外 WAN forward，可独立后台生成；只要 WAN 冻结，一份 cache 可跨 M1/M3 和所有 training seeds 共享。
- 在线评测成本仍是主要墙钟项：3 tasks x 2 domains x 100 episodes x checkpoints。

推荐资源策略：

1. CPU tests；
2. 1-GPU observation/cache shape smoke；
3. 8-GPU 1-step train smoke；
4. 根据显存决定 global batch，优先保持 M1/M3 相同；
5. 评测按 task/domain 多卡并行，但每个 cell append-only、独立审计。

## 11. 已知风险与决策点

1. **Stage-3 checkpoint 已取得**：仍须以 strict-load 和 zero-gate 的真实
   GPU 结果确认运行契约。
2. **layer 16 不一定最优**：Motus token 语义不同，必须先做小型 layer pilot。
3. **额外 observation branch 成本**：需要缓存并在 inference 中每次 replan 只算一次。
4. **GCA 可能与原生 joint attention 功能重叠**：M1 控制架构本身，M3-M1 才能隔离 contrastive。
5. **作者 dataset 采样不确定**：必须改为 manifest-driven，而不是只设一个全局 seed。
6. **action value 漂移**：作者代码加载 stats 但实际训练/部署走 raw qpos；正式审计必须防止任一侧意外启用 normalization。
7. **DeepSpeed/resume**：hook、冻结参数和 compact checkpoint 必须在 8 卡下实际验证。
8. **C2 不应混入主线**：现有 paired action 时间协议与 Motus 不同，naive action augmentation 后置。

## 12. 推荐的第一版锁定方案

- Base：作者 `Motus_robotwin2`；
- Tasks：Place A2B Left、Open Microwave、Move Stapler Pad；
- Official stream：50 Clean + 500 Random/task；
- Paired stream：现有 Train 30 trajectories/task，4 views，240 states/task；
- Content branch：current-frame-only frozen WAN，先比较 layers 8/16/24；
- Head/GCA：沿用 FastWAM 结构，随机同 seed 初始化；
- M1/M3：相同双数据流，M1 `lambda_ctr=0`，M3 初始候选 `0.1`；
- P-mode：先 M-P1/M-P2 的 M1 dev pilot，只选择一次；
- Formal seeds：三个训练 seeds；
- Final eval：每 task/domain 100 episodes，Clean + Official Random；
- C2：后置，不阻塞主实验。

下一步是完成正在运行的 GPU artifact gate，再跑 8-GPU matched M1/M3
smoke；未通过前不启动正式长训。
