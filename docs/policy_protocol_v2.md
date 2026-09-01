# Policy Protocol v2（Release-base 主线修订）

> **权威协议，2026-08-19。** 当前主实验改用作者发布的 RoboTwin 50-task 全任务 checkpoint 作为固定共同基座；按本协议绑定的 release 数据分区为每任务 50 Clean + 500 Official Random。Content Head、GCA、paired contrastive、P-v1/P-v2 和在线评测方法均不改变。原“三任务 Stage 1 基座训练”及其后续验证后移为独立扩展实验，不删除，也不与当前主结果混表。旧 30 Hz 数据、E0–E3、旧 cache/checkpoint 和 P-v1/P-v2 smoke 继续作为 legacy 证据保留。

## 1. 当前主线共同基座：作者 50-task release checkpoint

当前主线固定使用：

```text
B_release:
  checkpoint = /mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
  checkpoint_sha256 = 776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63
  stats      = /mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
  stats_sha256 = 7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095
```

`B_release` 是作者发布的全 RoboTwin 预训练 FastWAM，不是我们重新训练的三任务专用基座；其 release 数据按本协议绑定为每任务 50 Clean + 500 Official Random 的分区。三个目标任务已经包含在作者预训练任务中，因此本实验属于 **pretrained-policy adaptation / continued adaptation**，不是 zero-shot、未见任务泛化或从头训练实验。

正式运行前必须一次性锁定并审计：

- checkpoint 文件 SHA-256、大小和加载 key coverage；
- 配套 dataset stats 的 SHA-256，且 C1/C3（以及可选 C0 reference）使用同一份 stats；
- 模型配置、源码版本、三相机顺序、图像预处理和 action normalization；
- 作者对训练任务数与数据组成的声明来源。release 数据含 27,500 episodes，即 50 个任务块 × 550；但元数据没有内生 Clean/Random domain 字段，因此“每块前 50 Clean、后 500 Random”必须作为本协议的 hash-bound 分区规则记录，不能写成 checkpoint payload 自带的独立证明。

### 1.1 主实验分叉

- **C0（可选 reference）**：若已有同协议原始 FastWAM 结果，可附加原生 `B_release`，不做 Stage 2 训练；当前 C1/C3 主实验不要求重新运行它；
- **C1**：从 `B_release` 初始化，加入随机初始化的 Content Head/GCA，不施加 paired contrastive gradient；
- **C3**：从同一个 `B_release` 初始化，加入与 C1 完全相同的 Content Head/GCA，并施加 paired contrastive gradient；
- C1/C3 在每个 Stage 2 seed 内共享初始化、official 数据顺序、训练步数和所有公共随机流。

当前主因果比较是：

\[
\text{C3}-\text{C1},
\]

它只回答：在固定作者全任务预训练 FastWAM 上，paired contrastive supervision 是否改善三个目标任务的 official Random robustness，同时保持 Clean success rate。

### 1.2 修订原因和结果边界

按三任务 Stage 1 启动阶段前 10 steps 的实测速率推算，8-GPU 完整训练约为 91 小时/seed，三个独立基座顺序运行约需 11–12 天；这是早期 ETA，不是端到端完工计时。该主线修订由计算成本和执行效率触发，不依据任何 C0/C1/C3 在线结果；修订时没有可用于挑选方法的 Stage 1 Policy success 结果。

因此主实验可以声称“对作者全任务预训练策略的增量提升”，不能声称：

- 方法已在从头训练或三任务专用基座上得到验证；
- 方法泛化到未见任务；
- `C3-C0` 完全来自 contrastive loss；
- official Random 的全部提升只来自背景变化。

## 2. 原生官方格式的 Policy Paired 数据

旧 30 Hz C/R1/R2/R3 数据继续服务于 E0–E3，不直接用于正式 Policy C2/C3，也不进行 30→50 Hz 插值。Policy v2 已生成一套原生符合官方 setting 的数据：

```text
FastWAM/outputs/policy_content_adapter/native50hz_three_task_rgb640x480_v1/full_lerobot_v21
```

其 `meta/export_audit.json` 已记录 `status=PASS`、600 scene episodes、50 contents/task、50 Hz、32-step × 14D、`all_pairs_exact=true`、未使用插值，paired manifest SHA 为 `57114b6541b33c0c50e0c5a777f9bed870fe2f4847863f53517e99d8e43637b1`。切换到 `B_release` 不要求重采数据，但正式 cache/训练前仍须把这份现有 artifact 与新的 release lineage、state bank 和 split manifest 重新 SHA-lock。

### 每个样本的格式

每个时刻包含：

- 第一视角相机；
- 左腕相机；
- 右腕相机；
- 当前机器人状态；
- 原生 50 Hz 状态与动作；
- 从当前时刻开始的 32 步、14 维动作目标。

即：

\[
O_t=\{\text{first-person camera},\text{left wrist camera},\text{right wrist camera},\text{robot state}\},
\]

\[
Y_t=[a_t,a_{t+1},\ldots,a_{t+31}],\qquad a_t\in\mathbb{R}^{14}.
\]

### 四种场景版本

C、R1、R2、R3 是同一物理状态的四种场景版本，**不是四个相机**：

- C：Clean；
- R1：随机场景版本 1；
- R2：随机场景版本 2；
- R3：随机场景版本 3。

每种场景版本内部仍有第一视角、左腕和右腕三个同步相机。因此同一物理状态对应：

```text
4 种场景版本 × 每种场景 3 个相机
```

### 正确生成方式

专家策略只执行一次：

1. 记录原生 50 Hz 物理状态、机器人状态和完整动作轨迹；
2. 保存或可确定性恢复每个时刻的模拟器物理状态；
3. 固定机器人、目标物体、相机和未来动作；
4. 只修改不影响动作有效性的视觉因素；
5. 从相同物理状态重新渲染 C/R1/R2/R3；
6. 四种场景版本共享完全相同的未来 32-step action target。

因此：

\[
(C_t,R1_t,R2_t,R3_t)\rightarrow[a_t,\ldots,a_{t+31}].
\]

R1/R2/R3 优先改变背景、纹理、材质、光照和不发生碰撞的装饰物。暂不改变目标物体位置、机器人状态、桌高、相机内外参或参与碰撞的物体，否则共享动作标签可能失效。

禁止让专家分别执行四次后把近似轨迹伪装成 paired data，也禁止用 30→50 Hz 插值生成正式动作标签。

### 数据数量

| Split | Clean/task | Random/task | Physical trajectories/task |
|---|---:|---:|---:|
| Train | 30 | 90 | 30 |
| Val | 10 | 30 | 10 |
| Test | 10 | 30 | 10 |
| 总计 | 50 | 150 | 50 |

Random 由 R1/R2/R3 组成，每条 physical trajectory 对应 `1 Clean + 3 Random scene variants`。每条轨迹会产生大量原生 50 Hz、32-step action windows；30 条轨迹不是 30 张图片。

第一轮 Stage 2 不把整条轨迹的所有相邻帧都当成独立证据，而是从每条 Train physical trajectory 确定性选取 8 个能够完整覆盖未来 32 步、无需末端 padding 的状态。由此每个任务得到 `30 × 8 = 240` 个 physical-state groups，三个任务共 720 groups、2,880 scene views。C2 与 C3 必须绑定同一份 state-bank manifest 和 SHA，并按完全相同的 state 顺序取样。若第一轮结果显示数据不足，再增加新的 physical trajectories 与 style seeds，不靠重复抽取同一轨迹的更多相邻帧扩充样本数。

### 数据生成顺序与当前状态

先为每个任务运行：

```text
1 physical trajectory × C/R1/R2/R3
```

严格验证：

- 三相机身份正确并同步；
- 状态与动作记录为原生 50 Hz；
- 动作为 14 维，顺序、单位和 normalization 与 official stream 一致；
- 每个有效时刻能够生成 future 32-step window，末端 padding/mask 明确；
- C/R1/R2/R3 的物理状态、robot state 和 action target 逐项一致；
- 图像、状态、动作和时间戳对齐；
- 随机化不改变动作有效性；
- task、trajectory、state、scene variant 和 split 可追溯。

该 1×4 pilot→完整 30/10/10 gate 已执行；现有 full export 通过 native action/export contract。这个 PASS 只证明 paired 数据 artifact 合格，不表示 release-base cache、C1/C3 训练或在线 rollout 已完成。

## 3. Stage 2：从同一个 `B_release` 分叉

| 组别 | Head/GCA | Official C+R | 新 Paired C/R1/R2/R3 | 优先级与用途 |
|---|---|---|---|---|
| C0 Author Release | 无 | 不继续训练 | 无 | 可选外部参考；不阻塞主实验 |
| C1 Architecture-only（action-only continuation） | 有 | Action loss | `lambda_ctr=0` | 主线；控制架构与 official continuation |
| C3 Ours | 与 C1 完全相同 | Action loss | `lambda_ctr>0` Contrastive loss | 主线；我们的方法 |
| C2 Naive Aug | 与 C3 相同 | Action loss | Action loss | 后续辅助对照；不阻塞主结论 |

对应训练目标：

### C1/C3：配对主比较

\[
L(\lambda)=L_{\text{action}}(D_{\text{official}})
+\lambda L_{\text{contrastive}}(D_{\text{paired}}).
\]

C1 使用 `lambda=0`，C3 使用预先锁定的 `lambda=lambda_ctr>0`。正式实现固定让 C1/C3 按完全相同顺序读取 official 和 paired batches、执行相同公共 forward/随机流；C1 将 paired contrastive loss 乘零且不得获得该梯度。两者的共同训练数据、Head/GCA 初始 tensor、optimizer、LR、scheduler、步数和 effective global batch 必须逐项相同；唯一允许产生额外梯度的差异是 C3 的 paired contrastive supervision。

### C2：Naive Augmentation（后续辅助对照）

\[
L_{\text{C2}}=L_{\text{action}}(D_{\text{official}})+L_{\text{action}}(D_{\text{paired}}).
\]

C/R1/R2/R3 使用相同动作标签直接训练 Policy，不计算 contrastive loss。

若选择把 C0 加入同一个聚合器，可以为它生成一个仅用于部署的零门控 transport checkpoint；该 transport 必须证明 `gate=0`，并且在相同真实输入上的动作路径输出与原生 `B_release` 逐位一致。它不属于 Stage 2 训练。C0 必须一次性提供完整 3 tasks × 2 domains 六格，并与 C1/C3 使用完全相同的 rollout protocol 和 final seed bank；部分 C0 或不同协议的既有数字不得混入。

### 三个关键比较

- **C3−C1：主结论，paired contrastive supervision 的增量作用；**
- C3−C0：仅当存在上述同协议完整 C0 时的辅助结果；它混入 Head/GCA、action continuation 与 contrastive，不能作纯 contrastive 归因；
- C2−C1：后续辅助结果，普通随机场景 action augmentation 的增益；
- C3−C2：后续辅助结果，方法是否优于普通 augmentation。

新原生数据通过官方格式审计后，C2 解除 blocked 并成为正式对照。旧 30 Hz 数据不能解锁 C2。

## 4. Paired Contrastive 定义

对于同一个物理状态 (s)：

\[
G_s=\{C_s,R1_s,R2_s,R3_s\}.
\]

每个场景版本都把另外三个版本作为 positives：

```text
C  -> R1, R2, R3
R1 -> C,  R2, R3
R2 -> C,  R1, R3
R3 -> C,  R1, R2
```

Negatives 是同一个任务中的不同物理状态。R3 在 Policy v2 中是第三种随机场景训练版本，不是 Policy 测试环境。

## 5. Head 初始化和 Token Cache

正式主比较中：

- C1/C2/C3 使用完全相同的 seeded random Head/GCA 初始化；
- E0–E3 只用于确定 Layer 16、Head 结构和超参数；
- E2 contrastive-trained Head 不进入主比较，只作为可选 warm-start ablation。

若使用预训练 Head，必须另列：

| 组别 | Head 初始化 | Stage 2 contrastive |
|---|---|---|
| C1-random | 随机 | 无 |
| C1-pretrained | 预训练 | 无 |
| C3-pretrained | 与 C1-pretrained 相同 | 有 |

`C1-pretrained` 不能称为纯 Architecture-only。

### Cache 要求

当前主线必须直接从锁定 SHA 的 `B_release` 为新 paired 数据重新提取 Layer-16 tokens：

- 不复用 `e2_train.pt`；
- 不复用或拼接旧 E0/E1 cache；
- 不使用未完成三任务 `B_CR` 产生的任何 cache；
- 创建独立 Policy cache，例如 `policy_release50tasks_native50hz_four_scene_v1`；
- cache manifest 绑定 `B_release` checkpoint/stats SHA、数据 manifest/state-bank SHA、代码与预处理、camera/scene 顺序、layer、dtype、token shape 和 split；
- 对同一真实输入验证 cache extractor 与在线 native prefill 的 Layer-16 tokens bit-exact；
- 只要 Video DiT 在 P-v1/P-v2 中保持冻结，该 cache 可由 C1/C3 和三个 Stage 2 seeds 共享；未来若解冻 Video DiT，冻结 cache 立即失效；
- 旧 E0–E3 cache、checkpoint 和结果不覆盖、不修改。

## 6. P-v1 和 P-v2 的选择

### P-v1

- ActionDiT 冻结；
- 只训练新加入的 Head 和 GCA；
- 显存较低、训练稳定，但适应能力可能有限。

### P-v2

- Head/GCA 正常训练；
- ActionDiT 以较小学习率更新；
- 适应能力可能更强，但显存更高，也可能破坏原模型能力。

旧 smoke 只证明训练、梯度、checkpoint 和加载链路可以工作，没有在线 success rate，因此不能据此选择正式模式。

选择流程：

1. v2 四场景协议实现后，先分别运行 P-v1/P-v2 工程 smoke；smoke 只检查训练、梯度、保存和部署链路；
2. 从同一个固定 `B_release`、同一 Head/GCA 初始化和同一公共训练配方出发，使用 **C1/action-only 目标（`lambda_ctr=0`）** 各短训一个 P-v1/P-v2 dev-pilot checkpoint；
3. 使用同一组专用 online `dev_selection` simulator seed bank，在三个任务的 `demo_clean` 和 `demo_randomized` 上各运行 20 episodes/task/domain；
4. 先排除 Clean macro SR 比两者最佳 Clean 低超过 0.05 的候选，再在剩余候选中选择 official Random macro SR 更高者；若 Random 完全相同，固定选择 P-v1；
5. 将候选 checkpoint、rollout 结果、dev seed bank、规则和胜者写入不可覆盖、SHA-256 绑定的 selection manifest；
6. 后续 C1/C2/C3 全部绑定该 manifest 并使用同一个胜出模式；
7. 最终 test seed bank 必须标记为 `final_test`，显式证明与 `dev_selection` seed bank 不相交，且不参与模式选择。

禁止使用 C3 的 contrastive dev 结果或 `C3-C1` dev gap 选择 P 模式，避免整个 pipeline 的共享结构超参数偏向方法组。当前旧 `p_v1_dev_pilot.yaml`/`p_v2_dev_pilot.yaml` 若仍为 `lambda_contrastive>0`，必须先改为 release-base C1 pilot 并重新审计，不能直接沿用。

简单说：先小规模实战决定冻结还是微调 ActionDiT，再用统一设置比较 C1/C2/C3。

## 7. Policy 最终评测

只使用 RoboTwin 官方在线环境：

- `demo_clean`；
- `demo_randomized`。

`final_test` bank 只能在 P 模式、`lambda_ctr`、temperature、paired/official batch ratio、max steps、LR、三个 Stage 2 seeds 和全部正式 config SHA 均锁定后打开一次。当前必需评测矩阵是 C1/C3 的 36 格；C0 不再阻塞 final test。

每个 Stage 2 training seed、任务和环境使用相同数量的 episodes。正式实验建议 100 episodes/task/environment。所有方法使用相同的 simulator seeds、instruction、action horizon、inference steps、replan 和 success definition。

报告：

- 三任务分别的 Clean SR；
- 三任务分别的 official Random SR；
- 三任务 macro average；
- C1/C3 三个 Stage 2 seeds `[1,2,3]` 的 mean ± std；
- 主比较 C3−C1；若另有同协议完整 C0，再辅助报告 C3−C0；C2 完成后另报 C2−C1/C3−C2。条件允许时附配对置信区间。

正式聚合使用 schema v4、profile `c1_c3_primary`：严格要求 `2 controls × 3 seeds × 3 tasks × 2 domains = 36` 条 C1/C3 records。C0 是可选固定 reference，没有 training-seed 维度；若附加必须完整六格、同协议、同 seed bank，聚合总数为 42。不得复制三次 C0 数字冒充三个 training seed。三个 C1/C3 seeds 只覆盖 Stage 2 Head/GCA 初始化与优化随机性，不覆盖作者预训练基座的不确定性。

Policy 主表不报告 R3 success、Style distance、State distance、State/Style ratio 或 R3→Clean R@1；这些属于 E0–E3 representation 实验。

`demo_randomized` 可能同时包含背景、杂物、光照和桌高变化，因此结论只能表述为“提升 official Random 环境下的 Policy 鲁棒性”，不能把提升完全归因于背景。

## 8. 公平性要求

C1/C3 必须锁定：

- 完全相同的 `B_release` checkpoint、dataset stats、配置、源码、预处理和 SHA；
- 每个 Stage 2 seed 内完全相同的 Head/GCA 初始 tensor，并保存 tensor SHA；
- 相同 P-v1 或 P-v2 freeze/unfreeze contract；
- 相同 official 三任务数据范围、batch 顺序、action noise、diffusion timesteps 和 augmentation；
- 相同 optimizer、参数组 LR、scheduler、weight decay、optimizer steps、precision、gradient accumulation/clipping 和 effective global batch；
- 相同公共随机流，paired stream 不得扰动 official stream 的采样或噪声；
- 相同 rollout simulator seeds、instruction、horizon/replan/inference 参数和 success definition。

正式 Stage 2 seeds 锁定为 `[1,2,3]`。它们只表示 Head/GCA 初始化与 Stage 2 优化随机性；三个 seed 内分别运行配对的 `C1_i/C3_i`，报告三个 paired delta。它们不表示作者预训练基座的不确定性。

在打开任何 P-mode dev rollout 或 final-test bank 前，必须锁定 `lambda_ctr`、contrastive temperature、paired/official batch ratio、max steps、所有参数组 LR 和 config SHA。之后的超参数消融另表报告，不能回写主配置。

正式 config audit 应采用差异 allowlist：C1/C3 除方法名、输出路径和 `lambda_ctr: 0 -> >0`（以及由此必需的 contrastive gradient 开关）外，其他公共字段必须相等。C1/C3 必须绑定同一 paired manifest/state bank，并逐项读取相同 paired 顺序。

C2 保留为后续辅助对照，并与 C3 使用同一 paired 数据、Train/Val split 和 physical-state 顺序。它和 C3 的关键目标差异仍是：

```text
C2：paired data -> Action loss
C3：paired data -> Contrastive loss
```

## 9. 已落地的代码和文档

### 文档

- `docs/policy_protocol_v2.md`：本权威协议；
- `docs/policy_method.md`：方法说明；
- `docs/实验规划.md`：阶段、gate 与运行顺序；
- `docs/codex_goal_policy.md`：标记旧 v1 goal 为 Archived；
- `FastWAM/experiments/robotwin/policy_content_adapter/README_POLICY_CONTENT_ADAPTER.md`：区分 legacy smoke 与 v2 待实施状态。

### 数据采集与验证

已新增并由 fail-closed 单元测试覆盖：

- 原生 50 Hz 三相机 paired 数据采集；
- simulator state 保存/恢复或等价的确定性重渲染；
- C/R1/R2/R3 scene variants；
- 32-step、14D action window；
- 三相机/状态/动作时间同步审计；
- 四场景物理状态与动作一致性审计；
- 三任务 1×4 pilot gate 和完整 30/10/10 生成入口。

不实现 30→50 Hz 插值。

现有 `native50hz_three_task_rgb640x480_v1/full_lerobot_v21` full export 已通过现有 native action/export audit；release-base 主线只需复核并重新绑定 lineage/state-bank/cache provenance，不重新采集同一批数据。

### Policy 代码

已有可复用实现：

- Stage 1 原生三任务训练 wrapper/config；
- `full_550_per_task` loader 和逐域 manifest；
- v2 Layer-16 cache extractor；
- 四 scene variants 的 data/loss/train/config/smoke audits；
- C2 paired action supervision；
- C1/C2/C3 同初始化与 fairness audit；
- P-v1/P-v2 online-dev 选择 manifest 与 dev/final seed-bank 隔离审计；
- 对应 YAML 和单元测试。

但截至本次协议修订，active formal configs、cache extractor、C0 transport、训练审计和结果聚合仍强制绑定三任务 `B_CR` completion manifest。**“主线已改为 `B_release`”是获批的新实验规划，不表示现有正式命令已经完成适配。** 在启动正式 C0/C1/C3 前必须：

- 新增 fail-closed `author_release` base-lineage manifest；
- 将 checkpoint/stats SHA、作者来源和加载审计写入 lineage；
- 把 formal configs、cache extractor、C0 transport、P-mode selection、training/evaluation audit 从 `B_CR` ancestry 改为 `B_release` ancestry；
- 将 `training_seed` 明确定义为 Stage 2 seed，并允许 C0 没有 training seed；
- 更新测试并通过严格配置差异审计。

保持不变：

- E0–E3 代码、数据、cache、checkpoint 和历史结果；
- Content Head/GCA 核心结构；
- rollout policy 核心；
- Clean/Random evaluator 核心；
- `src/fastwam/**` 原生模型语义。

## 10. 推荐默认执行方案与状态

默认顺序：

1. 审计并锁定 `B_release` checkpoint、stats、源码/配置和 SHA，生成 `author_release` lineage manifest；
2. 适配并严格测试 release-base configs、cache/training/evaluation audits；
3. 用 dev seeds 运行 C0 原生部署与 bit-exact gate，不打开 final-test bank；
4. 对新的原生 50 Hz C/R1/R2/R3 数据及 state bank 做最终审计；
5. 从 `B_release` 提取独立的四场景 Layer-16 cache；
6. 从同一个 `B_release` 做 P-v1/P-v2 dev pilot，按预设规则锁定一次胜者；
7. 对 Stage 2 seeds `[1,2,3]` 分别配对训练 `C1_i/C3_i`；
8. 全部设置锁定后，在与 dev 完全分离的 final-test seed bank 上评测 C1/C3 的 `demo_clean` 和 `demo_randomized`；
9. 主表使用 36 格 C1/C3 strict aggregate，并以 `C3-C1` 为唯一主因果结论；已有同协议完整 C0 仅可选附加；
10. 主结果锁定后再运行 C2、E2 Head、query 数和 `lambda_ctr` 等辅助消融；
11. 最后执行下述三任务 `B_CR` 扩展验证；
12. 所有旧协议和产物保留为 legacy，不覆盖、不删除。

截至 2026-08-19，方法核心、原生 paired 数据链路、loader、Head/GCA、C1/C2/C3 loss、P-mode 选择和在线评测代码可复用；release-base formal lineage 与各级 fail-closed gate 仍需适配。任何已启动或部分完成的三任务 Stage 1 run 均不进入当前主结果，也不据其 loss 选择方法。

## 11. 后移实验：三任务 `B_CR` 基座及其后续复验

原 Stage 1 不取消，而是作为 **base-regime sensitivity extension** 后移。它同时改变基座训练语料、任务规模和训练来源，不应被简化称为纯 initialization ablation；它与 release-base 主实验分开命名、分开表格、分开结论。

每个三任务基座仍使用：

| Task | Clean | Official Random | 总数 |
|---|---:|---:|---:|
| Place A2B Left | 50 | 500 | 550 |
| Open Microwave | 50 | 500 | 550 |
| Move Stapler Pad | 50 | 500 | 550 |
| 合计 | 150 | 1500 | 1650 |

训练定义保持不变：从 Wan2.2 + 作者 ActionDiT 初始化，使用原生 `L_video + L_action`、5 epochs，不加入 Head/GCA/contrastive。已记录的显存修订仍为 8 GPUs、micro-batch 8/GPU、gradient accumulation 2、effective global batch 128；正式基座 seeds 为 `[1,2,3]`。

待资源允许时：

1. 完成 `B_CR^(1/2/3)`；
2. 为每个完成基座重新提取与其 SHA 匹配的 paired Layer-16 cache；
3. 在每个基座内复刻同初始化的 C1/C3；
4. 用同一在线协议检查 release-base 主结论是否对基座训练制度敏感。

这组结果不能补写进 release-base 主表，也不能用当前固定 `B_release` 的三个 Stage 2 seeds冒充三个独立基座。其作用是扩展验证，而不是当前主实验的前置条件。

## 12. 已锁定主结果后的 P-v2 follow-up

P-v1 正式矩阵完成后，允许增加一个独立、明确标记为 post-hoc 的 P-v2 机制实验。P-v2 中 C1/C3 都训练 ActionDiT，并继续共享同一 release base、初始化、数据顺序、step RNG、optimizer/batch/1800 steps；唯一 treatment difference 仍为 contrastive coefficient/gradient。

该 follow-up 原始 materialization 使用 seed1 + simulator seed53 的 20-episode/cell pilot。2026-08-22，在未形成任何完整 pilot manifest、未用 partial 值作决策之前，用户要求与原始 FastWAM RoboTwin 评测规模统一；因此通过 create-only amendment 改为每任务/每域 100 episodes（600 episodes/control）。Random `+3pp`、Clean `-3pp` 双阈值及所有训练设置保持不变，旧 20-episode partial 明确作废。seed59 与 seeds2/3 只有在该 100-episode pilot PASS 后才可解锁。P-v1 产物与结论保持不可变，完整细则见 `FastWAM/docs/pv2_actiondit_followup.md`。

实际 seed53 pilot 于 2026-08-22 完成：`C3-C1` Clean macro `+2.00 pp`、Official Random macro `+4.67 pp`，两项 gate 均 PASS。因此 seeds 2/3 的相同配方训练与此前未打开的 seed59、100-episode/cell confirmatory matrix 已按条件解锁；最终仍须完成 3 seeds × C1/C3 × 3 tasks × 2 domains 的 36 格审计后才能形成机制研究结论。
