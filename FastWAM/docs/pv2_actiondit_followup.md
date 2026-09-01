# P-v2 ActionDiT Follow-up Protocol

## 实验定位

本实验是查看 P-v1 正式结果之后新增的机制研究。P-v1 仍是主实验；本实验不能替代其结论，也不能被描述为预注册主结果。

- 不可变 P-v1 主实验：`outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1`
- P-v2 新输出：`outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1`
- 研究问题：当 ActionDiT 允许以低学习率适配时，paired contrastive supervision 是否能稳定提高在线 Policy Success Rate？

## 严格对照

两组都从同一个作者 release checkpoint 初始化，Video DiT、VAE 和 T5 冻结，ActionDiT、Content Head 与 GCA 训练：

| 组别 | ActionDiT | Head/GCA | `lambda_contrastive` |
| --- | --- | --- | ---: |
| C1-P-v2 | 训练，LR `1e-5` | 训练，LR `1e-4` | 0.0 |
| C3-P-v2 | 训练，LR `1e-5` | 训练，LR `1e-4` | 0.1 |

同一个 training seed 内，C1/C3 的基座、初始张量、official/paired 顺序、step RNG、action noise/timestep、optimizer、batch、精度和 1800-step budget 必须相同。唯一允许差异是 contrastive coefficient 与对应 gradient。

每步使用 1 个 official action sample，加 2 个 paired physical-state groups（每组4 views，因此是8 views）。1800步共抽取1800个 official samples，并产生3600次 paired-group draws。后者等于720个训练 states 的5倍 exposure budget；前者只约为466,240-frame official subset 的0.00386轮。因此这是 release checkpoint 上的受控短程适配，不是完整数据 epoch 或从头预训练。

训练 seeds 锁定为 `[1,2,3]`。Pilot 只使用 seed 1。

## 数据与初始化

- 作者 release checkpoint SHA：`776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63`
- stats SHA：`7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`
- 复用 native-50Hz paired 数据、official full-550/task stream、paired Layer-16 cache 和文本缓存。
- Video DiT 冻结，所以 Layer-16 cache 在 P-v2 下仍有效。
- 作者 checkpoint 内 824 个 ActionDiT 张量按名称、dtype、shape 与原始 bytes 计算一个共同初始 SHA；C1/C3 都必须绑定该审计。

## 执行与停止规则

1. CPU implementation/config/protocol tests 必须 0 failure。
2. 单卡顺序运行 C1/C3 各 3 steps；ActionDiT、Head、GCA 必须有 finite gradient 与实际参数更新，checkpoint 必须产生三个任务的 finite 14-D action。
3. GPU0/1 并行训练 seed 1 的 C1/C3，各 1800 steps，`world_size=1`。
4. 原 materialization 将 seed 53 pilot 锁为每格 20 episodes；但在两组 rollout 尚未形成任何 `completed_rollouts.json`、也未读取其结果作决策时，用户于 2026-08-22 明确要求对齐原始 FastWAM RoboTwin 的 100-episode 评测规模。因此新增不可变 `eval100` amendment：每个 control 的三个任务分别运行 Clean 100 和 Official Random 100，共 600 episodes/control。旧 20-episode partial 目录只作为 `INVALID_ABORTED_NOT_USED` 证据保留。
5. Pilot 同时满足以下两项才通过：
   - `C3-C1 Official Random macro >= +0.03`
   - `C3-C1 Clean macro >= -0.03`
6. Pilot 未通过：停止扩展，不做结果驱动调参，输出失败机制分析。
7. Pilot 通过：才可训练 seeds 2/3，并在未打开的 simulator seed 59 上对三个 training seeds 做 100 episodes/cell 的确认性评测。

seed 53 与旧 dev seed 23、旧 author-stock seed 42 不重叠。seed 59 的确认性 bank 在 pilot 决策前不得用于 rollout。

在线评测沿用作者 evaluator 的候选 seed + expert-valid filtering。C1/C3 使用相同起始 seed 和设置，但不声称最终接受的每个 episode 完全一一配对；pilot gate 只比较任务级与 macro SR。

这次 amendment 只改变在线 episode 数及由此派生的 seed-bank identity；训练 checkpoint、training seed、simulator seed 53、任务/域、gate 阈值、模型、loss、LR、batch 和 1800 steps 均不变。任何 cell 少于 100 episodes，或缺少六格中的任一格，pilot gate 都会 fail closed。

## Seed-1 Pilot 实际结果（2026-08-22）

严格 100-episode/task/domain pilot 已完成 1200 个 Policy episodes，并通过两项预设 gate：

| Task | C1 Clean | C3 Clean | Δ | C1 Random | C3 Random | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Place A2B Left | 81% | 77% | -4 pp | 81% | 84% | +3 pp |
| Open Microwave | 28% | 42% | +14 pp | 64% | 69% | +5 pp |
| Move Stapler Pad | 25% | 21% | -4 pp | 33% | 39% | +6 pp |
| Macro | 44.67% | 46.67% | **+2.00 pp** | 59.33% | 64.00% | **+4.67 pp** |

Random 要求 `>= +3 pp`，实际 `+4.67 pp`；Clean guard 要求 `>= -3 pp`，实际 `+2.00 pp`。因此 gate PASS，seeds 2/3 和 seed59 confirmatory bank 按原条件解锁。该结果仍是 post-hoc pilot，不替代 P-v1 主实验。

## 条件扩展

- seeds 2/3 的 C1-P-v2/C3-P-v2 继续使用完全相同的 1800-step 配方；同 seed 内只允许 contrastive coefficient/gradient 不同。
- confirmatory bank 使用 simulator seed 59，显式与 seed23、seed42、seed53 candidate ranges 分离。
- seed1/2/3 的六个 checkpoint 都在 seed59 上运行三个任务的 Clean 100 + Official Random 100，共 3600 confirmatory Policy episodes。
- 最终报告三 training seeds 的 mean ± std、matched-seed `C3-C1` 和 Student-t 95% CI（df=2）；不声称逐 episode 配对。

## 关键产物

- `materialization_manifest.json`
- `manifests/mechanism_protocol.json`
- `manifests/action_dit_initialization_audit.json`
- `manifests/dev_seed53_bank.json`
- `manifests/dev_seed53_100ep_bank_v1.json`
- `manifests/eval100_user_amendment_v1.json`
- `implementation_protocol_audit.json`
- `smoke/strict_smoke_audit.json`
- `pilot_posttrain_audit.json`
- `pilot_decision.json`

100-episode rollout 写入新目录 `pilot_rollouts_100ep_seed53_v1/`，禁止覆盖或拼接旧 `pilot_rollouts/`。

所有科学产物采用 create-only；状态使用 append-only event log。旧 P-v1 目录不得覆盖、删除或改写。

## 后台入口

安全默认只做 CPU prepare：

```bash
PHASE=prepare RUN_TESTS=1 \
bash experiments/robotwin/policy_content_adapter/run_pv2_actiondit_followup.sh
```

GPU 阶段需要显式 `CONFIRM_GPU_WORK=YES`，并由脚本先执行 GPU/Vulkan/SAPIEN preflight。长任务放在 tmux 中运行；脚本不会在 pilot gate 之前自动开启 seeds 2/3 或 seed59。
