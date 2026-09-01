# Motus Policy Content Adapter engineering gate report

Date: 2026-08-30 UTC

## Result

The Motus-specific Policy Content Adapter implementation passed its CPU,
artifact, and matched 8-GPU M-P1 smoke gates. This is an engineering result;
no formal long training or RoboTwin success-rate result is claimed here.

- CPU suite: 52 passed.
- Author checkpoint strict load: PASS, zero missing/unexpected keys.
- Zero-gate action identity: PASS.
- Motus Layer-16 paired cache: 720 physical states, 2,880 views, PASS.
- M1/M3 smoke: 3 optimizer steps each, world size 8, global batch 8.
- Strict M1/M3 audit: PASS across 24 matched rank-step rows.
- M1 uses `lambda_contrastive=0`; M3 uses `0.1`.
- Both runs changed Head/GCA and kept the frozen Action Expert bit-exact.
- Step 2 and step 3 distributed checkpoints include optimizer, scheduler,
  RNG, global-step, sampler ancestry, and resume sidecars.
- Both compact deployment checkpoints pass the rollout checkpoint validator.

An additional 8-GPU M-P2 smoke also passed for both M1 and M3. It used the
same three-step/data-sequence contract, trained the Action Expert at `1e-5`,
and verified nonzero, finite Action Expert gradients and changed final tensor
SHA values. Its strict 24-row M1/M3 pair audit passed.

## Authoritative artifacts

- Implementation audit:
  `/root/motus_policy_artifacts/motus_v1/audits/implementation_audit.json`
- Layer-16 cache:
  `/root/motus_policy_artifacts/motus_v1/layer16_token_cache`
- Final materialization and pair audit:
  `/root/motus_policy_artifacts/motus_v1/materialized/mp1_seed1_smoke_retry7`
- Final M1/M3 run root:
  `/root/motus_policy_artifacts/motus_v1/runs/mp1_seed1_smoke_retry7`
- Full log:
  `/root/motus_policy_artifacts/motus_v1/mp1_seed1_smoke_retry7.log`
- M-P2 materialization and pair audit:
  `/root/motus_policy_artifacts/motus_v1/materialized/mp2_seed1_smoke_v1`
- M-P2 M1/M3 runs:
  `/root/motus_policy_artifacts/motus_v1/runs/mp2_seed1_smoke_v1`

## Next scientific stage

Use the existing rollout tooling to run M0 as the unmodified author release
reference and to compare formally trained M1/M3 checkpoints on three tasks,
Clean and Official Random, 100 episodes per cell. Formal policy conclusions
must come from those online success rates, not from this smoke.
