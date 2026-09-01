# CausalWAM

This repository is a source-only handoff snapshot for the CausalWAM experiments.

- `FastWAM/`: the primary implementation, Pair-280 post-training, audits, tests, and RoboTwin integration.
- `Motus/`: the parallel Motus adaptation work.
- `docs/`: experiment protocol and cross-server handoff documents.

Large checkpoints, datasets, feature caches, simulator assets, optimizer states, logs, and rollout videos are intentionally excluded from Git. See [`docs/H100_HANDOFF_20260901.md`](docs/H100_HANDOFF_20260901.md) and [`docs/modelscope_upload_manifest_20260901.json`](docs/modelscope_upload_manifest_20260901.json).

The upstream projects remain credited in their respective subdirectories. This snapshot does not replace their licenses.
