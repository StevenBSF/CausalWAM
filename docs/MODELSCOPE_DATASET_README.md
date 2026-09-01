# CausalWAM artifacts

This internal dataset stores the large artifacts for the source repository:

```text
https://github.com/StevenBSF/CausalWAM
```

The authoritative transfer inventory is `docs/modelscope_upload_manifest_20260901.json`; the execution handoff is `docs/H100_HANDOFF_20260901.md`.

## Layout

- `artifacts/checkpoints/`: author FastWAM release model, stats, T5, VAE, and tokenizer.
- `artifacts/pair280_layer16_v1/`: Pair-280 Layer-16 cache, state bank, protocol, final seed1/C3 checkpoint, and audits.
- `artifacts/FastWAM/outputs/`: small hash-bound provenance manifests.
- `packages/official-text-cache/`: deterministic split tar of 68,704 official prompt embeddings.
- `packages/official-three-task-subset/`: deterministic split tar containing the exact 1,650 official episodes plus unchanged global metadata.
- `packages/native50hz-paired/`: deterministic split tar of the native 50Hz C/R1/R2/R3 paired dataset.
- `packages/robotwin-assets/`: deterministic split tar of the audited RoboTwin runtime assets.

Every split-tar directory contains `SHA256SUMS`, `CONTENTS.txt`, and `.package_complete`. Verify and extract with:

```bash
cd packages/<bundle>
sha256sum -c SHA256SUMS
cat <bundle>.tar.part-* | tar -xf - -C <target-directory>
```

The official three-task subset is extracted directly into the desired `robotwin2.0` dataset root. The other packages preserve their `FastWAM/...` prefix and should be extracted from the parent directory of the cloned `CausalWAM` repository.

## Download examples

Always clear proxy variables when direct ModelScope access is available:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY='*' no_proxy='*'

modelscope download StevenHZ/CausalWAM --repo-type dataset \
  --local-dir /data/CausalWAM-modelscope \
  --include 'artifacts/checkpoints/**' 'artifacts/pair280_layer16_v1/**' \
            'packages/official-text-cache/**' \
            'packages/official-three-task-subset/**' \
            'packages/native50hz-paired/**'
```

Do not report partial rollout artifacts as completed results. The ModelScope upload deliberately excludes the nine superseded optimizer checkpoints; only the final step-18,215 resume state is optional.
