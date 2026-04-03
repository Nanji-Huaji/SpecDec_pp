## Accuracy Prediction Head Checkpoints

This directory stores acceptance prediction head artifacts used by SpecDec++ and
tri-decoding style experiments.

### Recommended Layout

Prediction heads should be organized by model pair instead of a single model
name, because a head belongs to a directed edge in the decoding graph:

- `small_draft_acc_head_path` corresponds to `little_model -> draft_model`
- `draft_target_acc_head_path` corresponds to `draft_model -> target_model`

Recommended path layout:

```text
checkpoints/
  acc_head/
    <source_alias>--to--<target_alias>/
      <run_name>/
        config.json
        model.safetensors
        README.md
        trainer_state.json
```

Examples:

```text
checkpoints/acc_head/vicuna-68m--to--tiny-vicuna-1b/w6-l3/
checkpoints/acc_head/tiny-vicuna-1b--to--vicuna-13b-v1.5/w6-l3/
checkpoints/acc_head/qwen3-0.6b--to--qwen3-1.7b/w6-l3/
checkpoints/acc_head/qwen3-1.7b--to--qwen3-14b/w6-l3/
```

Recommended alias format:

- lowercase
- keep the model family prefix, for example `qwen3-1.7b`
- avoid absolute paths and Hugging Face repo ids in directory names

Recommended run name format:

- `w{weight}-l{layers}`
- append extra settings only when needed, for example `w6-l3-mix0.15`

### Registry And Fallback Rule

The project uses `checkpoints/acc_head_registry.json` as the primary registry for
known prediction heads.

Lookup order:

1. resolve the pair through the registry
2. if the pair is not registered, fall back to the default naming rule

Default local path rule:

```text
src/SpecDec_pp/checkpoints/acc_head/<source_alias>--to--<target_alias>/<run_name>
```

Default run name:

- `exp-weight6-layer3`
- special case: `qwen1.5-0.5b-chat--to--qwen1.5-1.8b-chat` falls back to `exp-weight-layer3`

This design keeps extension simple:

- if a pair is stable, add it to the registry
- if a new pair follows the default naming convention, no registry update is required

You can also generate the pair name or path directly from the command line:

```bash
python -m src.acc_head_registry "Qwen/Qwen2-1.5B" "Qwen/Qwen2-3B"
python -m src.acc_head_registry "Qwen/Qwen2-1.5B" "Qwen/Qwen2-3B" --format default-path
python -m src.acc_head_registry "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" --format resolved-path
```

In Python, use:

```python
from src.acc_head_registry import build_acc_head_pair_name, resolve_acc_head_path

pair_name = build_acc_head_pair_name("Qwen/Qwen2-1.5B", "Qwen/Qwen2-3B")
acc_head_path = resolve_acc_head_path("Qwen/Qwen2-1.5B", "Qwen/Qwen2-3B")
```

Callers should prefer these helpers over hardcoding checkpoint paths.

### Legacy Layout

Some existing checkpoints are still stored under legacy directories named after
the destination model, for example:

```text
checkpoints/qwen-3-14b/exp-weight6-layer3
checkpoints/vicuna-v1.5-13b/exp-weight6-layer3
```

`exp.py` currently supports both:

- the new pair-based lookup
- a fallback to the legacy destination-model layout

This means existing experiments keep working while the repository transitions to
the new layout.

### Known Legacy Mappings

| Legacy Directory | Source Model | Target Model | Suggested New Directory |
| ---------------- | ------------ | ------------ | ----------------------- |
| `llama-1.1b/exp-weight6-layer3` | `llama-68m` | `tiny-llama-1.1b` | `acc_head/llama-68m--to--tiny-llama-1.1b/w6-l3` |
| `llama-13b/exp-weight6-layer3` | `tiny-llama-1.1b` | `llama-2-13b` | `acc_head/tiny-llama-1.1b--to--llama-2-13b/w6-l3` |
| `llama-2-chat-70b/exp-weight6-layer3` | `llama-2-7b-chat` | `llama-2-chat-70b` | `acc_head/llama-2-7b-chat--to--llama-2-chat-70b/w6-l3` |
| `tiny-vicuna-1b/exp-weight6-layer3` | `vicuna-68m` | `tiny-vicuna-1b` | `acc_head/vicuna-68m--to--tiny-vicuna-1b/w6-l3` |
| `vicuna-v1.5-13b/exp-weight6-layer3` | `tiny-vicuna-1b` | `vicuna-13b-v1.5` | `acc_head/tiny-vicuna-1b--to--vicuna-13b-v1.5/w6-l3` |
| `qwen1.5-1.8b/exp-weight-layer3` | `qwen1.5-0.5b-chat` | `qwen1.5-1.8b-chat` | `acc_head/qwen1.5-0.5b-chat--to--qwen1.5-1.8b-chat/w6-l3` |
| `qwen1.5-7b/exp-weight6-layer3` | `qwen1.5-1.8b-chat` | `qwen1.5-7b-chat` | `acc_head/qwen1.5-1.8b-chat--to--qwen1.5-7b-chat/w6-l3` |
| `qwen-3-1.7b/exp-weight6-layer3` | `qwen3-0.6b` | `qwen3-1.7b` | `acc_head/qwen3-0.6b--to--qwen3-1.7b/w6-l3` |
| `qwen-3-14b/exp-weight6-layer3` | `qwen3-1.7b` | `qwen3-14b` | `acc_head/qwen3-1.7b--to--qwen3-14b/w6-l3` |

### Storage Recommendation

Use the main repository to store:

- path conventions
- metadata and README files
- lightweight download helpers

Do not rely on git to store all weight files long term. Prefer one of these:

1. Local-only artifacts under `checkpoints/acc_head/`, ignored by git.
2. A dedicated Hugging Face repository for published or shared prediction heads.

A dedicated Hugging Face repository becomes worthwhile when at least one of the
following is true:

- multiple machines need the same heads
- multiple people need to reuse the same heads
- you want versioned publication of stable checkpoints
- checkpoint size starts polluting the main project workflow

For the current workflow, a good balance is:

1. Keep the directory structure in this repository.
2. Ignore actual weight files in git.
3. Publish stable heads to a Hugging Face repository such as
   `your-org/specdecpp-acc-heads`.
4. Keep a small mapping file or README here that points to the published heads.

### Uploading To Hugging Face

This repository includes `upload_acc_heads.py` for uploading the pair-based
checkpoint layout to a Hugging Face model repository.

Preview the upload plan:

```bash
cd src/SpecDec_pp
python upload_acc_heads.py \
  --repo-id your-org/specdecpp-acc-heads \
  --dry-run
```

Upload a single pair:

```bash
cd src/SpecDec_pp
python upload_acc_heads.py \
  --repo-id your-org/specdecpp-acc-heads \
  --pair qwen3-1.7b--to--qwen3-14b
```

Upload all pair directories:

```bash
cd src/SpecDec_pp
python upload_acc_heads.py \
  --repo-id your-org/specdecpp-acc-heads
```

By default, `trainer_state.json` is skipped. Add `--include-trainer-state` if
you want to upload it as well.
