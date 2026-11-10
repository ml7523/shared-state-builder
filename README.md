# Shared truncated-state builder

This is a **separate dataset-generation project**. It does not import or run
Yuetong's Budget Forcing/Continue actions. The default and currently
implemented generation method is one ordinary **Single** sample per question.
The generation backend is selected by `--method`; `state_builder.run.METHOD_BACKENDS`
is the extension point for future generation methods. Changing the method
requires a new dataset version and a clear historical-cost policy.

The downloaded `math500_test.jsonl` is the MATH-500 test split pinned to
HuggingFaceH4 commit `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` and
SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`.
The public repository stores the pinned source URL and checksum; it downloads
the data from Hugging Face on first use instead of republishing the test file.
The prompt is copied from the team's Single/SC baseline. Every action that
consumes the resulting states must use the same model checkpoint and tokenizer;
the action sampling protocol is recorded separately from prefix generation.

## What it produces

For each selected problem, the backend generates **one trajectory up to the
`--total-budget` new-token cap**. It saves the exact prompt and complete
generated token ID sequence, raw response, thinking token segment, answer
token segment, seed, stop reason, latency and model/tokenizer fingerprints.
If the model reaches the cap without EOS, the saved trajectory is complete
*as collected*, but the model's solution may still be unfinished.

The builder then chooses cut points **inside unfinished thinking**. It never
uses a prefix containing `</think>` or EOS and leaves at least one global
token available. `--num-cut-points 3` selects three approximately even
positions per eligible trajectory; `--cut-tokens 256 512 1024` requests exact
generated-token counts. Explicit positions outside a question's valid
thinking span are recorded as rejected. A trajectory too short for the
requested number of even cuts yields no states and a skip reason.

The output directory contains:

| File | Contents |
|---|---|
| `trajectories.jsonl` | One full token trajectory per problem; **no gold answer** |
| `states.jsonl` | Exact `prompt_ids`, `prefix_ids`, `total_budget`, `spent_tokens`, provenance and state ID; **no gold answer** |
| `labels.jsonl` | Gold answers keyed by `state_id`, for offline grading only |
| `manifest.json`, `summary.json` | Input and code hashes, model identity, counts and physical collection cost |
| `records/*.json` | Atomic per-problem checkpoints used by `--resume` |

Each state includes `total_budget`, `spent_tokens`, and
`remaining_tokens = total_budget - spent_tokens`. At a Single cut of 512
generated tokens with a 2048-token global budget, the action starts with 1536
tokens remaining. Prompt IDs remain in the context but are not charged to this
**new-token** budget. The builder physically generated the rest of the full
trajectory to create the dataset. That suffix is **dataset-collection compute**,
not historical cost inside a counterfactual rollout. `summary.json` reports
the physical total.
All actions must start from the same saved `state_id`; do not regenerate each
action's prefix independently.

## Setup and run

Run these commands from this directory on a CUDA VM that can load Qwen3-14B
in BF16. The current L4's 23 GB of GPU memory is insufficient for 14B BF16;
use a larger GPU for the shared comparison. The builder defaults to
`Qwen/Qwen3-14B`, matching the two action runners.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m state_builder.dataset  # Download and verify the pinned test split
python -m unittest discover -s tests -v

# Small 14B pipeline check with a real model (incurs GPU cost)
python -m state_builder.run \
  --output runs/single14b_smoke_v1 --method single \
  --model-name Qwen/Qwen3-14B \
  --start-index 0 --num-problems 1 --total-budget 2048 \
  --num-cut-points 3 --seed 42

# Or request fixed token positions for the same model/configuration
python -m state_builder.run \
  --output runs/single14b_fixed_v1 --method single \
  --model-name Qwen/Qwen3-14B \
  --start-index 0 --num-problems 2 --total-budget 2048 \
  --cut-tokens 256 512 1024 --seed 42
```

Choose **one** cut mode. Pin `--revision` for a formal shared run. The default
sampling is temperature 0.6, top-p 0.95, top-k 20, with per-question seed
`(base_seed + MATH500_index) mod 2^32`. Do not treat different model sizes
or quantization as the same shared state dataset.

The output directory must be new. If generation stops after some problems,
rerun the **identical command with `--resume`**. Completed per-problem records
are reused only when the code, configuration, model and tokenizer identity
match. Published JSONL files are never silently replaced with different data.

`states.jsonl` is compatible with the team's `state_id`, `problem_id`,
`prompt_ids`, `prefix_ids`, `total_budget`, `spent_tokens` action schema. Its
optional identity fields match the SC backend fingerprint/signature algorithm.
The states also record `quantization: "none"`, which BF/Continue checks before
running; a 4-bit action model cannot reuse these BF16 prefixes as the same
counterfactual model.
Feed this file to each action runner; keep `labels.jsonl` out of model/action
inputs. The builder and both action runners now default to Qwen3-14B. Use
the same model revision, tokenizer and runtime for every action from a given
state. The complete comparison contract and any remaining CLI differences
are listed in the BF repository's `docs/SHARED_ACTION_INTERFACE.md`.

The MATH-500 test split is for evaluation, so parameter tuning should
use a separate development set.
