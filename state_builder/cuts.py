"""Pure conversion from one full token trajectory to immutable shared states."""

import hashlib
import json


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode()).hexdigest()


def select_cut_positions(lower: int, upper: int, *, num_cut_points: int | None,
                         cut_tokens: list[int] | None) -> tuple[list[int], list[int], str | None]:
    """Select only unfinished-thinking prefixes with at least one token left."""
    if (num_cut_points is None) == (cut_tokens is None):
        raise ValueError("Choose exactly one of num_cut_points or cut_tokens")
    if num_cut_points is not None:
        if type(num_cut_points) is not int or num_cut_points < 1:
            raise ValueError("num_cut_points must be positive")
        span = max(0, upper - lower + 1)
        if span < num_cut_points:
            return [], [], "too_short_for_requested_cuts"
        positions = [lower - 1 + ((span + 1) * k) // (num_cut_points + 1)
                     for k in range(1, num_cut_points + 1)]
        if len(set(positions)) != num_cut_points or not all(lower <= p <= upper for p in positions):
            raise AssertionError("Uniform cut construction failed")
        return positions, [], None
    if not cut_tokens or any(type(p) is not int or p < 1 for p in cut_tokens) or len(set(cut_tokens)) != len(cut_tokens):
        raise ValueError("cut_tokens must be distinct positive integers")
    positions = sorted(p for p in cut_tokens if lower <= p <= upper)
    rejected = sorted(p for p in cut_tokens if p < lower or p > upper)
    return positions, rejected, None if positions else "no_valid_explicit_cuts"


def build_records(*, backend, generation, dataset_index: int, row: dict,
                  total_budget: int, seed: int, num_cut_points: int | None = 3,
                  cut_tokens: list[int] | None = None, temperature: float = 0.6,
                  top_p: float = 0.95, top_k: int = 20) -> tuple[dict, list[dict], list[dict]]:
    """Return full trajectory, gold-free states, and separate labels.

    Counts use the generated-token axis, so prompt tokens never enter spent_tokens.
    Physical generation of the saved suffix is data-collection cost, not the
    historical cost at an earlier counterfactual state.
    """
    output = generation.output_ids
    prompt = generation.prompt_ids
    if (not prompt or not output or len(output) > total_budget or
            any(type(token) is not int or token < 0 for token in prompt + output)):
        raise ValueError("Invalid exact token sequence or total budget")
    eos_at = next((i for i, token in enumerate(output) if token in backend.eos_ids), None)
    if eos_at is not None and eos_at != len(output) - 1:
        raise ValueError("Output includes tokens after the first EOS")
    if (generation.finish_reason not in {"eos", "length"} or
            (generation.finish_reason == "eos") != (eos_at is not None)):
        raise ValueError("Finish reason disagrees with generated token IDs")
    start_at = next((i for i, token in enumerate(output) if token == backend.think_start_id), None)
    end_at = (next((i for i in range(start_at + 1, len(output))
                    if output[i] == backend.think_end_id), None)
              if start_at is not None else None)
    if eos_at is not None and end_at is not None and end_at > eos_at:
        end_at = None
    reasoning_end = (end_at if end_at is not None else
                     eos_at if eos_at is not None else len(output))
    lower = start_at + 2 if start_at is not None else 1
    upper = min(reasoning_end, total_budget - 1)
    if start_at is None:
        positions, rejected, skip_reason = [], [], "missing_think_start"
    else:
        positions, rejected, skip_reason = select_cut_positions(
            lower, upper, num_cut_points=num_cut_points, cut_tokens=cut_tokens
        )
    identity = {"model_name": backend.model_name,
                "quantization": "none",
                "model_revision": backend.model_revision(),
                "tokenizer_fingerprint": backend.tokenizer_fingerprint(),
                "generation_signature": backend.generation_signature()}
    trajectory_id = _digest({"dataset_index": dataset_index, "problem_id": row["unique_id"],
                             "prompt_ids": prompt, "output_ids": output,
                             "seed": seed, **identity})
    trajectory = {
        "trajectory_id": trajectory_id, "method": backend.method,
        "problem_id": row["unique_id"], "dataset_index": dataset_index,
        "question": row["problem"], "seed": seed,
        "prompt_ids": list(prompt), "output_ids": list(output),
        "reasoning_ids": (output[start_at:reasoning_end] if start_at is not None else []),
        "answer_ids": (output[end_at + 1:eos_at] if end_at is not None and eos_at is not None
                       else output[end_at + 1:] if end_at is not None else []),
        "user_prompt": generation.user_prompt, "chat_prompt": generation.chat_prompt,
        "raw_output": backend.decode(output), "total_budget": total_budget,
        "generated_tokens": len(output), "generation_latency_sec": generation.latency_sec,
        "finish_reason": generation.finish_reason, "has_thinking_end": end_at is not None,
        "has_eos": eos_at is not None, "reasoning_start_index": start_at,
        "thinking_end_index": end_at, "eos_index": eos_at,
        "eligible_cut_range": [lower, upper] if start_at is not None and lower <= upper else None,
        "selected_cut_tokens": positions, "rejected_cut_tokens": rejected,
        "state_skip_reason": skip_reason, "sampling": {
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
        }, **identity,
    }
    states = []
    labels = []
    for cut in positions:
        prefix = output[:cut]
        if (backend.think_start_id not in prefix or backend.think_end_id in prefix or
                any(token in backend.eos_ids for token in prefix)):
            raise AssertionError("A selected cut is not unfinished thinking")
        state = {
            "problem_id": row["unique_id"], "prompt_ids": list(prompt),
            "prefix_ids": list(prefix), "total_budget": total_budget,
            "spent_tokens": cut, "remaining_tokens": total_budget - cut,
            "question": row["problem"],
            "enable_thinking": True, "has_thinking_end": False,
            "finish_reason": "length", "dataset_index": dataset_index,
            "source_trajectory_id": trajectory_id, "source_seed": seed,
            "cut_token_count": cut, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, **identity,
        }
        state["state_id"] = _digest(state)
        states.append(state)
        labels.append({"state_id": state["state_id"],
                       "problem_id": row["unique_id"], "dataset_index": dataset_index,
                       "gold_answer": str(row["answer"]).strip()})
    return trajectory, states, labels
