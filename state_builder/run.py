"""Generate full Single token traces, then cut each trace into shared states."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import tempfile

from .backend import HFSingleBackend
from .cuts import build_records
from .dataset import DATA_PATH, DATA_SHA256, DATA_SOURCE, load_math500


METHOD_BACKENDS = {"single": HFSingleBackend}
DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompt.txt"


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def _hash(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _write_final(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Published output differs from records: {path}")
        return
    _write_atomic(path, content)


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":")) + "\n").encode()
                    for row in rows)


def _versions() -> dict:
    values = {}
    for name in ("torch", "transformers", "accelerate"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def _load_record(path: Path, expected_config_hash: str, index: int,
                 backend_identity: dict) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    checksum = record.pop("record_sha256", None)
    if checksum != _hash(record):
        raise ValueError(f"Corrupted checkpoint record: {path}")
    if (record.get("config_sha256") != expected_config_hash or
            record.get("dataset_index") != index or
            record.get("backend_identity") != backend_identity):
        raise ValueError(f"Checkpoint identity/configuration mismatch: {path}")
    return record


def main(argv=None, *, backend_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory, or existing one with --resume")
    parser.add_argument("--dataset", type=Path, default=DATA_PATH)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--method", choices=sorted(METHOD_BACKENDS), default="single")
    parser.add_argument("--model-name", default="Qwen/Qwen3-14B")
    parser.add_argument("--revision", help="Pin a model checkpoint/commit for shared runs")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-problems", type=int, default=3)
    parser.add_argument("--total-budget", type=int, default=2048)
    cuts = parser.add_mutually_exclusive_group()
    cuts.add_argument("--num-cut-points", type=int, help="Evenly spaced cuts per trajectory (default: 3)")
    cuts.add_argument("--cut-tokens", type=int, nargs="+",
                      help="Explicit generated-token counts, e.g. 256 512 1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if (args.start_index < 0 or args.num_problems < 1 or
            args.start_index + args.num_problems > 500 or args.total_budget < 3 or
            args.num_cut_points is not None and args.num_cut_points < 1 or
            args.cut_tokens is not None and
            (not args.cut_tokens or any(p < 1 for p in args.cut_tokens) or
             len(set(args.cut_tokens)) != len(args.cut_tokens)) or
            not 0 <= args.seed < 2**32 or
            not math.isfinite(args.temperature) or args.temperature <= 0 or
            not math.isfinite(args.top_p) or not 0 < args.top_p <= 1 or
            args.top_k < 0):
        parser.error("Invalid slice, budget, cut positions, seed, or sampling settings")
    if args.num_cut_points is None and args.cut_tokens is None:
        args.num_cut_points = 3
    try:
        rows = load_math500(args.dataset)
        prompt_bytes = args.prompt_file.read_bytes()
        prompt = prompt_bytes.decode("utf-8")
        if prompt.count("{question}") != 1:
            raise ValueError("Prompt template must contain {question} exactly once")
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    root = Path(__file__).resolve().parent
    code_files = ("backend.py", "cuts.py", "dataset.py", "run.py")
    config = {
        "method": args.method, "model_name": args.model_name,
        "requested_revision": args.revision,
        "start_index": args.start_index, "num_problems": args.num_problems,
        "total_budget": args.total_budget, "num_cut_points": args.num_cut_points,
        "cut_tokens": args.cut_tokens, "seed": args.seed,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "dataset_sha256": DATA_SHA256, "dataset_source": DATA_SOURCE,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in code_files},
        "package_versions": _versions(),
    }
    config_hash = _hash(config)
    config_path = args.output / "run_config.json"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            parser.error("Cannot resume: saved run_config.json differs or is missing")
    elif args.output.exists():
        parser.error("Output directory already exists; use a new name or --resume")

    if backend_factory is None:
        backend_factory = METHOD_BACKENDS[args.method]
    backend = backend_factory(args.model_name, revision=args.revision)
    if backend.method != args.method or backend.model_name != args.model_name:
        raise ValueError("Generation backend does not match requested method/model")
    identity = {"model_name": backend.model_name, "quantization": "none",
                "model_revision": backend.model_revision(),
                "tokenizer_fingerprint": backend.tokenizer_fingerprint(),
                "generation_signature": backend.generation_signature()}
    identity_path = args.output / "backend_identity.json"
    if args.resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise ValueError("Cannot resume with a different model/tokenizer/backend")
    else:
        args.output.mkdir(parents=True)
        _write_atomic(config_path, _json_bytes(config))
        _write_atomic(identity_path, _json_bytes(identity))

    selected = list(enumerate(rows[args.start_index:args.start_index + args.num_problems],
                              start=args.start_index))
    records = []
    for index, row in selected:
        record_path = args.output / "records" / f"{index:05d}.json"
        if record_path.exists():
            record = _load_record(record_path, config_hash, index, identity)
            if record["trajectory"]["problem_id"] != row["unique_id"]:
                raise ValueError(f"Dataset row changed for {record_path}")
        else:
            seed = (args.seed + index) % 2**32
            generation = backend.generate_trajectory(
                row["problem"], prompt, budget=args.total_budget, seed=seed,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            )
            trajectory, states, labels = build_records(
                backend=backend, generation=generation, dataset_index=index, row=row,
                total_budget=args.total_budget, seed=seed,
                num_cut_points=args.num_cut_points, cut_tokens=args.cut_tokens,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            )
            record = {"config_sha256": config_hash, "backend_identity": identity,
                      "dataset_index": index, "trajectory": trajectory,
                      "states": states, "labels": labels}
            _write_atomic(record_path, _json_bytes({**record, "record_sha256": _hash(record)}))
        records.append(record)
        print(f"{index}: {len(record['states'])} state(s), "
              f"{record['trajectory']['generated_tokens']} generated token(s)", flush=True)

    trajectories = [record["trajectory"] for record in records]
    states = [state for record in records for state in record["states"]]
    labels = [label for record in records for label in record["labels"]]
    if len({state["state_id"] for state in states}) != len(states):
        raise ValueError("Duplicate state_id in generated dataset")
    files = {"trajectories.jsonl": _jsonl(trajectories),
             "states.jsonl": _jsonl(states), "labels.jsonl": _jsonl(labels)}
    for name, content in files.items():
        _write_final(args.output / name, content)
    summary = {
        "complete": True, "method": args.method,
        "requested_problems": args.num_problems,
        "trajectories": len(trajectories), "shared_states": len(states),
        "states_per_problem": {str(r["dataset_index"]): len(r["states"]) for r in records},
        "skipped": {str(t["dataset_index"]): t["state_skip_reason"] for t in trajectories
                    if t["state_skip_reason"] is not None},
        "physical_collection_generated_tokens": sum(t["generated_tokens"] for t in trajectories),
        "physical_collection_latency_sec": sum(t["generation_latency_sec"] for t in trajectories),
        "note": "One full Single trajectory per problem; suffix generation is physical dataset cost, not historical state spent_tokens.",
    }
    manifest = {"config_sha256": config_hash, "config": config,
                "backend_identity": identity,
                "file_sha256": {name: hashlib.sha256(content).hexdigest()
                                for name, content in files.items()},
                "summary": summary}
    _write_final(args.output / "summary.json", _json_bytes(summary))
    _write_final(args.output / "manifest.json", _json_bytes(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
