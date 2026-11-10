"""Pinned MATH-500 test input, independent of any action repository."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

DATA_PATH = Path(__file__).resolve().parents[1] / "math500_test.jsonl"
DATA_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
DATA_SOURCE = ("https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/"
               "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl")


def download_math500(path: Path = DATA_PATH) -> Path:
    """Fetch the pinned source and publish it only after SHA-256 verification."""
    path = Path(path)
    if path.exists():
        raw = path.read_bytes()
    else:
        with urlopen(DATA_SOURCE, timeout=60) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("MATH-500 download differs from the pinned test split")
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_math500(path: Path = DATA_PATH) -> list[dict]:
    if Path(path) == DATA_PATH and not Path(path).exists():
        download_math500(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("MATH-500 input differs from the pinned test split")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 500 or any(not all(k in row for k in
                                   ("problem", "answer", "unique_id")) for row in rows):
        raise ValueError("MATH-500 input does not have the expected 500 rows")
    return rows


if __name__ == "__main__":
    print(download_math500())
