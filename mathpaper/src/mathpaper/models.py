from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np

from .store import data_path

MODEL_REPOS = {
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
}


def model_precision() -> str:
    precision = os.getenv("MATHPAPER_MODEL_PRECISION", "fp32")
    if precision not in {"fp32", "int8"}:
        raise ValueError("MATHPAPER_MODEL_PRECISION must be fp32 or int8.")
    return precision


def prepare_models(root: Path | None = None) -> dict:
    """Download pinned official ONNX exports and verify every file's SHA-256."""
    root = root or data_path()
    precision = model_precision()
    filename = "model-lock-int8.json" if precision == "int8" else "model-lock.json"
    directory = "models-int8" if precision == "int8" else "models"
    locked = json.loads(files("mathpaper").joinpath(filename).read_text())
    provenance = {}
    for role, repo in MODEL_REPOS.items():
        target = root / directory / role
        target.mkdir(parents=True, exist_ok=True)
        entry = locked[role]
        if entry["repo"] != repo:
            raise ValueError("Model repository does not match model-lock.json.")
        revision = entry["revision"]
        hashes = {}
        for name, expected in entry["sha256"].items():
            local = target / ("model.onnx" if name.endswith(".onnx") else Path(name).name)
            actual = _file_hash(local) if local.exists() else None
            if actual != expected:
                request = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", headers={"User-Agent": "MathPaper/0.1"})
                temporary = local.with_suffix(local.suffix+".part")
                try:
                    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                        while piece := response.read(1024*1024):
                            out.write(piece)
                    if _file_hash(temporary) != expected:
                        raise ValueError(f"Checksum mismatch for {role}/{name}; the file was not installed.")
                    temporary.replace(local)
                finally:
                    temporary.unlink(missing_ok=True)
            hashes[name] = expected
        provenance[role] = {"repo": repo, "revision": revision, "sha256": hashes}
        (target / "provenance.json").write_text(json.dumps(provenance[role], indent=2))
    return provenance


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class NeuralModels:
    def __init__(self, root: Path | None = None):
        self.root = root or data_path()
        self.precision = model_precision()
        self.directory = self.root / ("models-int8" if self.precision == "int8" else "models")
        self._sessions = {}
        self._tokenizers = {}
        self._lock = threading.RLock()

    def ready(self) -> bool:
        return all((self.directory / role / "model.onnx").exists() and (self.directory / role / "tokenizer.json").exists() for role in MODEL_REPOS)

    def tokenizer(self, role="embedding"):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(self.directory / role / "tokenizer.json"))

    def _session(self, role):
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        import onnxruntime as ort
        ort.disable_telemetry_events()
        if role not in self._sessions:
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(os.getenv("MATHPAPER_ORT_THREADS", "2")))
            options.inter_op_num_threads = 1
            if self.precision == "int8":
                options.enable_cpu_mem_arena = False
                options.enable_mem_pattern = False
            self._sessions[role] = ort.InferenceSession(str(self.directory / role / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        return self._sessions[role]

    def signature(self, role):
        path = self.directory / role / "provenance.json"
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "unprepared"

    def _encode(self, role, texts, max_length, paired=False):
        tokenizer = self.tokenizer(role)
        tokenizer.enable_truncation(max_length=max_length, strategy="only_second" if paired else "longest_first")
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        encodings = tokenizer.encode_batch(texts)
        return {
            "input_ids": np.array([x.ids for x in encodings], dtype=np.int64),
            "attention_mask": np.array([x.attention_mask for x in encodings], dtype=np.int64),
            "token_type_ids": np.array([x.type_ids for x in encodings], dtype=np.int64),
        }

    def embed(self, texts: list[str], batch_size=None) -> np.ndarray:
        batch_size = max(1, min(32, int(batch_size or os.getenv("MATHPAPER_INFERENCE_BATCH_SIZE", "16"))))
        output = []
        with self._lock:
            session = self._session("embedding")
            for start in range(0, len(texts), batch_size):
                inputs = self._encode("embedding", texts[start:start+batch_size], 256)
                values = session.run(None, {x.name: inputs[x.name] for x in session.get_inputs()})[0]
                if values.ndim == 3:
                    mask = inputs["attention_mask"][..., None]
                    values = (values*mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
                output.append(values.astype(np.float32))
        return np.concatenate(output, axis=0) if output else np.empty((0, 384), dtype=np.float32)

    def rerank(self, question: str, texts: list[str], batch_size=None, max_length=512) -> list[float]:
        batch_size = max(1, min(32, int(batch_size or os.getenv("MATHPAPER_INFERENCE_BATCH_SIZE", "16"))))
        if max_length > 512 or max_length < 64:
            raise ValueError("MiniLM pair length must be between 64 and 512 tokens.")
        tokenizer = self.tokenizer("reranker")
        if len(tokenizer.encode(question, add_special_tokens=False).ids)+3 >= max_length:
            raise ValueError("The question exceeds the reranker's input budget; shorten it.")
        scores = []
        with self._lock:
            session = self._session("reranker")
            for start in range(0, len(texts), batch_size):
                pairs = [(question, value) for value in texts[start:start+batch_size]]
                inputs = self._encode("reranker", pairs, max_length, paired=True)
                logits = session.run(None, {x.name: inputs[x.name] for x in session.get_inputs()})[0]
                scores.extend(float(v) for v in logits.reshape(-1))
        return scores
