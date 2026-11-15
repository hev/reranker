"""Self-hosted open-weight decision models with `transformers`: `model="local:<hf-id>"`.

    pip install "hev-rerank[local]"
    rerank(query, docs, model="local:perplexity-ai/pplx-decider-flash-v1.3", style="batch")

Runs on CUDA when available, else CPU. The request is the System One shape the hosted Perplexity and Clef providers
send (same prompts, `prompt_decisions.yaml`, and the same `decision_request` / `read_decision`), answered in-process
by the checkpoint's own reference inference code instead of over HTTPS. Port of hev/rerank-bench decide_adapters.py:
pinned revisions, the device sync around each timed call, and the shapes mapped onto the model's own `decide()`.

Supported checkpoints, each run through the inference code its repo ships (both Apache-2.0):
- Perplexity deciders (`perplexity-ai/pplx-decider-flash-v1.3` 4B, `perplexity-ai/pplx-decider-v1.3` 27B, later
  versions): `source/src/pplx_decider`.
- Cloudflare Clef (`Cloudflare/clef-flash` 9B, `Cloudflare/clef` 27B): `joint_schema_model.py` (`systemone`).
Other repos raise ValueError.

Turing GPUs (T4, free Colab) have no bfloat16, so on a GPU without native bf16 the weights load in float16;
`load(..., dtype=...)` overrides it. `torch`, `transformers` and `huggingface_hub` are imported only here, on the
first local call, so the hosted install never needs them.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from ._common import ProviderError, decision_request, read_decision

PROMPT = Path(__file__).with_name("prompt_decisions.yaml")

DEFAULT_MODEL = "perplexity-ai/pplx-decider-flash-v1.3"
# Pinned Hugging Face revisions (the commit the notebooks and the T4 run used). Unlisted repos load their main branch.
REVISIONS = {
    "perplexity-ai/pplx-decider-flash-v1.3": "d06fb3f2d97e0a52325a0d463c450cf0338f44fe",
}
# Only what decisions need: skips the 0.5 GB Locate (span extraction) adapter.
ALLOW_PATTERNS = ["*.json", "*.jinja", "*.safetensors", "*.py", "LICENSE", "NOTICE", "source/**"]
IGNORE_PATTERNS = ["locate-span/*"]


class LocalModel:
    """A loaded checkpoint and the mode its batch questions run in. Pass one as `rerank(..., client=...)`
    to choose device, dtype or mode yourself; by default `rerank` loads one per model id and keeps it."""

    def __init__(self, model_id: str, decider: Any, device: str, dtype: Any, mode: str):
        self.model_id, self.decider, self.device, self.dtype, self.mode = model_id, decider, device, dtype, mode
        self.calls: list[dict] = []      # {"latency_ms", "n_requests", "input_tokens"} per decide() call

    def decide(self, requests: list[dict]) -> list[dict]:
        torch = sys.modules["torch"]
        sync = torch.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        sync()
        t = time.perf_counter()
        out = self.decider.decide(requests)
        sync()
        ms = (time.perf_counter() - t) * 1000
        self.calls.append({"latency_ms": ms, "n_requests": len(requests),
                           "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in out)})
        return out

    def __repr__(self):
        return f"LocalModel({self.model_id!r}, device={self.device!r}, dtype={self.dtype}, mode={self.mode!r})"


_LOADED: dict[tuple, LocalModel] = {}
_LOCK = threading.Lock()


def _require_local_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        raise ImportError(f"model='local:...' needs the local extra: pip install 'hev-rerank[local]' ({e})") from None


def _import_package(checkpoint: Path, name: str):
    """Import the checkpoint's bundled package under a name unique to the checkpoint, so two checkpoints with
    different versions of the same inference code can live in one process."""
    src = checkpoint / "source" / "src" / name
    if not (src / "__init__.py").exists():
        return None
    alias = f"_hev_rerank_{name}_{abs(hash(str(checkpoint))):x}"
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, src / "__init__.py", submodule_search_locations=[str(src)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def default_dtype(device: str):
    import torch
    if not device.startswith("cuda"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def load(model_id: str = DEFAULT_MODEL, *, device: str | None = None, dtype: Any = None, mode: str = "isolated",
         revision: str | None = None, **kwargs: Any) -> LocalModel:
    """Download (once, into the Hugging Face cache) and load a decision checkpoint.

    device: "cuda", "cpu", ... (default: cuda if available). dtype: torch dtype (default: bf16 where the GPU has it,
    else fp16; fp32 on CPU).
    Perplexity deciders: mode="isolated" (default) reads a batch's shared state once and answers every question
    from it, the request layout the decider was trained on; "separate" runs one sequence per question, the
    checkpoint's evaluated computation, at ~one full state per document in batch shape. Extra keyword arguments go to
    `Decider.from_pretrained` (e.g. `batch_tokens`).
    Clef: extra keyword arguments go to the backbone's `from_pretrained` (e.g. `quantization_config=` for 4-bit).
    """
    _require_local_deps()
    import torch
    from huggingface_hub import snapshot_download

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or default_dtype(device)
    path = Path(snapshot_download(model_id, revision=revision or REVISIONS.get(model_id),
                                  allow_patterns=ALLOW_PATTERNS, ignore_patterns=IGNORE_PATTERNS))
    pkg = _import_package(path, "pplx_decider")
    if pkg is not None:
        return LocalModel(model_id, _load_pplx(pkg, path, device, dtype, mode, kwargs), device, dtype, mode)
    if (path / "joint_schema_model.py").exists():
        return LocalModel(model_id, _load_clef(model_id, path, device, dtype, kwargs), device, dtype, "joint")
    raise ValueError(f"local:{model_id}: no supported inference code in the checkpoint. hev-rerank runs checkpoints that "
                     "ship source/src/pplx_decider (perplexity-ai/pplx-decider v1.3+) or joint_schema_model.py "
                     "(Cloudflare/clef, Cloudflare/clef-flash).")


def _load_pplx(pkg, path: Path, device: str, dtype, mode: str, kwargs: dict):
    model_mod = sys.modules[pkg.__name__ + ".model"]
    orig = model_mod._backbone
    # The reference loader picks bf16 on any CUDA device; load the backbone in the dtype we chose instead (fp16 on a
    # T4) and cast the 255-row readout head to match.
    model_mod._backbone = lambda checkpoint, config, _dtype: orig(checkpoint, config, dtype)
    try:
        decider = pkg.Decider.from_pretrained(path, device=device, mode=mode, locate=False, **kwargs)
    finally:
        model_mod._backbone = orig
    decider.readout.to(dtype)
    return decider


class _ClefDecider:
    """Cloudflare's reference `systemone(model, processor, request)`, one request at a time, behind decide()."""

    def __init__(self, module, model, processor, name: str):
        self.module, self.model, self.processor, self.name = module, model, processor, name

    def decide(self, requests: list[dict]) -> list[dict]:
        return [self.module.systemone(self.model, self.processor, {"model": self.name, **r}) for r in requests]


def _load_clef(model_id: str, path: Path, device: str, dtype, kwargs: dict):
    alias = f"_hev_rerank_clef_{abs(hash(str(path))):x}"
    mod = sys.modules.get(alias)
    if mod is None:
        spec = importlib.util.spec_from_file_location(alias, path / "joint_schema_model.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    model, processor = mod.load_release_model(path, device=device, dtype=dtype, **kwargs)
    return _ClefDecider(mod, model, processor, model_id.rsplit("/", 1)[-1])


def connect(model: str = DEFAULT_MODEL) -> LocalModel:
    """The cached LocalModel for `model` (loaded on first use with the defaults of `load`)."""
    with _LOCK:
        if model not in _LOADED:
            _LOADED[model] = load(model)
        return _LOADED[model]


def score(client: LocalModel, model: str, query, docs, style, p) -> list[float]:
    """One request. Batch that runs out of GPU memory is split in half and retried (batch scores are absolute
    probabilities, so halves stay comparable; rerank-bench's Strands adapter splits on its context window the same
    way). Choice can't be split: its probabilities are relative to the list."""
    state, questions = decision_request(query, docs, style, p)
    try:
        return _answers(client, [{"state": state, "questions": questions}], [(len(docs), style)])[0]
    except Exception as e:
        if not _is_oom(e):
            raise
        _free(client)
        if style != "batch" or len(docs) == 1:
            raise ProviderError(f"local: out of GPU memory scoring {len(docs)} documents in style={style!r} on "
                                f"{getattr(client, 'device', 'the GPU')}; use a shorter list, shorter documents or style='batch'/'pair'") from None
    half = len(docs) // 2
    return score(client, model, query, docs[:half], style, p) + score(client, model, query, docs[half:], style, p)


def _is_oom(e: BaseException) -> bool:
    torch = sys.modules.get("torch")
    return bool(torch) and isinstance(e, torch.OutOfMemoryError)


def _free(client):
    torch = sys.modules.get("torch")
    if torch is not None and str(getattr(client, "device", "")).startswith("cuda"):
        torch.cuda.empty_cache()


def score_pairs(client: LocalModel, model: str, query, docs: Sequence[Any], p) -> list[float]:
    """Pair shape: one request per document, as the hosted providers send them, handed to the model in one
    `decide()` list so the GPU pads them into batches instead of running 30 forward passes."""
    reqs = []
    for d in docs:
        state, questions = decision_request(query, [d], "pair", p)
        reqs.append({"state": state, "questions": questions})
    return [s[0] for s in _answers(client, reqs, [(1, "pair")] * len(docs))]


def _answers(client, reqs, shapes) -> list[list[float]]:
    try:
        out = client.decide(reqs)
    except (KeyError, TypeError) as e:
        raise ProviderError(f"local: {type(e).__name__}: {e}") from None
    return [read_decision(r.get("answers", {}), n, style) for r, (n, style) in zip(out, shapes)]
