"""model="local:<hf-id>": the self-hosted provider, with a fake decider in place of the checkpoint.

The fake answers like the hosted fake server (conftest): relevance 0.9 for documents containing "relevant", else 0.1.
`test_load_*` build a tiny fake checkpoint (a bundled `pplx_decider` package) and need torch + transformers +
huggingface_hub (`pip install hev-rerank[local]`); they skip without them.
"""
import subprocess
import sys
import textwrap

import pytest

from hev_rerank import RerankResult, load_prompt, local, rerank
from hev_rerank.rerank import LOCAL_AUTO, MODELS, resolve_model

from conftest import decision_answers
from test_rerank import DOCS, Q

FLASH = "perplexity-ai/pplx-decider-flash-v1.3"


class FakeDecider:
    """Stands in for LocalModel: same decide(list[request]) -> list[response] contract."""

    def __init__(self, model_id):
        self.model_id, self.calls = model_id, []

    def decide(self, requests):
        self.calls.append(requests)
        return [{"answers": decision_answers(r), "usage": {"input_tokens": 10, "output_tokens": 0}} for r in requests]

    def requests(self):
        return [r for c in self.calls for r in c]


@pytest.fixture
def fake(monkeypatch):
    loaded = {}

    def connect(model=local.DEFAULT_MODEL):
        return loaded.setdefault(model, FakeDecider(model))

    monkeypatch.setattr(local, "connect", connect)
    return loaded


def shape(req):
    q = list(req["questions"].values())
    return "choice" if q[0]["type"] == "choice" else "pair" if "document" in req["state"] else "batch"


@pytest.mark.parametrize("style", ["pair", "batch", "choice", "auto"])
def test_local_every_style(fake, style):
    hits = rerank(Q, DOCS, model=f"local:{FLASH}", style=style)
    want = "choice" if style == "auto" else style
    dec = fake[FLASH]

    assert all(isinstance(h, RerankResult) for h in hits)
    assert sorted(h.index for h in hits) == [0, 1, 2, 3]
    assert {h.index for h in hits[:2]} == {1, 3}
    assert all(h.document is DOCS[h.index] for h in hits)
    if want == "choice":
        assert sum(h.score for h in hits) == pytest.approx(1.0)
    else:
        assert hits[0].score == pytest.approx(0.9) and hits[-1].score == pytest.approx(0.1)
    # One decide() call per query in every shape; pair hands the model one request per document.
    assert len(dec.calls) == 1
    assert len(dec.requests()) == (4 if want == "pair" else 1)
    assert all(shape(r) == want for r in dec.requests())
    assert "kept-but-not-sent" not in str(dec.requests())


def test_local_uses_hosted_prompts(fake):
    """Same request as hosted Perplexity / Clef: prompt_decisions.yaml through decision_request."""
    p = load_prompt(local.PROMPT)
    rerank(Q, DOCS[:2], model="local:", style="batch")
    rerank(Q, DOCS[:2], model="local:", style="pair")
    rerank(Q, DOCS[:2], model="local:", style="choice")
    b, pr, c = fake[FLASH].requests()[0], fake[FLASH].requests()[1], fake[FLASH].requests()[3]
    assert b["state"] == {"query": Q, "documents": {"D00": DOCS[0], "D01": DOCS[1]}}
    assert b["questions"]["D01"] == {"type": "noul", "instructions": p["question"].format(id="D01"), "criteria": p["criteria"]}
    assert pr["state"] == {"query": Q, "document": DOCS[0]}
    assert pr["questions"]["D00"]["instructions"] == p["pair_question"]
    assert c["questions"]["best"] == {"type": "choice", "instructions": p["choice_question"],
                                      "criteria": {"D00": "documents.D00", "D01": "documents.D01"}}


def test_local_auto_table():
    assert resolve_model(f"local:{FLASH}") == ("local", FLASH, "choice")
    assert resolve_model("local:") == ("local", local.DEFAULT_MODEL, LOCAL_AUTO[local.DEFAULT_MODEL])
    assert resolve_model("local:someone/unmeasured-decider") == ("local", "someone/unmeasured-decider", "batch")
    assert "local" not in MODELS                      # the hosted table is unchanged


def test_local_choice_threshold_raises(fake):
    with pytest.raises(ValueError, match="sum to one"):
        rerank(Q, DOCS, model="local:", threshold=0.1)        # auto -> choice on flash-v1.3
    assert fake == {}
    assert [h.index for h in rerank(Q, DOCS, model="local:", style="batch", threshold=0.5)] == [1, 3]


def test_local_batch_splits_and_choice_limits(fake):
    docs = [f"doc {i}" + (" relevant" if i % 7 == 0 else "") for i in range(70)]
    hits = rerank(Q, docs, model="local:", style="batch")
    assert sorted(len(r["questions"]) for r in fake[FLASH].requests()) == [10, 30, 30]
    assert {h.index for h in hits[:10]} == {i for i in range(70) if i % 7 == 0}
    with pytest.raises(ValueError, match="one call"):
        rerank(Q, docs, model="local:", style="choice")


def test_local_client_is_used(fake):
    mine = FakeDecider("mine")
    rerank(Q, DOCS, model="local:org/other", style="pair", client=mine)
    assert len(mine.requests()) == 4 and fake == {}


def test_hosted_import_does_not_pull_torch():
    code = "import sys, hev_rerank; hev_rerank.rerank; print(any(m in sys.modules for m in ('torch', 'transformers')))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


def test_missing_extra_message(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_torch(name, *a, **k):
        if name in ("torch", "transformers"):
            raise ImportError(f"No module named '{name}'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(ImportError, match=r"hev-rerank\[local\]"):
        local.load(FLASH)


# ---- load(): a tiny fake checkpoint with its own pplx_decider package ------------------------------

FAKE_MODEL_PY = '''
import torch

def _backbone(checkpoint, config, dtype):
    return torch.nn.Linear(4, 4, dtype=dtype)

class Decider:
    @classmethod
    def from_pretrained(cls, path, *, device="cuda", mode="separate", locate=None, **kw):
        d = cls()
        d.path, d.device, d.mode, d.locate, d.kw = path, device, mode, locate, kw
        d.backbone = _backbone(path, {}, torch.bfloat16 if device == "cuda" else torch.float32)
        d.readout = torch.nn.Linear(4, 255, bias=False, dtype=torch.bfloat16)
        return d

    def decide(self, requests):
        return [{"answers": {k: {"type": "noul", "noul": 0.5 + 0.1 * len(r["questions"])} for k in r["questions"]},
                 "usage": {"input_tokens": 7}} for r in requests]
'''


@pytest.fixture
def fake_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    pkg = tmp_path / "ckpt" / "source" / "src" / "pplx_decider"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .model import Decider\n")
    (pkg / "model.py").write_text(textwrap.dedent(FAKE_MODEL_PY))
    seen = {}

    def snapshot_download(repo, revision=None, **kw):
        seen.update(repo=repo, revision=revision, **kw)
        return str(tmp_path / "ckpt")

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    return seen


def test_load_fp16_override_and_pins(fake_checkpoint):
    import torch
    m = local.load(FLASH, device="cpu", dtype=torch.float16)
    assert fake_checkpoint["repo"] == FLASH and fake_checkpoint["revision"] == local.REVISIONS[FLASH]
    assert "locate-span/*" in fake_checkpoint["ignore_patterns"]
    assert m.decider.backbone.weight.dtype == torch.float16            # the reference loader asked for bf16/fp32
    assert m.decider.readout.weight.dtype == torch.float16
    assert m.decider.mode == "isolated" and m.decider.locate is False
    assert [round(s, 3) for s in local.score(m, FLASH, Q, ["a", "b"], "batch", load_prompt(local.PROMPT))] == [0.7, 0.7]
    assert m.calls and m.calls[0]["n_requests"] == 1 and m.calls[0]["input_tokens"] == 7
    # The original loader is restored after loading.
    mod = sys.modules[type(m.decider).__module__]
    assert mod._backbone(None, {}, torch.float32).weight.dtype == torch.float32


def test_load_defaults_cpu_fp32(fake_checkpoint):
    import torch
    m = local.load("org/unpinned", device="cpu", mode="separate", batch_tokens=1024)
    assert fake_checkpoint["revision"] is None
    assert m.dtype == torch.float32 and m.decider.mode == "separate" and m.decider.kw == {"batch_tokens": 1024}


def test_load_rejects_unknown_checkpoints(fake_checkpoint, tmp_path):
    import shutil
    shutil.rmtree(tmp_path / "ckpt" / "source")
    with pytest.raises(ValueError, match="pplx_decider"):
        local.load("org/not-a-decider", device="cpu")


def test_batch_oom_splits_in_half_choice_raises(fake, monkeypatch):
    import types

    class OOM(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(OutOfMemoryError=OOM))
    tight = FakeDecider("tight")
    real = tight.decide

    def decide(requests):                         # this "GPU" holds at most 8 batch questions per request
        if any(len(r["questions"]) > 8 for r in requests) or any(q["type"] == "choice" and len(q["criteria"]) > 8
                                                                 for r in requests for q in r["questions"].values()):
            raise OOM("CUDA out of memory")
        return real(requests)

    tight.decide = decide
    docs = [f"doc {i}" + (" relevant" if i % 7 == 0 else "") for i in range(30)]
    hits = rerank(Q, docs, model="local:", style="batch", client=tight)
    assert sorted(len(r["questions"]) for r in tight.requests()) == [7, 7, 8, 8]
    assert {h.index for h in hits[:5]} == {0, 7, 14, 21, 28} and hits[0].score == pytest.approx(0.9)
    from hev_rerank import ProviderError
    with pytest.raises(ProviderError, match="out of GPU memory.*choice"):
        rerank(Q, docs, model="local:", style="choice", client=tight)


def test_local_auto_clef():
    assert resolve_model("local:Cloudflare/clef-flash") == ("local", "Cloudflare/clef-flash", "pair")
    assert resolve_model("local:Cloudflare/clef") == ("local", "Cloudflare/clef", "pair")


FAKE_CLEF_PY = '''
import torch

def load_release_model(path, device="cuda", dtype=torch.bfloat16, **kw):
    return {"device": device, "dtype": dtype, "kw": kw}, "processor"

def systemone(model, processor, request, max_length=16384):
    answers = {k: {"type": "noul", "noul": 0.25} for k in request["questions"]}
    return {"model": request["model"], "answers": answers, "usage": {"input_tokens": 3, "output_tokens": 0}}
'''


def test_load_clef_backend(fake_checkpoint, tmp_path):
    import shutil
    import torch
    shutil.rmtree(tmp_path / "ckpt" / "source")
    (tmp_path / "ckpt" / "joint_schema_model.py").write_text(textwrap.dedent(FAKE_CLEF_PY))
    m = local.load("Cloudflare/clef-flash", device="cpu", quantization_config="4bit")
    assert m.decider.model == {"device": "cpu", "dtype": torch.float32, "kw": {"quantization_config": "4bit"}}
    hits = rerank(Q, DOCS, model="local:Cloudflare/clef-flash", client=m)     # auto -> pair, one request per doc
    assert [h.score for h in hits] == [0.25] * 4 and m.calls[-1]["n_requests"] == 4
