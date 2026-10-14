"""Decision requests (Cloudflare Clef's ``/v1/systemone``) on the 27B: a runtime adapter, every row's final state, a head.

A decision is a prompt that only prefills: its adapter's terms on (``adapters.applied``), no drafter, no prefix cache,
no sampled token. Rank 0 keeps every row's final normed state; Clef's joint schema head (fp32, rank 0) reads them with
the bf16 output-embedding rows of the option tokens and gives one probability per option. Chat never applies the
adapter, so its bits do not change.

An adapter directory (``TF_ADAPTERS=name=dir``) holds the factors (``adapters.load``), ``joint_head.safetensors`` with
``joint_head_config.json``, Cloudflare's ``joint_schema_model.py`` (the encoder and head, Apache-2.0) and
``lm_head_rows.safetensors`` (the base model's bf16 output embedding).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

MAX_LENGTH = 16384        # Clef's release default: a decision's prompt (state truncated to fit)


def configured() -> dict[str, Path]:
    """``TF_ADAPTERS``: ``name=dir[,name=dir]`` -> {name: dir}, sorted so every rank indexes them alike."""

    raw = os.environ.get("TF_ADAPTERS", "").strip()
    out = {}
    for part in filter(None, (p.strip() for p in raw.split(","))):
        name, _, where = part.partition("=")
        if not name or not where:
            raise ValueError(f"TF_ADAPTERS: name=dir, not {part!r}")
        out[name.strip()] = Path(where.strip())
    return dict(sorted(out.items()))


class Decision:
    """A decision prompt in the scheduler: the fields it reads off a stream, and every row's final state at the end."""

    background = False
    done = False
    preempted = False
    waiting = False
    vision = None
    constraint = None
    error: Exception | None = None

    def __init__(self, prompt: list[int], adapter: str, on_hidden=None):
        self.prompt, self.adapter = list(prompt), adapter
        self.on_hidden = on_hidden                    # run on the scheduler's thread with every row's final state
        self.result = None
        self.sid = -1
        self.st = None
        self.rows: list[torch.Tensor] = []
        self.hidden: torch.Tensor | None = None
        self.prefill_s = 0.0
        self.queued = time.perf_counter()

    @property
    def count(self) -> int:
        return 0

    def stats(self) -> dict:
        return {"hidden": self.hidden, "result": self.result, "prefill_s": self.prefill_s, "tokens": len(self.prompt),
                "seconds": time.perf_counter() - self.queued}


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(f"tensorfold_clef_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # its dataclasses resolve their module while it executes
    spec.loader.exec_module(mod)
    return mod


class _Tokens:
    """The ``tokenizer(text, add_special_tokens=False).input_ids`` the encoder calls, over a ``tokenizers`` file."""

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __init__(self, tok):
        self.tok = tok
        self.pad_token_id = tok.token_to_id("<|endoftext|>") or 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        return self._Out([int(t) for t in self.tok.encode(text, add_special_tokens=add_special_tokens).ids])


class _MediaIds:
    """The processor ``encode_record`` calls for a record's images: the expanded media block's ids, already prepared."""

    def __init__(self, ids):
        self.ids = list(ids)

    def __call__(self, text=None, images=None, videos=None, return_tensors=None, **_):
        return {"input_ids": torch.tensor([self.ids])}


class _Lexical:
    """The output-embedding rows the head averages per option, read from the memory-mapped bf16 table (no copy held)."""

    def __init__(self, path: Path, device):
        import numpy as np

        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        name, info = next((k, v) for k, v in header.items() if k != "__metadata__")
        if info["dtype"] != "BF16" or len(info["shape"]) != 2:
            raise ValueError(f"{path}: {name} must be a 2-D bf16 table, not {info['dtype']} {info['shape']}")
        a, b = info["data_offsets"]
        self.table = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + a, shape=tuple(info["shape"]))
        self.device = device

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        import numpy as np

        rows = np.ascontiguousarray(self.table[ids.to("cpu", torch.long).numpy()])
        return torch.from_numpy(rows.view(np.int16)).view(torch.bfloat16).to(self.device, torch.float32)


class Head:
    """Clef's encoder and joint schema head for one adapter, on rank 0 in fp32."""

    def __init__(self, adapter_dir: Path, tokenizer, device):
        from safetensors.torch import load_file

        adapter_dir = Path(adapter_dir)
        self.jsm = _module(adapter_dir / "joint_schema_model.py")
        cfg = json.loads((adapter_dir / "joint_head_config.json").read_text())
        head = self.jsm.JointSchemaHead(**cfg)
        head.load_state_dict(load_file(str(adapter_dir / "joint_head.safetensors")), strict=True)
        self.head = head.to(device=device, dtype=torch.float32).eval()
        self.lexical = _Lexical(adapter_dir / "lm_head_rows.safetensors", device)
        self.tok = _Tokens(tokenizer)
        self.device = device

    def encode(self, request: dict, truncate: bool = True, vision=None, limits=None):
        """The record's prompt, and with images the prepared vision prompt (None for text); ``truncate`` False refuses
        a record whose state would be cut (oMLX's 413). Images: ``vision`` is the engine's ``QwenCudaVision``."""

        processor, media = None, None
        if request.get("images"):
            if vision is None:
                raise ValueError("image decisions need an engine started with --vision")
            media = self._media(request["images"], vision, limits)
            processor = _MediaIds(media.token_ids)
        enc = self.jsm.encode_record(self.tok, request, max_length=MAX_LENGTH, processor=processor)
        if not truncate:
            whole = self.jsm.encode_record(self.tok, request, max_length=1 << 30, processor=processor)
            if len(whole.input_ids) > len(enc.input_ids):
                raise OverflowError(f"the record needs {len(whole.input_ids)} tokens; at most {MAX_LENGTH} fit")
        if media is None:
            return enc, None
        from dataclasses import replace

        from tensorfold.vision.qwen_processing import media_positions

        # the images' rows sit after the system prefix: Qwen's rope index over the whole record, as the processor's
        positions, delta, spans, _ = media_positions(enc.input_ids, media.image_grid_thw, None, vision.frontend.config)
        positions.setflags(write=False)
        return enc, replace(media, token_ids=tuple(enc.input_ids), position_ids=positions, rope_delta=delta,
                            image_spans=spans)

    def _media(self, urls, vision, limits):
        """Clef's media block (one image marker each, then a newline), expanded and preprocessed as chat images are."""

        from tensorfold.vision.images import DEFAULT_LIMITS, ImageSource, load_images

        if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
            raise ValueError("images must be a list of data URIs (or URLs where the server allows them)")
        images = load_images([ImageSource(u) for u in urls], limits=limits or DEFAULT_LIMITS,
                             allow_urls=bool(getattr(vision, "allow_urls", False)))
        return vision.prepare(self.jsm.IMAGE_PLACEHOLDER * len(images) + "\n", images)

    @torch.no_grad()
    def answer(self, request: dict, enc, hidden: torch.Tensor) -> dict:
        n = len(enc.input_ids)
        ids = torch.tensor([list(enc.input_ids)], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        logits = self.head(hidden[:n].to(self.device, torch.float32)[None], ids, mask, [enc], self.lexical)[0]
        questions = request["questions"]
        answers = {
            q.question_id: self.jsm.systemone_answer(
                questions[q.question_id], dict(zip(q.option_ids, lg.float().softmax(-1).tolist())))
            for q, lg in zip(enc.questions, logits)
        }
        reply = {"model": request["model"], "answers": answers, "usage": {"input_tokens": n, "output_tokens": 0}}
        if request.get("raw"):                 # a TensorFold extension: unrounded option probabilities, encoder order
            reply["raw"] = {q.question_id: {"type": ["noul", "choice", "score"][q.question_type],
                                            "options": list(q.option_ids), "probs": lg.float().softmax(-1).tolist()}
                            for q, lg in zip(enc.questions, logits)}
        return reply

    def check(self, request: dict) -> None:
        """Cloudflare's request checks (``systemone``): model and state, at least one typed question with criteria."""

        questions = request.get("questions")
        if not isinstance(request.get("model"), str) or "state" not in request:
            raise ValueError("model and state are required")
        if request.get("state") is None:
            raise ValueError("state must not be null")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        for qid, q in questions.items():
            if not isinstance(q, dict) or q.get("type") not in self.jsm.QUESTION_TYPES:
                raise ValueError(f"{qid}: type must be noul, choice, or score")
            if q["type"] != "noul" and not q.get("criteria"):
                raise ValueError(f"{qid}: criteria must not be empty")
            if q["type"] == "choice" and not isinstance(q["criteria"], dict):
                raise ValueError(f"{qid}: choice criteria map option ids to descriptions")
            if q["type"] == "score" and not isinstance(q["criteria"], list):
                raise ValueError(f"{qid}: score criteria are a list of level descriptions")
        if request.get("videos"):
            raise ValueError("video input is not supported")
