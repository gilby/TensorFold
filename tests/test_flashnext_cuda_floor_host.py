"""Flash Next's CUDA startup floor, checked on machines with no NVIDIA GPU: an injected torch stands in for the device."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold.cuda import build
from tensorfold.families import qwen4_exp


def _fake_torch(monkeypatch, capability, name="Some GPU"):
    """A torch module that makes this machine look like it has a CUDA card of ``capability``."""

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda *a: capability,
                                get_device_name=lambda *a: name)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_the_family_refuses_a_card_below_the_floor_before_any_weight_loads(tmp_path, monkeypatch):
    """Red on the current code: today Flash Next starts on any CUDA card from 8.9 up, naming nothing it runs on."""

    _fake_torch(monkeypatch, (11, 0), "Orin-class card")
    built = []
    from tensorfold.families.qwen4_exp.cuda import engine as flashnext_engine
    with pytest.raises(ValueError, match=r"GB10, RTX 50 and RTX PRO 6000.*Orin-class card.*11\.0"):
        qwen4_exp.cuda_engine(tmp_path, no_drafts=True)
    assert not built                       # the refusal must come before the engine exists, let alone its weights


@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (9, 0), (11, 0)])
def test_a_card_below_sm_120_is_refused_by_name(monkeypatch, capability):
    _fake_torch(monkeypatch, capability, "Some GPU")
    with pytest.raises(ValueError, match=r"sm_120 and sm_121.*Some GPU"):
        build.refuse_small_gpu()


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_the_supported_cards_keep_serving(monkeypatch, capability):
    _fake_torch(monkeypatch, capability, "Some GPU")
    build.refuse_small_gpu()


def test_a_machine_without_cuda_changes_nothing(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False, get_device_capability=lambda *a: (12, 0),
                                get_device_name=lambda *a: "Fake GPU")
    monkeypatch.setitem(sys.modules, "torch", torch)
    build.refuse_small_gpu()


def test_a_named_capability_below_the_floor_is_admitted(monkeypatch):
    """``also`` admits exactly the capabilities it names (sm_70 for NVFP4 checkpoints); its neighbours stay refused."""

    _fake_torch(monkeypatch, (7, 0), "Tesla V100")
    build.refuse_small_gpu(also=(build.VOLTA,))
    _fake_torch(monkeypatch, (8, 6), "Some GPU")
    with pytest.raises(ValueError, match=r"sm_120 and sm_121.*Some GPU"):
        build.refuse_small_gpu(also=(build.VOLTA,))


def test_sm70_refuses_a_checkpoint_that_is_not_nvfp4(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 32}}')
    _fake_torch(monkeypatch, (7, 0), "Tesla V100")
    with pytest.raises(ValueError, match=r"Tesla V100.*7\.0"):
        qwen4_exp.cuda_engine(tmp_path, no_drafts=True)
