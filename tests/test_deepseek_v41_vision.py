"""DeepSeek-V4.1-Flash image geometry and prompt expansion (CPU; the tower's parity is a tool against oMLX)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from tensorfold.vision.deepseek_v41_processing import (IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START,  # noqa: E402
                                                        DeepSeekV41ImageProcessor, VisionGeometry, image_patches,
                                                        num_image_tokens, plan_image_grid)

PH = "<｜deepseek_image｜>"
VISION = {"patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 1024, "min_pixels": 295936,
          "max_wh_ratio": None, "hidden_size": 1024, "intermediate_size": 2816, "num_attention_heads": 16}
G = VisionGeometry.from_config(VISION)


@pytest.mark.parametrize("size", [(900, 600), (300, 1400), (4000, 3000), (64, 64), (5000, 40), (40, 5000), (544, 544)])
def test_every_image_fits_its_token_budget_and_patch_grid(size):
    n_h, n_w, height, width = plan_image_grid(size[0], size[1], G, G.max_n_token)
    assert num_image_tokens(n_h, n_w) <= 1024
    assert height % 14 == 0 and width % 14 == 0
    assert (n_h, n_w) == (math.ceil(height // 14 / 3), math.ceil(width // 14 / 3))


def test_patches_and_types_follow_the_reference_layout():
    image = Image.new("RGB", (900, 600), (127, 127, 127))
    patches, n_vit_h, n_vit_w, types = image_patches(image, G, G.max_n_token)
    assert patches.shape == (n_vit_h * n_vit_w, 3, 14, 14)
    assert np.allclose(patches, 127 / 255 * 2 - 1)
    n_llm_h, n_llm_w = math.ceil(n_vit_h / 3), math.ceil(n_vit_w / 3)
    assert types == (IMAGE_START,) + ((IMAGE,) * n_llm_w + (IMAGE_NEW_LINE,)) * n_llm_h + (IMAGE_END,)


class _Tokenizer:
    """Characters as ids; the placeholder as one id."""

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            if text.startswith(PH, i):
                out.append(129264)
                i += len(PH)
            else:
                out.append(ord(text[i]) % 1000)
                i += 1
        return out


class _Image:
    def __init__(self, size, h):
        self.size, self.content_hash, self.detail = size, h, "auto"

    def to_pil(self):
        return Image.new("RGB", self.size, (10, 200, 30))


def test_prepare_expands_each_placeholder_into_its_span():
    proc = DeepSeekV41ImageProcessor({"image_token_id": 129264, "vision_config": VISION}, _Tokenizer())
    prepared = proc.prepare(f"ab{PH}cd{PH}e", [_Image((900, 600), "x"), _Image((300, 1400), "y")])
    assert len(prepared.image_spans) == 2 and prepared.image_hashes == ("x", "y")
    for (begin, end), kinds in zip(prepared.image_spans, prepared.types):
        assert end - begin == len(kinds)
        assert set(prepared.token_ids[begin:end]) == {129264}
    assert prepared.token_ids.count(129264) == sum(len(k) for k in prepared.types)
    with pytest.raises(ValueError):
        proc.prepare("no images here", [_Image((900, 600), "x")])


def test_image_parts_render_as_placeholders_joined_like_the_encoder():
    from tensorfold.families.deepseek_v41.prompts import render

    out = render([{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image"},
                                               {"type": "text", "text": "what?"}]}])
    assert f"<｜User｜>look\n\n{PH}\n\nwhat?<｜Assistant｜>" in out
    with pytest.raises(ValueError):
        render([{"role": "user", "content": f"smuggled {PH}"}])
