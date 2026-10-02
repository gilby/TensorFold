# Gemma 4 12B unified (`gemma4_unified`)

Measured on an M5 Ultra (256 GB) with MLX 0.32.3, mlx-lm 0.31.3, serving `gemma-4-12B-it-oQ8e` with the
`gemma-4-12B-it-qat-assistant-4bit` MTP assistant. Packages: `src/tensorfold/families/gemma4_unified/` and
`src/tensorfold/kernels/gemma/dense/v1/`. Text only: the unified checkpoint's patch embedder and audio
projection are dropped at load.

## The model

- `Gemma4UnifiedForConditionalGeneration`; its text model (`gemma4_unified_text`) is Gemma 4's dense decoder:
  48 layers, 40 sliding (window 1,024; 16 query heads over 8 KV heads, head dim 256) and 8 full attention
  (16 query heads over 1 KV head, head dim 512, keys reused as values, proportional RoPE). Hidden 3,840,
  GeGLU MLP width 15,360, per-layer scalar, vocabulary 262,144 tied and soft-capped at 30.
- The 12B has no per-layer inputs and no shared-KV layers (`check` refuses checkpoints that have them, and
  MoE ones, which the `gemma4` family serves).
- mlx-lm's `gemma4` wrapper loads it (both the oQ layout and the mlx-vlm layout sanitize to `gemma4_text`).

## Kernels

Prompts run mlx-lm's forward on the planned chunks. Decode rows run Gemma 4's v1 attention, head norm/RoPE
and KV caches with a dense tail (`kernels/gemma/dense/v1/glue.py`) and row-exact affine projections at any
oQ width in groups of 64 (4-bit also 32): the M5 lane matmul (`--lane-kernels auto` on tensor-unit GPUs) or
`row_matmul`'s simd kernels (`--lane-kernels off`, any Mac). Stacked q|k|v and gate|up split into runs of one
width when oQ boosts one member. The lane matmul keeps tiled copies of the decode weights beside mlx-lm's,
so resident weights are about twice the checkpoint (23.8 GiB for oQ8e).

## Drafts

`--drafter` takes a `gemma4_unified_assistant` (MTP) checkpoint: four layers that attend over the target's
last sliding and last full layer's keys from the last kept row's position, fed the target's scaled embedding
of the pending token next to the final-normed hidden row, then their own projected output. Up to 6 drafts a
round (the depth learner picks); shared rounds draft every stream in one assistant forward a step. The
assistant runs MLX's stock kernels: drafts only propose. Ordered (centroid) embeddings are refused.

## Checkpoint

oQe, groups of 64: `tools/oq_tf_friendly_gemma12.py` (oMLX's python) makes `gemma-4-12B-it-oQ8e` (12 GB,
uniform 8-bit affine g64, imatrix) from `google/gemma-4-12B-it`; level 6 gives an oQ6e (9.4 GB, 6-bit with
five 8-bit boosts) the family also reads.

## Results (M5 Ultra, 2026-10-02, lane matmul, other inference processes idle during each cell)

| | greedy | temperature 1 |
| --- | ---: | ---: |
| single stream, 512 tokens, median of 4 prompts | 131.7 tok/s | 94.0 tok/s |
| single stream, `"draft": false` | 52.0 | 51.1 |
| N=2 aggregate | 229.6 | 193.6 |
| N=4 aggregate | 238.7 | 220.8 |
| prefill, ~5,240-token prompt with a fresh nonce | 2,870-3,230 tok/s | |

Drafted replies equal `"draft": false` (3 prompts, temperature 0) and N=2/N=4 concurrent replies equal serial
ones (4 prompts, greedy and seeded sampling), by token hash. Exact windows 16 rows; shared forwards up to 64.
