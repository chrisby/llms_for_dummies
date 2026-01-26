# LLMs for Dummies

```
 __  __           _                  _     _     __  __
|  \/  | ___   __| | ___ _ __ _ __  | |   | |   |  \/  |___
| |\/| |/ _ \ / _` |/ _ \ '__| '_ \ | |   | |   | |\/| / __|
| |  | | (_) | (_| |  __/ |  | | | || |___| |___| |  | \__ \
|_|  |_|\___/ \__,_|\___|_|  |_| |_||_____|_____|_|  |_|___/

  __               ____                            _
 / _| ___  _ __   |  _ \ _   _ _ __ ___  _ __ ___ (_) ___  ___
| |_ / _ \| '__|  | | | | | | | '_ ` _ \| '_ ` _ \| |/ _ \/ __|
|  _| (_) | |     | |_| | |_| | | | | | | | | | | | |  __/\__ \
|_|  \___/|_|     |____/ \__,_|_| |_| |_|_| |_| |_|_|\___||___/
```

A hands-on guide to understanding how Large Language Models work under the hood. No hand-waving, no black boxes — just clear explanations with runnable code and a GPU memory simulator that makes the invisible visible.

## What's This About?

Most LLM explanations stop at "attention is all you need" and wave their hands at the rest. This repo takes a **bottom-up approach** — we build a simulated GPU memory hierarchy and use it to actually *see* why certain algorithms exist:

- **Interactive simulations** — watch SRAM fill up and overflow as sequence length grows
- **Step-by-step memory traces** — see exactly which tensors are in HBM vs SRAM at each step
- **Working code you can break** — change tile sizes, sequence lengths, and head dimensions to build intuition

## Notebooks

### Attention Mechanisms

| Notebook | What You'll Learn |
|----------|-------------------|
| [Standard Attention](attention/standard_attention.ipynb) | Why the N×N attention matrix is a memory bottleneck, and how tiling strategies (no tiling → 1D → 2D) each solve one problem while creating another |
| [FlashAttention](attention/flash_attention.ipynb) | How online softmax lets you do 2D tiling without ever writing the N×N matrices S and P to HBM — reducing traffic by ~3x |

### More Topics

| Topic | Status |
|-------|--------|
| Positional Encoding (Sinusoidal, RoPE) | Coming soon |

## Getting Started

```bash
pip install -e .
jupyter notebook
```

Start with [standard_attention.ipynb](attention/standard_attention.ipynb) — the FlashAttention notebook builds on it.

## What You'll Walk Away With

- **Why attention is O(N²)** and what that actually means in terms of memory, not just compute
- **The tiling tradeoff**: 1D tiling gives O(Nd) HBM traffic but can't scale; 2D tiling scales but has O(N²) traffic — and why FlashAttention gets the best of both
- **How GPU memory hierarchies work** — HBM vs SRAM, why the compute/memory ratio matters, and what "memory-bound" actually means
- **How online softmax works** — maintaining running max and sum statistics to compute softmax incrementally without needing full rows

## How It Works

The notebooks use a simulated A100 GPU (`gpu_sim.py`) that tracks:
- **SRAM occupancy** — with visual bars that show when you overflow
- **HBM traffic** — reads and writes, so you can see the O(N) vs O(N²) difference
- **Compute vs memory time** — whether the GPU is actually doing useful work or just waiting on memory

There are no real GPU kernels here. The `Tensor` API mimics PyTorch-style operations but only tracks shapes and memory movements. The point is understanding the *algorithm*, not the hardware.

## License

Apache 2.0
