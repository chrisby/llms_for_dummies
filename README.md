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

## Why this Repo?
To be a good ML scientist/engineer/practitioner, you need to understand the system. To be exceptional, you need to develop an intuition for its components. This repo tries to help you develop that intuition for different parts of the modern LLM stack — going deep where necessary, staying high-level when possible. Importantly, I try to provide tools like visualizations or a GPU simulator to make things concrete. Everything is easy to hack, and since notebooks are first-class citizens, everything runs on your machine — no GPUs required.

## Notebooks

### Attention Mechanisms

| Notebook | What You'll Learn |
|----------|-------------------|
| [Standard Attention](attention/standard_attention.ipynb) | Why the $N \times N$ attention matrix is a memory bottleneck, and how tiling strategies affect memory consumption |
| [FlashAttention](attention/flash_attention.ipynb) | How online softmax helps reducing traffic by ~3x |

### Planned Topics

Happy to consider any other topics you're interested in. Just raise an issue!

| Topic | Status |
|-------|--------|
| Positional Encoding (Sinusoidal, RoPE) | Coming soon |

## Getting Started

```bash
pip install -e .
jupyter notebook
```

## License

Apache 2.0
