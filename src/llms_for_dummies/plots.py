"""Plotting utilities for attention notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .gpu_sim import GPUSpec


def plot_1d_tiling_sram_limit(gpu: GPUSpec, d: int):
    """Show how K+V grow with N until they fill SRAM, making 1D tiling impossible."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    bpf = gpu.bytes_per_float
    N_values = np.arange(64, 1200, 8)
    kv_kb = 2 * N_values * d * bpf / 1024
    sram_kb = gpu.sram_size * bpf / 1024
    N_cross = gpu.sram_size / (2 * d)

    fig, ax = plt.subplots(figsize=(8, 4))

    # K+V line
    ax.plot(N_values, kv_kb, color=sns.color_palette("deep")[3], linewidth=2.5,
            label='K + V  (2 · N · d)')

    # SRAM capacity
    ax.axhline(y=sram_kb, color='#555555', linestyle='--', linewidth=1.5,
               label=f'SRAM capacity ({sram_kb:.0f} KB)')

    # Shaded regions
    ax.fill_between(N_values, 0, np.minimum(kv_kb, sram_kb),
                    alpha=0.20, color=sns.color_palette("deep")[3],
                    label='K + V (must stay in SRAM)')
    ax.fill_between(N_values, np.minimum(kv_kb, sram_kb), sram_kb,
                    where=(kv_kb < sram_kb), alpha=0.20,
                    color=sns.color_palette("deep")[2],
                    label='Remaining SRAM (for Q tile, S tile, ...)')
    ax.fill_between(N_values, sram_kb, np.maximum(kv_kb, sram_kb),
                    where=(kv_kb >= sram_kb), alpha=0.15,
                    color=sns.color_palette("deep")[3])

    ax.set_xlabel('Sequence length N')
    ax.set_ylabel('Memory (KB)')
    ax.set_title('1D Tiling: K + V Eventually Fill SRAM')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xlim(N_values[0], N_values[-1])
    ax.set_ylim(0, sram_kb * 1.5)
    plt.tight_layout()
    plt.show()
