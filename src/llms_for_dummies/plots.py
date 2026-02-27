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


def plot_pe_properties(
    base_embeds: np.ndarray,
    pe_embeds: np.ndarray,
    label: str = "PE",
    gap: int = 4,
    anchor_positions=None,
):
    """Four-panel overview of a positional encoding's key properties.

    Args:
        base_embeds:      Raw token embeddings, shape (N, d).
        pe_embeds:        PE matrix, shape (N, d).
        label:            Name of the PE scheme.
        gap:              Offset for the translation-invariance panel.
        anchor_positions: Positions to highlight in the similarity panel.
                          Defaults to [0, N//4, N//2].
    """
    N = len(base_embeds)
    if anchor_positions is None:
        anchor_positions = [0, N // 4, N // 2]

    pal = sns.color_palette("deep")
    sim_matrix = pe_embeds @ pe_embeds.T

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(label, fontsize=13, fontweight="bold")

    # [0, 0] Compatible: norm before vs. after adding PE
    ax = axes[0, 0]
    ax.plot(np.linalg.norm(base_embeds, axis=1),
            label="Embeddings", color=pal[0], linestyle="--")
    ax.plot(np.linalg.norm(base_embeds + pe_embeds, axis=1),
            label=f"Embeddings + {label}", color=pal[3])
    ax.set_xlabel("Token position")
    ax.set_ylabel("Norm")
    ax.set_title("Embedding compatibility — norm before/after PE")
    ax.legend(fontsize=9)

    # [0, 1] Translation invariant: PE(i) · PE(i+gap) should be flat
    ax = axes[0, 1]
    sims = np.diag(sim_matrix, k=gap)
    ax.plot(sims, color=pal[3], label=label)
    ax.axhline(np.mean(sims), linestyle="--", color="grey", linewidth=1, label="Ideal (constant)")
    ax.set_xlabel("Token position")
    ax.set_ylabel(f"$PE(\\mathrm{{pos}}) \\cdot PE(\\mathrm{{pos}}+{gap})$")
    ax.set_title(f"Translation invariance — $PE(\\mathrm{{pos}}) \\cdot PE(\\mathrm{{pos}}+{gap})$")
    ax.legend(fontsize=9)

    # [1, 0] Smooth distance: full N×N similarity heatmap
    ax = axes[1, 0]
    im = ax.imshow(sim_matrix, cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    ax.set_title("Proximity — similarity heatmap")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # [1, 1] Smooth distance: similarity from a few anchor positions
    ax = axes[1, 1]
    for pos in anchor_positions:
        ax.plot(sim_matrix[pos], label=f"sim(pos {pos}, pos j)", alpha=0.8)
    ax.set_xlabel("Position j")
    ax.set_ylabel("Dot product")
    ax.set_title("Proximity — anchor curves")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_sinusoidal_wavelengths(
    bases=(100, 10000, 100000),
    dim_pairs=(0, 4, 8, 16),
    d_model: int = 64,
    n_positions: int = 100,
    highlight_pos: tuple = (5, 40),
):
    """Stacked waterfall showing sinusoidal PE frequencies at different bases.

    Each row is a waterfall of dimension pairs for one base value, stacked
    vertically with a shared x-axis so the effect of changing the base is
    immediately visible.

    Args:
        bases:       Base values to compare (one subplot per base).
        dim_pairs:   Frequency indices i for the waterfall rows.
        d_model:     Model dimension.
        n_positions: Number of positions shown on the x-axis.
    """
    pal = sns.color_palette("deep")
    n_bases = len(bases)
    fig, axes = plt.subplots(
        n_bases, 1, figsize=(12, 2.8 * n_bases), sharex=True,
    )
    if n_bases == 1:
        axes = [axes]

    positions = np.arange(n_positions)
    spacing = 2.5

    for ax_idx, (ax, base) in enumerate(zip(axes, bases)):
        for j, i in enumerate(dim_pairs):
            freq = 1.0 / (base ** (2 * i / d_model))
            wavelength = 2 * np.pi / freq
            offset = -j * spacing

            sin_y = np.sin(freq * positions) + offset
            cos_y = np.cos(freq * positions) + offset
            ax.plot(positions, sin_y,
                    color=pal[j], linewidth=1.3)
            ax.scatter(positions, sin_y,
                       color=pal[j], s=6, zorder=3)
            ax.plot(positions, cos_y,
                    color=pal[j], linewidth=1.3, linestyle="--", alpha=0.5)
            ax.scatter(positions, cos_y,
                       color=pal[j], s=6, alpha=0.4, zorder=3)
            ax.axhline(offset, color="grey", lw=0.3, linestyle="--")

            wl_str = (f"{wavelength:.0f}" if wavelength < 1_000
                      else f"{wavelength / 1_000:.0f}k")
            ax.text(n_positions + 2, offset, f"$i={i}$   λ ≈ {wl_str}",
                    va="center", fontsize=9, color=pal[j], fontweight="bold")

        ax.set_yticks([])
        ax.set_xlim(-1, n_positions)
        ax.set_title(f"base = {base:,}", fontsize=10, pad=10)

        # fine ↔ coarse arrow
        top_y = spacing / 2
        bot_y = -(len(dim_pairs) - 1) * spacing - spacing / 2
        ax.annotate("", xy=(-4, bot_y), xytext=(-4, top_y),
                    arrowprops=dict(arrowstyle="<->", color="grey", lw=1.2),
                    annotation_clip=False)
        mid_y = (top_y + bot_y) / 2
        ax.text(-7, mid_y, "fine\n⇅\ncoarse", ha="center", va="center",
                fontsize=8, color="grey", clip_on=False)

        # Vertical highlights for selected positions
        if highlight_pos is not None:
            hl_colors = ["black", "#8B0000"]
            for h_idx, pos_val in enumerate(highlight_pos):
                hc = hl_colors[h_idx % len(hl_colors)]
                ax.axvspan(pos_val - 0.4, pos_val + 0.4,
                           color=hc, alpha=0.07, zorder=0)
                for j, i in enumerate(dim_pairs):
                    freq = 1.0 / (base ** (2 * i / d_model))
                    y_sin = np.sin(freq * pos_val) + (-j * spacing)
                    y_cos = np.cos(freq * pos_val) + (-j * spacing)
                    ax.scatter([pos_val], [y_sin], color=hc,
                               s=25, zorder=5, edgecolors="white", linewidths=0.5)
                    ax.scatter([pos_val], [y_cos], color=hc,
                               s=25, zorder=5, edgecolors="white", linewidths=0.5,
                               marker="D")
                if ax_idx == 0:
                    sign = 1 if h_idx % 2 == 0 else -1
                    ax.annotate(
                        f"$PE(\\mathrm{{pos}}={pos_val})$",
                        xy=(pos_val, top_y),
                        xytext=(pos_val + sign * 8, top_y + 1.2),
                        fontsize=9, fontweight="bold", color=hc,
                        arrowprops=dict(arrowstyle="->", color=hc, lw=1),
                    )

    # Single legend on the first subplot with neutral-coloured handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="grey", linewidth=1.3,
               label="$\\sin(i)$"),
        Line2D([], [], color="grey", linewidth=1.3, linestyle="--", alpha=0.5,
               label="$\\cos(i\\!+\\!1)$"),
        Line2D([], [], color="black", marker="o", linestyle="None",
               markersize=4, label="$PE(\\mathrm{pos})$ — sin"),
        Line2D([], [], color="black", marker="D", linestyle="None",
               markersize=4, label="$PE(\\mathrm{pos})$ — cos"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right",
                   framealpha=0.9, ncol=2, columnspacing=1.2)

    axes[-1].set_xlabel("Position")
    plt.tight_layout()
    plt.show()


def plot_pe_heatmap(pe: np.ndarray, max_len: int = 128, d_model: int = 64):
    """Heatmap of a positional encoding matrix.

    Args:
        pe:        PE matrix, shape (max_len, d_model).
        max_len:   Number of positions (rows) to display.
        d_model:   Model dimension (columns) to display.
    """
    sns.set_theme(style="dark")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pe[:max_len, :d_model], aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Position")
    ax.set_title("Sinusoidal Positional Encoding")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()


def plot_pe_rotation(
    pos: int = 10,
    k: int = 7,
    dim_pairs: tuple = (4, 8, 16),
    d_model: int = 128,
):
    """Show how advancing by k positions rotates each sin/cos pair on the unit circle.

    One circle per frequency index. The point at `pos` is shown, the point at
    `pos+k` is shown, and the rotation arc between them is drawn.  Low-index
    pairs rotate fast (large arc), high-index pairs rotate slowly (small arc).

    Args:
        pos:       Starting position.
        k:         Offset (number of positions to advance).
        dim_pairs: Frequency indices i to display (one circle each).
        d_model:   Model dimension (controls frequency spacing).
    """
    sns.set_theme(style="white", palette="muted", font_scale=1.1)
    pal = sns.color_palette("deep")
    col_start = pal[0]   # blue — pos
    col_end = pal[3]     # red  — pos + k
    col_arc = pal[1]     # orange — rotation arc

    n = len(dim_pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    for col, i in enumerate(dim_pairs):
        ax = axes[col]

        omega = 1.0 / (10000 ** (2 * i / d_model))
        theta = omega * pos
        phi = omega * k
        phi_deg = np.degrees(phi)

        # Cap arc at one full revolution
        phi_draw = min(phi, 2 * np.pi - 0.15)

        # Unit circle
        circle_t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(circle_t), np.sin(circle_t),
                color="#cccccc", linewidth=1.2)

        # Subtle axes through origin
        ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
        ax.axvline(0, color="#dddddd", lw=0.6, zorder=0)

        # Points
        x0, y0 = np.cos(theta), np.sin(theta)
        x1, y1 = np.cos(theta + phi), np.sin(theta + phi)

        # Radius vectors from origin — the "vectors" being rotated
        ax.annotate("", xy=(x0, y0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col_start,
                                    lw=1.5, mutation_scale=12, alpha=0.6))
        ax.annotate("", xy=(x1, y1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col_end,
                                    lw=1.5, mutation_scale=12, alpha=0.6))

        # θ arc from x-axis to the starting vector
        theta_draw = theta % (2 * np.pi)
        r_theta = 0.30
        theta_arc = np.linspace(0, theta_draw, 40)
        ax.plot(r_theta * np.cos(theta_arc), r_theta * np.sin(theta_arc),
                color=col_start, linewidth=1.2, alpha=0.5)
        mid_theta = theta_draw / 2
        ax.text(0.42 * np.cos(mid_theta), 0.42 * np.sin(mid_theta),
                r"$\theta$", fontsize=9, ha="center", va="center",
                color=col_start, alpha=0.7)

        # Dots on the circle
        ax.scatter([x0], [y0], color=col_start, s=80, zorder=5,
                   edgecolors="white", linewidths=1.0)
        ax.scatter([x1], [y1], color=col_end, s=80, zorder=5,
                   edgecolors="white", linewidths=1.0)

        # --- Label placement ---
        ang0 = np.arctan2(y0, x0)
        ang1 = np.arctan2(y1, x1)
        label_r = 1.38

        tx0, ty0 = label_r * np.cos(ang0), label_r * np.sin(ang0)
        tx1, ty1 = label_r * np.cos(ang1), label_r * np.sin(ang1)

        label_dist = np.sqrt((tx1 - tx0) ** 2 + (ty1 - ty0) ** 2)
        if label_dist < 0.6:
            mid_ang = (ang0 + ang1) / 2
            nudge = 0.35
            tx0 = label_r * np.cos(ang0) + nudge * np.cos(mid_ang + np.pi / 2)
            ty0 = label_r * np.sin(ang0) + nudge * np.sin(mid_ang + np.pi / 2)
            tx1 = label_r * np.cos(ang1) + nudge * np.cos(mid_ang - np.pi / 2)
            ty1 = label_r * np.sin(ang1) + nudge * np.sin(mid_ang - np.pi / 2)

        ax.annotate(f"pos = {pos}",
                    xy=(x0, y0), xytext=(tx0, ty0),
                    fontsize=8.5, color=col_start, fontweight="bold",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=col_start,
                                    lw=0.8, shrinkB=3))
        ax.annotate(f"pos = {pos + k}",
                    xy=(x1, y1), xytext=(tx1, ty1),
                    fontsize=8.5, color=col_end, fontweight="bold",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=col_end,
                                    lw=0.8, shrinkB=3))

        # --- Rotation arc with arrowhead ---
        r_arc = 0.55
        arc_t = np.linspace(theta, theta + phi_draw, 80)
        ax.plot(r_arc * np.cos(arc_t), r_arc * np.sin(arc_t),
                color=col_arc, linewidth=2.2, alpha=0.85, solid_capstyle="round")
        ax.annotate("",
                    xy=(r_arc * np.cos(arc_t[-1]), r_arc * np.sin(arc_t[-1])),
                    xytext=(r_arc * np.cos(arc_t[-4]), r_arc * np.sin(arc_t[-4])),
                    arrowprops=dict(arrowstyle="-|>", color=col_arc, lw=2,
                                    mutation_scale=14))

        # φ label
        mid_angle = theta + phi_draw / 2
        phi_label_r = 0.30 if phi_deg > 30 else r_arc + 0.22
        ax.text(phi_label_r * np.cos(mid_angle),
                phi_label_r * np.sin(mid_angle),
                f"$\\phi = {phi_deg:.0f}°$",
                fontsize=9, ha="center", va="center", color=col_arc,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.8))

        # Title with wavelength
        wl = 2 * np.pi / omega
        wl_str = f"{wl:.0f}" if wl < 1_000 else f"{wl / 1_000:.0f}k"
        ax.set_title(f"$i = {i}$   (λ ≈ {wl_str})", fontsize=10.5, pad=10)

        # Styling
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.75, 1.75)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"Advancing by $k = {k}$ positions rotates each sin/cos pair",
        fontsize=12, fontweight="bold",
    )
    plt.show()
