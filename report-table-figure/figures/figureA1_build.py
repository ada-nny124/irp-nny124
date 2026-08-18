import os
import math
import matplotlib.pyplot as plt
import numpy as np


def remake_plot(output_path="figures/physics_feature_contribution.png", metrics=None, baseline_r2=0.862):
    """Create the two-panel figure the user requested.

    - Top: 4 bars — raw, raw + simple transforms, raw + physics-engineered, raw + both
    - Bottom: simple transform checks (individual simple-transform deltas)
    If `metrics` is None, example numbers from the original figure are used.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if metrics is None:
        # Example numbers (grouped held-out R^2 delta vs baseline)
        # top_metrics are absolute increases over raw baseline
        # top_metrics are deltas over the raw baseline R^2
        top_metrics = {
            "Raw": 0.0,
            "Raw + simple": 0.009,
            "Raw + physics": 0.014,
            "All": 0.023,
        }

        # bottom: simple transform check deltas (vs raw baseline)
        bottom_metrics = [
            ("Raw", 0.0),
            ("v_inf^2", +0.004),
            ("1/r_p", -0.004),
            ("f_spin", -0.021),
            ("radius", +0.006),
            ("all simple", +0.009),
        ]
    else:
        top_metrics = metrics.get("top", {})
        bottom_metrics = metrics.get("bottom", [])

    # Start plotting — prefer seaborn style but gracefully fall back if unavailable
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        try:
            import seaborn as _sns  # noqa: F401
            plt.style.use("seaborn")
        except Exception:
            plt.style.use("classic")
    fig, axes = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [2, 1]})

    # Top panel: grouped bar chart style — show deltas (±) as bar heights
    ax = axes[0]
    labels = list(top_metrics.keys())
    deltas = [top_metrics[k] for k in labels]
    # bar heights show deltas (change vs baseline). We'll annotate R^2 underneath each delta.
    values = deltas
    colors = ["#a6cee3", "#33a02c", "#ff7f00", "#e31a1c"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors[: len(values)], edgecolor="none")

    # add zero baseline line
    ax.axhline(0, color='k', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.tick_params(axis='x', which='major', pad=8)
    ax.set_ylabel("Change in grouped held-out $R^2$ vs raw baseline")
    ax.set_title("Contribution of Physics-Derived Features")

    # Remove text overlays and annotations per request

    # Bottom panel: simple transforms check (horizontal baseline at 0)
    ax2 = axes[1]
    names = [n for n, v in bottom_metrics]
    deltas_b = [v for n, v in bottom_metrics]
    vals = deltas_b  # bottom panel should also show deltas (±)
    x2 = np.arange(len(names))
    bars2 = ax2.bar(x2, vals, color="#6c83b5", edgecolor="none")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names, rotation=0, ha="center", fontsize=8)
    ax2.tick_params(axis='x', which='major', pad=8)
    ax2.set_ylabel("Change in grouped held-out $R^2$ vs raw baseline")
    ax2.set_title("Simple Transform Checks")

    # Annotate bars with absolute R^2 and delta inside the bar
    def annotate_bar(ax, bar, delta, abs_r2):
        cx = bar.get_x() + bar.get_width() / 2.0
        h = bar.get_height()
        # delta string (prominent) and R^2 string (smaller, under the delta)
        delta_str = f"{delta:+.3f}"
        r2_str = f"R² {abs_r2:.3f}"

        # place delta near the center of the bar (or just above if very small)
        if delta >= 0:
            delta_y = max(h * 0.5, 0.001)
            r2_y = delta_y - 0.02
        else:
            delta_y = min(h * 0.5, -0.001)
            r2_y = delta_y - 0.02

        face = bar.get_facecolor()
        text_color = "white" if face[0] < 0.6 else "black"

        ax.text(cx, delta_y, delta_str, ha="center", va="center", fontsize=10, fontweight="semibold", color=text_color)
        # place R^2 slightly below the delta for reference (smaller)
        ax.text(cx, r2_y, r2_str, ha="center", va="center", fontsize=8, color=text_color)

    for rect, abs_v, d in zip(bars, values, deltas):
        annotate_bar(ax, rect, abs_v, d)
    for rect, abs_v, d in zip(bars2, vals, deltas_b):
        annotate_bar(ax2, rect, abs_v, d)

    # Layout tweaks: add space between top and bottom, leave room for x-label rotation
    plt.subplots_adjust(hspace=0.45, top=0.92)

    # Save figure
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # produce full figure
    remake_plot()
    # also produce the bottom-only simple-transform checks as a separate file
    def remake_simple_transform_plot(output_path="figures/simple_transform_checks.png", metrics=None, baseline_r2=0.862):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if metrics is None:
            bottom_metrics_local = [
                ("Raw", 0.0),
                ("v_inf^2", +0.004),
                ("1/r_p", -0.004),
                ("f_spin", -0.021),
                ("radius", +0.006),
                ("all simple", +0.009),
            ]
        else:
            bottom_metrics_local = metrics.get("bottom", [])

        names = [n for n, v in bottom_metrics_local]
        deltas_b = [v for n, v in bottom_metrics_local]

        plt.style.use("classic")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        x = np.arange(len(names))
        bars = ax.bar(x, deltas_b, color="#6c83b5", edgecolor="none")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0, ha="center", fontsize=8)
        ax.set_ylabel("Δ R²")
        ax.set_title("Simple Transform Checks")

        # annotate delta and small R^2 under it (use baseline)
        for rect, d in zip(bars, deltas_b):
            cx = rect.get_x() + rect.get_width() / 2.0
            h = rect.get_height()
            delta_str = f"{d:+.3f}"
            r2_str = f"R² {baseline_r2 + d:.3f}"
            # place delta inside or just above
            if d >= 0:
                dy = max(h * 0.5, 0.001)
                r2y = dy - 0.02
            else:
                dy = min(h * 0.5, -0.001)
                r2y = dy - 0.02
            face = rect.get_facecolor()
            text_color = "white" if face[0] < 0.6 else "black"
            ax.text(cx, dy, delta_str, ha="center", va="center", fontsize=10, fontweight="semibold", color=text_color)
            ax.text(cx, r2y, r2_str, ha="center", va="center", fontsize=8, color=text_color)

        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    remake_simple_transform_plot()
