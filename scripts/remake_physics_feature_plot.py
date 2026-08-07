import os
import math
import matplotlib.pyplot as plt
import numpy as np


def remake_plot(output_path="figures/physics_feature_contribution.png", metrics=None):
    """Create the two-panel figure the user requested.

    - Top: 4 bars — raw, raw + simple transforms, raw + physics-engineered, raw + both
    - Bottom: simple transform checks (individual simple-transform deltas)
    If `metrics` is None, example numbers from the original figure are used.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if metrics is None:
        # Example numbers (grouped held-out R^2 delta vs baseline)
        # top_metrics are absolute increases over raw baseline
        top_metrics = {
            "Raw physical inputs": 0.0,
            "Raw + simple transforms": 0.009,  # approx +0.004..+0.009 in image
            "Raw + composite physics features": 0.014,
            "Raw + simple + physics": 0.023,
        }

        # bottom: simple transform check deltas (vs raw baseline)
        bottom_metrics = [
            ("Raw only", 0.0),
            ("v_inf^2", +0.004),
            ("1/r_p", -0.004),
            ("f_spin", -0.021),
            ("asteroid_radius", +0.006),
            ("all simple transforms", +0.009),
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

    # Top panel: grouped bar chart style
    ax = axes[0]
    labels = list(top_metrics.keys())
    values = [top_metrics[k] for k in labels]
    colors = ["#a6cee3", "#33a02c", "#ff7f00", "#e31a1c"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors[: len(values)], edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Change in grouped held-out $R^2$ vs raw baseline")
    ax.set_title("Contribution of Physics-Derived Features")

    # Remove text overlays and annotations per request

    # Bottom panel: simple transforms check (horizontal baseline at 0)
    ax2 = axes[1]
    names = [n for n, v in bottom_metrics]
    vals = [v for n, v in bottom_metrics]
    x2 = np.arange(len(names))
    bars2 = ax2.bar(x2, vals, color="#6c83b5", edgecolor="none")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("Change in grouped held-out $R^2$ vs raw baseline")
    ax2.set_title("Simple Transform Checks")

    # Layout tweaks: add space between top and bottom, leave room for x-label rotation
    plt.subplots_adjust(hspace=0.45, top=0.92)

    # Save figure
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    remake_plot()
