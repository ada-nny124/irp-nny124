from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
SPIN_ORDER = ("no spin", "3h z", "4.7h z")
SPIN_LABELS = {
    "no spin": "No spin",
    "3h z": "3 h, +z",
    "4.7h z": "4.7 h, +z",
}
SPIN_COLORS = {
    "no spin": "#333333",
    "3h z": "#245c90",
    "4.7h z": "#9a6700",
}
VELOCITY_ORDER = (0.0, 0.2, 0.4, 0.6, 0.8)
VELOCITY_COLORS = {
    0.0: "#1f4e79",
    0.2: "#3a7d44",
    0.4: "#a6ad00",
    0.6: "#d17c00",
    0.8: "#d94801",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bound-table",
        default="extraction_outputs/bound_outcomes.csv",
        help="Path to the extracted bound-outcomes CSV.",
    )
    parser.add_argument(
        "--png-out",
        default=str(SCRIPT_DIR / "regime_dependent_parameter_story.png"),
        help="PNG output path.",
    )
    return parser.parse_args()


def parse_spin_label(spin_code: str) -> str:
    if not spin_code:
        return "no spin"
    return f"{float(spin_code[1:4]) / 10.0:g}h {spin_code[4:] or 'none'}"


def load_series(path: Path) -> dict[float, dict[str, list[tuple[float, float]]]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    grouped: dict[tuple[float, float], dict[str, float]] = defaultdict(dict)

    for row in rows:
        if row["mass_code"] != "A2000":
            continue
        if row["resolution_code"] != "n65":
            continue
        if row["timestep"] != "90000":
            continue
        if float(row["fof_linking_length"]) != 0.004:
            continue

        velocity = float(row["velocity_code"][1:]) / 10.0
        if velocity not in VELOCITY_ORDER:
            continue

        periapsis = float(row["periapsis_code"][1:]) / 10.0
        spin = parse_spin_label(row["spin_code"])
        grouped[(velocity, periapsis)][spin] = float(row["bound_mass_fraction"])

    series: dict[float, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for (velocity, periapsis), spin_map in sorted(grouped.items()):
        if not set(SPIN_ORDER).issubset(spin_map):
            continue
        for spin in SPIN_ORDER:
            series[velocity][spin].append((periapsis, spin_map[spin]))
    return series


def compute_spin_delta_series(
    series: dict[float, dict[str, list[tuple[float, float]]]]
) -> dict[float, list[tuple[float, float]]]:
    deltas: dict[float, list[tuple[float, float]]] = {}
    for velocity, spin_series in series.items():
        peri_to_values: dict[float, list[float]] = defaultdict(list)
        for spin in SPIN_ORDER:
            for periapsis, bmf in spin_series[spin]:
                peri_to_values[periapsis].append(bmf)
        deltas[velocity] = []
        for periapsis, values in sorted(peri_to_values.items()):
            if len(values) == len(SPIN_ORDER):
                deltas[velocity].append((periapsis, max(values) - min(values)))
    return deltas


def compute_velocity_delta_series(path: Path) -> dict[str, list[tuple[float, float]]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    grouped: dict[tuple[str, float], dict[float, float]] = defaultdict(dict)

    for row in rows:
        if row["mass_code"] != "A2000":
            continue
        if row["resolution_code"] != "n65":
            continue
        if row["timestep"] != "90000":
            continue
        if float(row["fof_linking_length"]) != 0.004:
            continue

        spin = parse_spin_label(row["spin_code"])
        if spin not in SPIN_ORDER:
            continue
        velocity = float(row["velocity_code"][1:]) / 10.0
        if velocity not in VELOCITY_ORDER:
            continue
        periapsis = float(row["periapsis_code"][1:]) / 10.0
        grouped[(spin, periapsis)][velocity] = float(row["bound_mass_fraction"])

    deltas: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (spin, periapsis), velocity_map in sorted(grouped.items()):
        if len(velocity_map) < 3:
            continue
        deltas[spin].append((periapsis, max(velocity_map.values()) - min(velocity_map.values())))
    return deltas


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def make_plot(
    spin_deltas: dict[float, list[tuple[float, float]]],
    velocity_deltas: dict[str, list[tuple[float, float]]],
    png_out: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.4), sharey=True)
    ax_spin, ax_velocity = axes

    for velocity in VELOCITY_ORDER:
        delta_points = spin_deltas.get(velocity, [])
        if not delta_points:
            continue
        color = VELOCITY_COLORS[velocity]
        ax_spin.plot(
            [point[0] for point in delta_points],
            [point[1] for point in delta_points],
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=5.3,
            label=f"{velocity:g}",
        )

    for spin in SPIN_ORDER:
        delta_points = velocity_deltas.get(spin, [])
        if not delta_points:
            continue
        color = SPIN_COLORS[spin]
        ax_velocity.plot(
            [point[0] for point in delta_points],
            [point[1] for point in delta_points],
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=5.3,
            label=SPIN_LABELS[spin],
        )

    for ax in axes:
        ax.set_xlim(1.05, 2.45)
        ax.set_ylim(0.0, 0.30)
        style_axes(ax)
        ax.axvspan(1.05, 1.55, color="#e8f1f8", alpha=0.52, zorder=0)
        ax.axvspan(1.55, 2.45, color="#f7efe3", alpha=0.48, zorder=0)
        ax.axvline(1.55, color="#b8b8b8", linewidth=1.1, linestyle="--", zorder=1)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
        ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=18)

    ax_spin.set_ylabel("Local spread in BMF", fontsize=18)
    ax_spin.set_title("Change spin, hold velocity family fixed", fontsize=14, pad=10)
    ax_velocity.set_title("Change velocity, hold spin state fixed", fontsize=14, pad=10)

    velocity_handles = [
        Line2D([0], [0], color=VELOCITY_COLORS[velocity], linewidth=2.3, marker="o", markersize=5.3, label=f"{velocity:g}")
        for velocity in VELOCITY_ORDER
        if spin_deltas.get(velocity)
    ]
    ax_spin.legend(
        handles=velocity_handles,
        title=r"$v_{\infty}$ (km s$^{-1}$)",
        loc="upper left",
        frameon=True,
        fontsize=10,
        title_fontsize=12,
    )

    spin_handles = [
        Line2D([0], [0], color=SPIN_COLORS[spin], linewidth=2.3, marker="o", markersize=5.3, label=SPIN_LABELS[spin])
        for spin in SPIN_ORDER
        if velocity_deltas.get(spin)
    ]
    ax_velocity.legend(
        handles=spin_handles,
        title="Fixed spin state",
        loc="upper left",
        frameon=True,
        fontsize=10,
        title_fontsize=12,
    )

    fig.suptitle("Different regimes can make different inputs look important", fontsize=18, y=0.98)
    fig.text(
        0.5,
        0.075,
        "Matched subset: A2000, n65, timestep 90000, FoF 0.004. "
        "Left: ΔBMF_spin = max(BMF across spins) - min(BMF across spins). "
        "Right: ΔBMF_velocity = max(BMF across velocities) - min(BMF across velocities).",
        ha="center",
        fontsize=10.0,
        color="#444444",
    )
    fig.text(
        0.5,
        0.04,
        "The global average cannot show this regime dependence: one input can look moderate overall but become a strong local separator in a restricted region.",
        ha="center",
        fontsize=10.0,
        color="#444444",
    )

    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.03, 0.14, 0.99, 0.91))
    fig.savefig(png_out, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    series = load_series(Path(args.bound_table))
    spin_deltas = compute_spin_delta_series(series)
    velocity_deltas = compute_velocity_delta_series(Path(args.bound_table))
    make_plot(spin_deltas, velocity_deltas, Path(args.png_out))
    total_curves = sum(len(spin_series) for spin_series in series.values())
    print(f"Velocity families plotted: {len(series)}")
    print(f"Spin curves plotted: {total_curves}")
    print(f"PNG: {args.png_out}")


if __name__ == "__main__":
    main()
