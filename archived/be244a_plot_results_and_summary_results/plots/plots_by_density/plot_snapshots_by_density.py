###############################################################################
# This file is part of SWIFT.
# Copyright (c) 2023 Jacob Kegerreis (jacob.kegerreis@durham.ac.uk)
##############################################################################

"""Plot the snapshots from the DemoImpact example, slicing particles by density."""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import h5py
import woma

# Plotting options
font_size = 20
params = {
    "axes.labelsize": font_size,
    "font.size": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "font.family": "serif",
}
matplotlib.rcParams.update(params)

def load_snapshot(filename):
    """Load and convert the particle data to plot."""
    with h5py.File(filename, "r") as f:
        # Units from file metadata
        file_to_SI = woma.Conversions(
            m=float(f["Units"].attrs["Unit mass in cgs (U_M)"]) * 1e-3,
            l=float(f["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2,
            t=float(f["Units"].attrs["Unit time in cgs (U_t)"]),
        )

        # Particle data centered in the box
        A2_pos = (
            np.array(f["PartType0/Coordinates"][()])
            - 0.5 * f["Header"].attrs["BoxSize"]
        ) * file_to_SI.l
        
        # Grab the density
        densities = np.array(f["PartType0/Densities"][()])

    return A2_pos, densities


def plot_snapshot(A2_pos, densities): 
    """Plot a mid-plane slice with auto-scaling to see the particles."""
    plt.figure(figsize=(8, 7))
    ax = plt.gca()

    # 1. Convert to Earth Radii
    R_E = 6.3710e6  # meters
    x = A2_pos[:, 0] / R_E
    y = A2_pos[:, 1] / R_E
    z = A2_pos[:, 2] / R_E

    # 2. Dynamic Slicing
    # Instead of a fixed meter value, take the middle 5% of the particle distribution
    z_threshold = (np.max(z) - np.min(z)) * 0.05
    mask = np.abs(z - np.median(z)) < z_threshold

    # 3. Plotting with LARGER points
    # If they still look like specks, increase 's' to 20 or 50
    scatter = ax.scatter(
        x[mask], y[mask], 
        s=15, 
        c=densities[mask], 
        cmap='magma', 
        norm=colors.LogNorm() 
    )

    # 4. AUTO-ZOOM AXES
    # This ensures the 'specks' fill the screen so you can see if they are planets
    padding = 0.5 
    ax.set_xlim(np.min(x[mask]) - padding, np.max(x[mask]) + padding)
    ax.set_ylim(np.min(y[mask]) - padding, np.max(y[mask]) + padding)
    
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ ($R_\oplus$)")
    ax.set_ylabel(r"$y$ ($R_\oplus$)")
    plt.colorbar(scatter, label="Density")
    plt.tight_layout()


def get_sort_key(filename):
    """Get a sort key for the snapshot files, handling extensions and extra tags."""
    # Strip extension to prevent "13000.hdf5" ValueError
    name_clean = filename.replace(".hdf5", "")
    
    if "init" in name_clean:
        return -1
    
    parts = name_clean.split("_")
    try:
        # Find the index of the version marker 'v00'
        idx = parts.index("v00")
        # The number is usually the next part
        num_str = parts[idx + 1]
        # Handle cases like "90000_fof" by splitting on underscore again
        return int(num_str.split("_")[0])
    except (ValueError, IndexError):
        return 999999


if __name__ == "__main__":
    snapshot_dir = "snapshots"
    plot_dir = "plots"

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Get and sort files
    files = [f for f in os.listdir(snapshot_dir) if f.endswith(".hdf5")]
    files = sorted(files, key=get_sort_key)

    for filename in files:
        full_path = os.path.join(snapshot_dir, filename)
        
        try:
            print(f"Processing {filename}...", end="\r")
            
            # Load and Plot
            A2_pos, densities = load_snapshot(full_path)
            plot_snapshot(A2_pos, densities)

            # Save
            save_name = filename.replace(".hdf5", ".png")
            plt.savefig(os.path.join(plot_dir, save_name), dpi=200)
            plt.close()
            
        except OSError as e:
            print(f"\n[Error] Skipping {filename}: File is likely truncated/corrupted.")
        except Exception as e:
            print(f"\n[Error] Failed to process {filename}: {e}")

    print("\nDone! All valid plots are in the /plots folder.")