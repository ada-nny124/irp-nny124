###############################################################################
# Debugged Plotter: Forced Material Detection and Thicker Slicing
###############################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import h5py

def get_material_mapping(init_path):
    with h5py.File(init_path, "r") as f:
        pids = f["PartType0/ParticleIDs"][()]
        mats = f["PartType0/MaterialIDs"][()]
        
        # DEBUG: Print what is actually in the file
        unique_mats = np.unique(mats)
        print(f"DEBUG: Found Material IDs in Init: {unique_mats}")
        
    return dict(zip(pids, mats))

def load_snapshot(filename, id_map):
    with h5py.File(filename, "r") as f:
        u_l = float(f["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2
        box = f["Header"].attrs["BoxSize"]
        pos = (f["PartType0/Coordinates"][()] - 0.5 * box) * u_l
        pids = f["PartType0/ParticleIDs"][()]
        
        # Match IDs to the materials from Init
        mats = np.array([id_map.get(pid, -1) for pid in pids])
    return pos, mats

def plot_snapshot(pos, mats, save_path):
    R_E = 6.3710e6
    x, y, z = pos[:, 0] / R_E, pos[:, 1] / R_E, pos[:, 2] / R_E

    # INCREASED SLICE THICKNESS
    # Using 1.0 R_E ensures we catch the mantle even if it's spread out
    z_limit = 1.0 
    mask = np.abs(z - np.median(z)) < z_limit

    plt.figure(figsize=(10, 8))
    
    # Plotting loop
    unique_found = np.unique(mats[mask])
    print(f"Snapshot {os.path.basename(save_path)} contains materials: {unique_found}")

    for m_id in unique_found:
        m_mask = (mats == m_id) & mask
        plt.scatter(x[m_mask], y[m_mask], s=2, label=f"Material {m_id}", alpha=0.6)

    # Auto-Zoom
    pad = 0.5
    plt.xlim(np.min(x[mask]) - pad, np.max(x[mask]) + pad)
    plt.ylim(np.min(y[mask]) - pad, np.max(y[mask]) + pad)
    
    plt.gca().set_aspect("equal")
    plt.legend(loc='upper right')
    plt.title(os.path.basename(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    snap_dir = "snapshots"
    plot_dir = "plots_debug"
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)

    files = sorted([f for f in os.listdir(snap_dir) if f.endswith(".hdf5")])
    init_file = [f for f in files if "init" in f][0]
    
    # 1. Map IDs
    id_map = get_material_mapping(os.path.join(snap_dir, init_file))

    # 2. Process
    for f in files:
        if "6000" in f and "init" not in f: continue # Skip corrupted
        try:
            p, m = load_snapshot(os.path.join(snap_dir, f), id_map)
            plot_snapshot(p, m, os.path.join(plot_dir, f.replace(".hdf5", ".png")))
        except Exception as e:
            print(f"Failed {f}: {e}")