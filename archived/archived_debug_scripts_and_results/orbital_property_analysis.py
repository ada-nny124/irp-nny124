import os
import h5py
import numpy as np
import pandas as pd
import woma

# --- Constants for Mars ---
G = 6.67430e-11  
M_MARS = 6.4171e23  
R_MARS = 3.3895e6  

def calculate_physics(pos, vel):
    """Core orbital mechanics calculation."""
    r_mag = np.linalg.norm(pos, axis=1)
    v_sq = np.sum(vel**2, axis=1)
    energy = (v_sq / 2.0) - (G * M_MARS / r_mag)
    h_vec = np.cross(pos, vel)
    h_mag = np.linalg.norm(h_vec, axis=1)
    
    a = -G * M_MARS / (2.0 * np.where(energy == 0, 1e-10, energy))
    e_sq = 1.0 + (2.0 * energy * h_mag**2) / (G * M_MARS)**2
    e = np.sqrt(np.maximum(e_sq, 0))
    
    is_bound = energy < 0
    # Outcomes: Apoapsis for bound, v_inf for unbound
    outcomes = np.zeros(len(pos))
    outcomes[is_bound] = a[is_bound] * (1.0 + e[is_bound]) / R_MARS
    outcomes[~is_bound] = np.sqrt(np.abs(2.0 * energy[~is_bound])) / 1000.0 # km/s
    
    return outcomes, is_bound

def process_all_snapshots(snapshot_dir):
    summary_data = []
    
    files = [f for f in os.listdir(snapshot_dir) if f.endswith(".hdf5")]
    # Simple sort based on the time-stamp in filename
    files.sort() 

    for filename in files:
        if "6000" in filename and "init" not in filename: continue # Skip corrupted
        
        path = os.path.join(snapshot_dir, filename)
        try:
            with h5py.File(path, "r") as f:
                u_l = float(f["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2
                u_v = u_l / float(f["Units"].attrs["Unit time in cgs (U_t)"])
                
                box = f["Header"].attrs["BoxSize"]
                pos = (f["PartType0/Coordinates"][()] - 0.5 * box) * u_l
                vel = f["PartType0/Velocities"][()] * u_v
                pids = f["PartType0/ParticleIDs"][()]

            outcomes, is_bound = calculate_physics(pos, vel)
            
            # Calculate Summary Stats
            bound_frac = np.sum(is_bound) / len(is_bound)
            avg_apoapsis = np.mean(outcomes[is_bound]) if any(is_bound) else 0
            
            summary_data.append({
                "snapshot": filename,
                "bound_mass_fraction": bound_frac,
                "avg_apoapsis_Rm": avg_apoapsis,
                "particle_count": len(pids)
            })

            # For the FINAL snapshot (90ks), save a detailed CSV for ML
            if "90000" in filename and "fof" not in filename:
                df_ml = pd.DataFrame({
                    "particle_id": pids,
                    "x_Rm": pos[:, 0] / R_MARS,
                    "y_Rm": pos[:, 1] / R_MARS,
                    "z_Rm": pos[:, 2] / R_MARS,
                    "v_mag_kms": np.linalg.norm(vel, axis=1) / 1000.0,
                    "is_bound": is_bound.astype(int),
                    "outcome_val": outcomes # Ra or V_inf
                })
                df_ml.to_csv("ml_training_data_90ks.csv", index=False)
                print(f"Created detailed ML dataset from {filename}")

        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    # Save the summary of all snapshots
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("simulation_summary.csv", index=False)
    print("Created simulation_summary.csv")

if __name__ == "__main__":
    process_all_snapshots("snapshots")