import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import glob
import os
import sys
import gc

def run_fof(coords, linking_length):
    """Efficient FoF for clustering."""
    tree = cKDTree(coords)
    pairs = tree.query_pairs(linking_length, output_type='ndarray')
    n_particles = len(coords)
    if len(pairs) == 0: return np.arange(n_particles)
    
    row, col = pairs[:, 0], pairs[:, 1]
    data = np.ones(len(pairs), dtype=bool)
    adj_matrix = csr_matrix((data, (row, col)), shape=(n_particles, n_particles))
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    del adj_matrix
    gc.collect()
    return labels

def extract_master_dataset():
    sim_dirs = sorted(glob.glob("sim_*"))
    if not sim_dirs:
        sim_dirs = ["."]

    G_SI = 6.67430e-11  
    M_MARS_SI = 6.4171e23  
    LINKING_LENGTH = 0.5 
    master_dataset = []

    for sim_path in sim_dirs:
        sim_id = os.path.basename(sim_path) if sim_path != "." else "test_run"
        print(f"--> Harvesting {sim_id}...")

        snapshots = sorted(glob.glob(os.path.join(sim_path, "snapshots/*.hdf5")))
        if not snapshots: continue

        max_fragments = 0
        peak_density = 0.0
        time_at_peak = 0.0

        for i, filepath in enumerate(snapshots):
            try:
                # --- NEW: Filename Parsing Logic ---
                # This assumes filename like: demo_impact_n50_v5.2_p3500.hdf5
                fname = os.path.basename(filepath)
                try:
                    parts = fname.replace('.hdf5', '').split('_')
                    # We look for the parts starting with 'v' and 'p'
                    v_input = float([p for p in parts if p.startswith('v')][0][1:])
                    p_input = float([p for p in parts if p.startswith('p')][0][1:])
                except:
                    # Fallback if naming convention isn't met
                    v_input, p_input = 5.0, 3500.0 

                with h5py.File(filepath, "r") as f:
                    u_mass = float(np.array(f["Units"].attrs["Unit mass in cgs (U_M)"]).item()) * 1e-3
                    u_length = float(np.array(f["Units"].attrs["Unit length in cgs (U_L)"]).item()) * 1e-2
                    u_time = float(np.array(f["Units"].attrs["Unit time in cgs (U_t)"]).item())
                    u_vel = u_length / u_time

                    raw_coords = np.array(f["PartType0/Coordinates"][()])
                    raw_masses = np.array(f["PartType0/Masses"][()])
                    raw_densities = np.array(f["PartType0/Densities"][()])

                    current_dens = np.max(raw_densities)
                    if current_dens > peak_density:
                        peak_density = current_dens
                        time_at_peak = float(np.array(f["Header"].attrs["Time"]).item()) * u_time

                    labels = run_fof(raw_coords, LINKING_LENGTH)
                    counts = np.bincount(labels)
                    current_frags = np.sum(counts >= 10)
                    if current_frags > max_fragments:
                        max_fragments = current_frags

                    if i == len(snapshots) - 1:
                        raw_vels = np.array(f["PartType0/Velocities"][()])
                        total_mass_si = np.sum(raw_masses) * u_mass
                        pos_si = raw_coords * u_length
                        vel_si = raw_vels * u_vel
                        mass_si = raw_masses * u_mass

                        r_mag = np.linalg.norm(pos_si, axis=1)
                        v_mag = np.linalg.norm(vel_si, axis=1)
                        specific_energy = (0.5 * v_mag**2) - ((G_SI * M_MARS_SI) / r_mag)
                        m_bound = np.sum(mass_si[specific_energy < 0])

                        master_dataset.append({
                            "Sim_ID": sim_id,
                            "Input_Velocity_km_s": v_input,
                            "Input_Periapsis_km": p_input,
                            "Input_Total_Mass_kg": total_mass_si,
                            "Max_Fragments_Observed": int(max_fragments),
                            "Time_to_Peak_Density_s": time_at_peak,
                            "Output_Bound_Fraction": m_bound / total_mass_si if total_mass_si > 0 else 0,
                            "Output_Max_Fragment_Mass_kg": np.max(np.bincount(labels, weights=mass_si))
                        })

                    del raw_coords, raw_masses, raw_densities
                    gc.collect()

            except Exception as e:
                print(f"Error in {filepath}: {e}")

    pd.DataFrame(master_dataset).to_csv("master_training_data.csv", index=False)
    print("-" * 50)
    print("SUCCESS: Master CSV saved with filename-parsed parameters.")

if __name__ == "__main__":
    extract_master_dataset()