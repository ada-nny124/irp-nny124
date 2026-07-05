import os
import h5py
import numpy as np
import pandas as pd

# --- Constants for Mars ---
G = 6.67430e-11  
M_MARS = 6.4171e23  
R_MARS = 3.3895e6  
UNBOUND_ID = 2147483647  

OUTPUT_DIR = "extraction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_val(attr):
    """Safely extracts a value whether HDF5 loaded it as a scalar or an array."""
    return np.atleast_1d(attr)[0]

def get_masses_exacting(h5_file):
    """Targets the InitialMassTable found in your diagnostic."""
    if "PartType0/Masses" in h5_file:
        return h5_file["PartType0/Masses"][()]
    
    if "Header" in h5_file and "InitialMassTable" in h5_file["Header"].attrs:
        m_val = get_val(h5_file["Header"].attrs["InitialMassTable"])
        if m_val > 0:
            num = get_val(h5_file["Header"].attrs["NumPart_ThisFile"])
            return np.full(int(num), m_val)
            
    raise KeyError(f"Could not find non-zero mass in {h5_file.filename}")

def calculate_initial_spin(pos, vel, mass):
    """Derives spin period (hours) and orientation directly from particle vectors."""
    total_mass = np.sum(mass)
    if total_mass == 0: return 0.0, 0.0
    
    com_pos = np.sum(pos * mass[:, None], axis=0) / total_mass
    com_vel = np.sum(vel * mass[:, None], axis=0) / total_mass
    pos_prime = pos - com_pos
    vel_prime = vel - com_vel
    
    L_vec = np.sum(mass[:, None] * np.cross(pos_prime, vel_prime), axis=0)
    
    x, y, z = pos_prime[:, 0], pos_prime[:, 1], pos_prime[:, 2]
    r2 = x**2 + y**2 + z**2
    I_tensor = np.array([
        [np.sum(mass * (r2 - x**2)), np.sum(mass * (-x * y)), np.sum(mass * (-x * z))],
        [np.sum(mass * (-x * y)), np.sum(mass * (r2 - y**2)), np.sum(mass * (-y * z))],
        [np.sum(mass * (-x * z)), np.sum(mass * (-y * z)), np.sum(mass * (r2 - z**2))]
    ])
    
    try:
        omega_vec = np.linalg.inv(I_tensor).dot(L_vec)
        omega_mag = np.linalg.norm(omega_vec)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
        
    if omega_mag == 0: return 0.0, 0.0
        
    spin_period_hr = (2 * np.pi / omega_mag) / 3600.0
    spin_orientation_angle = np.degrees(np.arccos(omega_vec[2] / omega_mag))
    return spin_period_hr, spin_orientation_angle

def calculate_orbital_elements(pos, vel):
    """Calculates standard Keplerian elements."""
    r_mag, v_mag = np.linalg.norm(pos), np.linalg.norm(vel)
    if r_mag == 0: return [0]*5
    energy = (v_mag**2 / 2.0) - (G * M_MARS / r_mag)
    h_vec = np.cross(pos, vel)
    h_mag = np.linalg.norm(h_vec)
    a = -G * M_MARS / (2.0 * energy) if abs(energy) > 1e-10 else np.inf
    e_vec = (np.cross(vel, h_vec) / (G * M_MARS)) - (pos / r_mag)
    e = np.linalg.norm(e_vec)
    i = np.degrees(np.arccos(h_vec[2] / h_mag)) if h_mag > 0 else 0
    return a, e, i, a*(1-e), a*(1+e) if e < 1 else np.inf

def process_simulation(directory):
    summary_list, catalog_list = [], []
    inits = [f for f in os.listdir(directory) if f.startswith("init_")]
    
    for init_name in inits:
        sim_id = init_name.replace("init_", "").replace(".hdf5", "")
        f_phys_path = os.path.join(directory, f"{sim_id}_90000.hdf5")
        f_fof_path = os.path.join(directory, f"{sim_id}_90000_fof_0.0020_0000.hdf5")
        
        if not os.path.exists(f_phys_path) or not os.path.exists(f_fof_path):
            continue

        try:
            # --- PARSE FILENAME METADATA ---
            parts = sim_id.split('_')
            material_id = f"{parts[0]}_{parts[1]}"  
            r_p_target = float(parts[4].replace('r', '')) / 10.0 
            v_inf = float(parts[5].replace('v', '')) / 10.0      

            # --- PROCESS INITIAL CONDITIONS ---
            with h5py.File(os.path.join(directory, init_name), "r") as f_i:
                # Wrapped in get_val() to prevent scalar indexing errors
                u_l = float(get_val(f_i["Units"].attrs["Unit length in cgs (U_L)"])) * 1e-2
                u_t = float(get_val(f_i["Units"].attrs["Unit time in cgs (U_t)"]))
                u_v = u_l / u_t
                u_m = float(get_val(f_i["Units"].attrs.get("Unit mass in cgs (U_M)", 1.0))) * 1e-3
                
                init_pos = f_i["PartType0/Coordinates"][()] * u_l
                init_vel = f_i["PartType0/Velocities"][()] * u_v
                init_mass = get_masses_exacting(f_i) * u_m
                
                total_init_mass = np.sum(init_mass)
                spin_hr, spin_deg = calculate_initial_spin(init_pos, init_vel, init_mass)

            # --- PROCESS FINAL OUTCOMES (90ks) ---
            with h5py.File(f_phys_path, "r") as f_p:
                box = float(get_val(f_p["Header"].attrs["BoxSize"])) * u_l
                p_ids = f_p["PartType0/ParticleIDs"][()]
                pos = (f_p["PartType0/Coordinates"][()] * u_l) - (0.5 * box)
                vel = f_p["PartType0/Velocities"][()] * u_v
                mass = get_masses_exacting(f_p) * u_m
                
                df_phys = pd.DataFrame({
                    "p_id": p_ids, "m": mass,
                    "x": pos[:,0], "y": pos[:,1], "z": pos[:,2],
                    "vx": vel[:,0], "vy": vel[:,1], "vz": vel[:,2]
                })

            with h5py.File(f_fof_path, "r") as f_f:
                f_ids = f_f["PartType0/ParticleIDs"][()]
                g_ids = f_f["PartType0/FOFGroupIDs"][()]
                df_fof = pd.DataFrame({"p_id": f_ids, "g_id": g_ids})

            # Exact Alignment
            df = pd.merge(df_phys, df_fof, on="p_id")
            bound_df = df[(df["g_id"] >= 0) & (df["g_id"] < UNBOUND_ID)]
            
            def calc_com(g):
                tm = g["m"].sum()
                return pd.Series({
                    "m_kg": tm,
                    "cx": (g["x"]*g["m"]).sum()/tm, "cy": (g["y"]*g["m"]).sum()/tm, "cz": (g["z"]*g["m"]).sum()/tm,
                    "cvx": (g["vx"]*g["m"]).sum()/tm, "cvy": (g["vy"]*g["m"]).sum()/tm, "cvz": (g["vz"]*g["m"]).sum()/tm
                })

            frags = bound_df.groupby("g_id").apply(calc_com, include_groups=False).reset_index()

            # --- BUILD CATALOG ---
            current_sim_catalog = []
            for _, f in frags.iterrows():
                a, e, i, rp, ra = calculate_orbital_elements(np.array([f.cx, f.cy, f.cz]), np.array([f.cvx, f.cvy, f.cvz]))
                frag_data = {
                    "sim_id": sim_id,
                    "fragment_id": int(f.g_id),
                    "fragment_mass_kg": f.m_kg,
                    "mass_ratio": f.m_kg / total_init_mass if total_init_mass > 0 else 0,
                    "semi_major_axis_a": a,
                    "eccentricity_e": e,
                    "inclination_i": i,
                    "apoapsis_Rm": ra / R_MARS,
                    "periapsis_Rm": rp / R_MARS
                }
                current_sim_catalog.append(frag_data)
                catalog_list.append(frag_data)

            # --- CALCULATE DERIVED SUMMARY METRICS ---
            bound_mass = bound_df["m"].sum()
            bound_fraction = bound_mass / total_init_mass if total_init_mass > 0 else 0
            
            valid_frags = [f for f in current_sim_catalog if f["eccentricity_e"] < 1]
            if valid_frags:
                avg_apo = np.mean([f["apoapsis_Rm"] for f in valid_frags])
                spread_rm = np.max([f["apoapsis_Rm"] for f in valid_frags]) - np.min([f["periapsis_Rm"] for f in valid_frags])
                debris_spread_km = spread_rm * (R_MARS / 1000.0)
            else:
                avg_apo, debris_spread_km = 0.0, 0.0

            # --- BUILD SUMMARY ---
            summary_list.append({
                "sim_id": sim_id,
                "n_particles": len(df_phys),
                "asteroid_mass_kg": total_init_mass,
                "r_p_target_Rm": r_p_target,
                "v_inf_kms": v_inf,
                "spin_period_hr": spin_hr,
                "spin_orientation_angle": spin_deg,
                "material_id": material_id,
                "total_bound_mass_kg": bound_mass,
                "bound_mass_fraction": bound_fraction,
                "largest_remnant_mass_kg": frags["m_kg"].max() if not frags.empty else 0,
                "fragment_count": len(frags),
                "avg_apoapsis_Rm": avg_apo,
                "debris_spread_km": debris_spread_km
            })

        except Exception as e:
            import traceback
            print(f"Error in {sim_id}: {e}")
            traceback.print_exc()

    # Export
    pd.DataFrame(summary_list).to_csv(os.path.join(OUTPUT_DIR, "parameter_study_summary.csv"), index=False)
    pd.DataFrame(catalog_list).to_csv(os.path.join(OUTPUT_DIR, "fragment_catalog.csv"), index=False)
    print(f"Extraction complete. Results in {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_simulation("snapshots")