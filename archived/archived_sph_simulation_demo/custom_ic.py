import numpy as np
import h5py
import woma
import os

# =======================================================================
# 0. PHYSICAL CONSTANTS (SI Units)
# =======================================================================
M_EARTH = 5.9722e24   # kg
R_EARTH = 6.371e6     # m

# =======================================================================
# 1. CONFIGURE ANEOS TABLES
# =======================================================================
path_forsterite = "EoSTables/ANEOS_forsterite_S19.txt"
path_iron = "EoSTables/ANEOS_Fe85Si15_S20.txt"

woma.misc.glob_vars.aneos_forsterite_table = path_forsterite
woma.misc.glob_vars.aneos_iron_table = path_iron

# =======================================================================
# 2. INITIALIZE BASE PLANETS
# =======================================================================
print("Generating base planetary profiles and SPH particles...")

mat_iron = "ANEOS_Fe85Si15"
mat_rock = "ANEOS_forsterite"

# Define Target Planet (0.9 M_earth)
m_target = 0.9 * M_EARTH
target = woma.Planet(
    M = m_target,
    A1_mat_layer = [mat_iron, mat_rock],
    A1_T_rho_type = ["adiabatic", "adiabatic"],
    A1_M_layer = [0.3 * m_target, 0.7 * m_target], 
    T_s = 500.0, 
    P_s = 1e5,
    verbosity = 0 # Silenced standard output to keep your terminal clean
)
target.gen_prof_L2_find_R_R1_given_M1_M2(R_min=0.90*R_EARTH, R_max=1.02*R_EARTH)

# Define Impactor Planet (0.1 M_earth)
m_impactor = 0.1 * M_EARTH
impactor = woma.Planet(
    M = m_impactor,
    A1_mat_layer = [mat_iron, mat_rock],
    A1_T_rho_type = ["adiabatic", "adiabatic"],
    A1_M_layer = [0.3 * m_impactor, 0.7 * m_impactor],
    T_s = 500.0,
    P_s = 1e5,
    verbosity = 0
)
impactor.gen_prof_L2_find_R_R1_given_M1_M2(R_min=0.42*R_EARTH, R_max=0.52*R_EARTH)

# Generate Particle Distributions
total_particles = 100000
target_N = int(total_particles * (target.M / (target.M + impactor.M)))
impactor_N = total_particles - target_N

base_target_particles = woma.ParticlePlanet(target, target_N, N_ngb=48)
base_impactor_particles = woma.ParticlePlanet(impactor, impactor_N, N_ngb=48)

print(f"Planets built: Target R={target.R/R_EARTH:.2f} Re, Impactor R={impactor.R/R_EARTH:.2f} Re")

# =======================================================================
# 3. TRAJECTORY AND HDF5 GENERATION
# =======================================================================
def create_parameterized_ic(v_km_s, p_km, time_to_impact_hours=1.0):
    filename = f"impact_v{v_km_s}_p{p_km}.hdf5"
    print(f"Generating -> {filename}")

    # FIXED: Use A2_ prefixes for 2-Dimensional spatial arrays (Nx3)
    t_pos, t_vel = np.copy(base_target_particles.A2_pos), np.copy(base_target_particles.A2_vel)
    i_pos, i_vel = np.copy(base_impactor_particles.A2_pos), np.copy(base_impactor_particles.A2_vel)

    v_impact, y_offset = v_km_s * 1000.0, p_km * 1000.0
    time_to_impact_sec = time_to_impact_hours * 3600.0

    R_contact = target.R + impactor.R
    contact_dist_sq = R_contact**2 - y_offset**2
    base_x = np.sqrt(contact_dist_sq) if contact_dist_sq > 0 else 0.0
    x_offset = base_x + (v_impact * time_to_impact_sec)

    M_tot = target.M + impactor.M
    rel_pos, rel_vel = np.array([x_offset, y_offset, 0.0]), np.array([-v_impact, 0.0, 0.0])

    t_pos -= (impactor.M / M_tot) * rel_pos
    t_vel -= (impactor.M / M_tot) * rel_vel
    i_pos += (target.M / M_tot) * rel_pos
    i_vel += (target.M / M_tot) * rel_vel

    pos = np.vstack((t_pos, i_pos))
    vel = np.vstack((t_vel, i_vel))
    
    # FIXED: Use A1_ prefixes for 1-Dimensional physical properties
    mass = np.concatenate((base_target_particles.A1_m, base_impactor_particles.A1_m))
    u = np.concatenate((base_target_particles.A1_u, base_impactor_particles.A1_u))
    rho = np.concatenate((base_target_particles.A1_rho, base_impactor_particles.A1_rho))
    h = np.concatenate((base_target_particles.A1_h, base_impactor_particles.A1_h))
    
    # WoMa correctly maps the IDs to 402 and 400 internally, so we just grab them directly
    swift_ids = np.concatenate((base_target_particles.A1_mat_id, base_impactor_particles.A1_mat_id))

    box_size = 100.0 * R_EARTH
    pos += (box_size / 2.0)

    with h5py.File(filename, "w") as f:
        header = f.create_group("Header")
        header.attrs.create("NumPart_ThisFile", [len(pos), 0, 0, 0, 0, 0])
        header.attrs.create("NumPart_Total", [len(pos), 0, 0, 0, 0, 0])
        header.attrs.create("BoxSize", box_size)
        header.attrs.create("Time", 0.0)
        header.attrs.create("NumFilesPerSnapshot", 1)
        header.attrs.create("MassTable", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        part0 = f.create_group("PartType0")
        part0.create_dataset("Coordinates", data=pos, dtype='d')
        part0.create_dataset("Velocities", data=vel, dtype='f')
        part0.create_dataset("Masses", data=mass, dtype='f')
        part0.create_dataset("InternalEnergies", data=u, dtype='f')
        part0.create_dataset("Densities", data=rho, dtype='f') 
        part0.create_dataset("SmoothingLengths", data=h, dtype='f')
        part0.create_dataset("ParticleIDs", data=np.arange(1, len(pos) + 1), dtype='Q')
        part0.create_dataset("MaterialIDs", data=swift_ids, dtype='i')

if __name__ == "__main__":
    if not os.path.exists(path_forsterite) or not os.path.exists(path_iron):
        print(f"Error: ANEOS tables not found at {path_forsterite}")
    else:
        print("Starting IC generation...\n")
        create_parameterized_ic(v_km_s=7.0, p_km=15000)   
        print("\nAll initial conditions generated successfully!")