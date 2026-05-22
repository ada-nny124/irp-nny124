import h5py
import matplotlib.pyplot as plt
import numpy as np

# Physical constant for plotting
R_EARTH = 6.371e6  # meters

# 1. Open the HDF5 file your custom_ic.py script generated
filename = "impact_v7.0_p15000.hdf5"
print(f"Loading {filename}...")

with h5py.File(filename, "r") as f:
    # Read coordinates and convert from meters to Earth Radii
    coords = f["PartType0/Coordinates"][:] / R_EARTH
    
    # Read material IDs so we can color-code the core and mantle
    mat_ids = f["PartType0/MaterialIDs"][:]

# Separate the X and Y coordinates (we ignore Z for a 2D top-down view)
x = coords[:, 0]
y = coords[:, 1]

# 2. Set up the plot
plt.figure(figsize=(8, 8))

# Plot the Rock Mantle (Material 400) in grey
rock_mask = (mat_ids == 400)
plt.scatter(x[rock_mask], y[rock_mask], s=0.1, color='grey', label='Mantle (Forsterite)')

# Plot the Iron Core (Material 402) in orange
iron_mask = (mat_ids == 402)
plt.scatter(x[iron_mask], y[iron_mask], s=0.1, color='darkorange', label='Core (Iron)')

# 3. SHIFT THE CAMERA
# The SWIFT box is 100 Earth Radii wide, and your planets are at the exact center (50, 50).
# We set the graph limits to look right at them (from 40 to 60).
plt.xlim(40, 60)
plt.ylim(40, 60)

plt.xlabel("X Position (Earth Radii)")
plt.ylabel("Y Position (Earth Radii)")
plt.title("Initial Conditions: Tidal Disruption Setup")
plt.legend(loc="upper right")
plt.grid(True, linestyle=':', alpha=0.6)

# Save and show the image
plt.savefig("fixed_plot.png", dpi=300)
print("Saved plot as fixed_plot.png!")
plt.show()