import h5py
import numpy as np

fof_path = "snapshots/Ma_xp_A2000_n70_r16_v00_90000_fof_0.0020_0000.hdf5"
with h5py.File(fof_path, "r") as f:
    # Let's find where the IDs are and what's in them
    def print_ids(name, obj):
        if isinstance(obj, h5py.Dataset) and ("GroupID" in name or "id" in name.lower()):
            ids = obj[()]
            print(f"Found dataset: {name}")
            print(f"  Min ID: {np.min(ids)}")
            print(f"  Max ID: {np.max(ids)}")
            print(f"  Particles with ID >= 0: {np.sum(ids >= 0)}")

    f.visititems(print_ids)