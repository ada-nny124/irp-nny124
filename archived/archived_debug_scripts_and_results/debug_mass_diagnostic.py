import h5py

# Point this to your actual physical snapshot file
file_path = "snapshots/Ma_xp_A2000_n70_r16_v00_90000.hdf5"

with h5py.File(file_path, "r") as f:
    print(f"--- Searching for Mass in {file_path} ---")
    
    def find_mass_keys(name, obj):
        # Look for 'Mass' in dataset names or attribute keys
        if "mass" in name.lower():
            print(f"Dataset found: {name}")
        
        for key in obj.attrs.keys():
            if "mass" in key.lower():
                print(f"Attribute found in '{name}': {key} = {obj.attrs[key]}")

    f.visititems(find_mass_keys)
    
    # Also specifically check the Header and Units
    if "Header" in f:
        print("\n--- Header Attributes ---")
        for k, v in f["Header"].attrs.items():
            print(f"{k}: {v}")
            
    if "Units" in f:
        print("\n--- Units Attributes ---")
        for k, v in f["Units"].attrs.items():
            print(f"{k}: {v}")