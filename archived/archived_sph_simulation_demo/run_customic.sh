#!/bin/bash

# Resolution and Filenames
N_label="v7"
SOURCE_IC="impact_v7.0_p15000.hdf5"
TARGET_IC="demo_impact_$N_label.hdf5"

# Ensure the specific initial condition file exists
if [ -e "$SOURCE_IC" ]
then
    echo "Using local file $SOURCE_IC as initial condition..."
    # Copying to the name expected by the .yml file
    cp "$SOURCE_IC" "$TARGET_IC"
else
    echo "Error: $SOURCE_IC not found in the current directory."
    exit 1
fi

# Download equation of state tables if not already present
if [ ! -e ../EoSTables/ANEOS_forsterite_S19.txt ]
then
    echo "Downloading EoT tables..."
    cd ../EoSTables
    ./get_eos_tables.sh
    cd -
fi

# Run SWIFT
# Note: Ensure demo_impact_v7.yml is configured to output snapshots 
# with the basename 'demo_impact_v7'
../../../swift --hydro --self-gravity --threads=28 demo_impact_"$N_label".yml 2>&1 | tee output_"$N_label".txt

# Plot the snapshots
# This will now process demo_impact_v7_0000.hdf5 through 0028.hdf5
python3 plot_snapshots.py