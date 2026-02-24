import h5py

UKDALE_PATH = "data/ukdale/ukdale.h5"

with h5py.File(UKDALE_PATH, "r") as f:
    print("Top-level keys:")
    for key in f.keys():
        print(" -", key)

    print("\nKeys inside building1:")
    for key in f["building1"].keys():
        print(" -", key)

    elec = f["building1"]["elec"]

    print("\nMeters in building1/elec:")
    for key in elec.keys():
        print(" -", key)

    # Inspect one meter in detail
    meter1 = elec[list(elec.keys())[0]]

    print("\nKeys inside first meter:")
    for key in meter1.keys():
        print(" -", key)
