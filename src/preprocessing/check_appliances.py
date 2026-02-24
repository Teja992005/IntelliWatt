import pandas as pd

H5_PATH = "data/ukdale/ukdale.h5"

def main():
    store = pd.HDFStore(H5_PATH)

    print("\nAvailable keys in UK-DALE HDF5:\n")
    for key in store.keys():
        print(key)

    print("\nChecking kettle data...\n")

    kettle_key = "/building1/kettle"

    if kettle_key in store.keys():
        kettle_df = store[kettle_key]
        print("✅ Kettle data found!")
        print("Shape:", kettle_df.shape)
        print("\nSample rows:")
        print(kettle_df.head())
    else:
        print("❌ Kettle data NOT found in this house.")

    store.close()

if __name__ == "__main__":
    main()
