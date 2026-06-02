import pandas as pd
import sys

def main():
    calib_path = "data/raw/validation_camera_calibration_10203656353524179475_7625_000_7645_000.parquet"

    try:
        df = pd.read_parquet(calib_path)
        print("✅ Parquet File Loaded Successfully!")
        print(f"Total Rows (Frames/Sensors): {len(df)}")
        print("\n--- Available Component Columns ---")
        for col in df.columns:
            print(f" - {col}")

    except Exception as e:
        print(f"❌ Error loading file: {e}")

if __name__ == "__main__":
    main()
