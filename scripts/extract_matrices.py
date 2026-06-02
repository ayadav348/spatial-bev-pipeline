import pandas as pd
import numpy as np
import json
import os
import pyarrow.parquet as pq  # Explicitly import pyarrow to read schemas cleanly

def detect_dataset(parquet_path):
    """Inspects the schema metadata of the file to identify the dataset provider."""
    # Read just the schema header instead of loading rows
    parquet_file = pq.ParquetFile(parquet_path)
    columns_blob = "".join(parquet_file.schema.names)

    if "CameraCalibrationComponent" in columns_blob or "segment_context_name" in columns_blob:
        return "WAYMO"
    elif "nuscenes" in columns_blob or "sample_token" in columns_blob:
        return "NUSCENES"
    else:
        return "STANDARD_OPENCV"

def main():
    calib_path = "data/raw/validation_camera_calibration_10203656353524179475_7625_000_7645_000.parquet"

    if not os.path.exists(calib_path):
        print(f"❌ Error: Calibration file missing at {calib_path}")
        return

    dataset_type = detect_dataset(calib_path)
    print(f"📡 Schema Inspector Detected Dataset Provider: {dataset_type}")

    df = pd.read_parquet(calib_path)

    # Target our primary front-facing lens configuration block
    # Waymo Node 1 is Front. For unmanaged sets, we default to the first available row.
    if dataset_type == "WAYMO":
        row = df[df['key.camera_name'] == 1].iloc[0]
    else:
        row = df.iloc[0]

    # Pull Intrinsic Parameter Blocks
    if dataset_type == "WAYMO":
        fx = float(row['[CameraCalibrationComponent].intrinsic.f_u'])
        fy = float(row['[CameraCalibrationComponent].intrinsic.f_v'])
        cx = float(row['[CameraCalibrationComponent].intrinsic.c_u'])
        cy = float(row['[CameraCalibrationComponent].intrinsic.c_v'])
        extrinsic_flat = row['[CameraCalibrationComponent].extrinsic.transform']
        E = np.array(extrinsic_flat).reshape(4, 4).tolist()
    else:
        # Standard Fallbacks for OpenCV formats
        fx, fy, cx, cy = 2000.0, 2000.0, 960.0, 640.0
        E = np.eye(4).tolist()

    # Bundle into a standardized JSON packet for our C++ engine runtime
    runtime_config = {
        "dataset_type": dataset_type,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "extrinsic_matrix": E
    }

    os.makedirs("config", exist_ok=True)
    with open("config/runtime_geometry.json", "w") as f:
        json.dump(runtime_config, f, indent=4)

    print("✅ Geometric runtime configuration package exported to config/runtime_geometry.json")

if __name__ == "__main__":
    main()
