# Spatial BEV Pipeline

Real-time monocular Bird's Eye View (BEV) perception and multi-object tracking pipeline implemented in C++. Implements Inverse Perspective Mapping via camera-calibrated homography projection, LUT-accelerated frame warping, and a custom Eigen-based Kalman Filter for Bayesian state tracking under dynamic occlusion. Validated against the Waymo Open Dataset v2.

**Stack:** C++17, OpenCV, Eigen3, Python (PyArrow/Pandas for dataset tooling)

---

## Goal

Build a high-throughput, real-time pipeline that transforms monocular camera frames into a top-down Bird's Eye View representation of the ground plane around a vehicle, and maintains persistent tracked estimates of detected objects across time — including through sensor occlusion windows — using kinematic prediction.

The pipeline is designed to be dataset-agnostic (Waymo, nuScenes, standard OpenCV), deriving all spatial geometry from real camera calibration files rather than synthetic assumptions.

---

## What Is Currently Implemented

### Phase 1 — Camera Calibration & Geometry Engine (`include/config.hpp`)

- `CameraConfig` struct holds pinhole intrinsics (`fx`, `fy`, `cx`, `cy`) and a 4x4 camera-to-vehicle extrinsic transform (`E_cam2veh`) as an Eigen matrix.
- `DatasetType` enum supports `WAYMO`, `NUSCENES`, and `STANDARD_OPENCV` coordinate conventions.
- `SpatialMath::project_3d_to_2d` implements a universal pinhole projection function that correctly handles axis convention differences between dataset providers:
  - Waymo: X-forward, Y-left, Z-up
  - Standard CV / nuScenes: Z-forward, X-right, Y-down
- `Config::load_runtime_geometry` parses a JSON calibration file at runtime into a `CameraConfig` — no external JSON library dependency; uses a custom inline token scanner.

### Phase 2 — Waymo Open Dataset Tooling (`scripts/`)

- `inspect_waymo.py`: Loads a Waymo v2 Parquet calibration file and prints all available component columns and frame count.
- `extract_matrices.py`: Auto-detects the dataset provider from Parquet schema metadata, extracts front-camera intrinsics (`f_u`, `f_v`, `c_u`, `c_v`) and the full 4x4 extrinsic transform from the `CameraCalibrationComponent`, and exports a standardized `config/runtime_geometry.json` for consumption by the C++ engine.

### Phase 3 — Adaptive Inverse Perspective Mapping Engine (`src/main.cpp`)

Two IPM implementations are present for performance comparison:

**Baseline — Per-Frame Loop Projection (`run_old_loop_projection`):**
- For every BEV output pixel, back-projects the corresponding 3D ground-plane world coordinate through the extrinsic inverse and pinhole model to find its source pixel in the camera image.
- Processes a 400x500 BEV grid at ~425ms per frame.

**Optimized — Pre-computed Look-Up Table (`initialize_spatial_lut` + `cv::remap`):**
- Performs the identical back-projection math once at boot time and stores the result as two float maps (`map_x`, `map_y`).
- All subsequent frames execute via `cv::remap`, reducing per-frame latency to **~3.22ms** — a **132x speedup**, enabling 300+ FPS throughput.
- LUT output is saved to `bev_grid_test.png`.

**BEV Grid Parameters (configurable in `main.cpp`):**
| Parameter | Value |
| :--- | :--- |
| Output resolution | 400 x 500 px |
| Forward range | 5m – 50m |
| Lateral range | -20m – +20m |
| Simulated input | 1920 x 1280 px |

### Phase 4 — 4D Kalman Filter for Temporal State Estimation (`include/kalman_filter.hpp`, `src/kalman_filter.cpp`)

A heap-allocation-free Kalman Filter implemented entirely with fixed-size Eigen matrices — no dynamic memory allocation in the tracking hot path.

**State space:**
- State vector: `x = [X, Y, Ẋ, Ẏ]ᵀ` — 2D BEV position and velocity
- Measurement vector: `z = [X, Y]ᵀ` — derived from homography projection of detected object bottom-center pixel

**Implementation details:**
- Constant-Velocity (CV) kinematic model with piecewise continuous white noise `Q` matrix scaled by `dt`.
- State transition matrix `F` injects the time step `dt` at predict time rather than at construction, supporting variable frame rates.
- `predict(dt, process_noise)` propagates state and covariance forward using `x = Fx`, `P = FPFᵀ + Q`.
- `update(measurement, meas_noise)` computes the Kalman gain, innovation residual, and applies the Joseph-form covariance update `P = (I − KH)P(I − KH)ᵀ + KRKᵀ`, which keeps `P` symmetric positive semi-definite under floating-point rounding.
- Occlusion handling: the update step is simply skipped during sensor dropout frames; the filter coasts on velocity prediction alone.

**Verified performance (10 Hz simulation, 15 m/s target, 500ms occlusion window between t=1.1s–1.5s, fixed RNG seed = 42 for reproducibility):**

| Metric | Value |
| :--- | :--- |
| Tracking lag at t=0.4s (warm-up) | ~16% position lag (9.23 m est. vs 11.00 m true) |
| Final velocity estimate | 14.91 m/s (true: 15.0 m/s) |
| Re-acquisition after occlusion | Converges within 2–3 frames |

> These figures are reproducible: the verification drill in `main.cpp` uses a fixed RNG seed, so running `./bev_warp` reproduces the exact table above.

---

## Build

**Dependencies:** CMake 3.16+, OpenCV, Eigen3 3.4+

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./bev_warp
```

**Run the Kalman filter unit tests:**

```bash
cd build
ctest --output-on-failure
# or run the test binary directly:
./test_kalman
```

**Calibration extraction (requires Python + pyarrow + pandas):**

```bash
python scripts/extract_matrices.py
```

Place the Waymo v2 calibration Parquet file at:
```
data/raw/validation_camera_calibration_<segment>.parquet
```

---

## Performance Summary

| Method | Latency | Throughput |
| :--- | :--- | :--- |
| Baseline loop projection | ~425.5 ms/frame | ~2 FPS |
| LUT + `cv::remap` | ~3.22 ms/frame | ~310 FPS |
| **Speedup** | **132x** | |
