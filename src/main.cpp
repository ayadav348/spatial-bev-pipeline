#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <numeric>
#include <random>
#include <iomanip>
#include "config.hpp"
#include "kalman_filter.hpp"

// Global Look-Up Tables for the optimized method
cv::Mat map_x, map_y;

// ============================================================================
// METHOD 1: YOUR OLD METHOD (Looping through every pixel & calculating math)
// ============================================================================
cv::Mat run_old_loop_projection(const cv::Mat& src_frame, const CameraConfig& cam,
                                int bev_width, int bev_height,
                                double min_x, double max_x, double min_y, double max_y)
{
    cv::Mat bev_map = cv::Mat::zeros(bev_height, bev_width, CV_8UC3);

    Eigen::Matrix4d E_veh2cam = cam.E_cam2veh.inverse();
    Eigen::Matrix3d R = E_veh2cam.block<3,3>(0,0);
    Eigen::Vector3d T = E_veh2cam.block<3,1>(0,3);

    for (int v_bev = 0; v_bev < bev_height; ++v_bev) {
        for (int u_bev = 0; u_bev < bev_width; ++u_bev) {

            double X_world = max_x - (static_cast<double>(v_bev) / bev_height) * (max_x - min_x);
            double Y_world = min_y + (static_cast<double>(u_bev) / bev_width) * (max_y - min_y);
            double Z_world = 0.0;

            Eigen::Vector3d P_world(X_world, Y_world, Z_world);
            Eigen::Vector3d P_cam = R * P_world + T;

            if (P_cam.x() > 0.1) {
                Eigen::Vector2i pixel = SpatialMath::project_3d_to_2d(
                    P_cam, cam.fx, cam.fy, cam.cx, cam.cy, cam.type
                );

                int u_img = pixel.x();
                int v_img = pixel.y();

                if (u_img >= 0 && u_img < src_frame.cols && v_img >= 0 && v_img < src_frame.rows) {
                    bev_map.at<cv::Vec3b>(v_bev, u_bev) = src_frame.at<cv::Vec3b>(v_img, u_img);
                }
            }
        }
    }
    return bev_map;
}

// ============================================================================
// METHOD 2: YOUR NEW METHOD (Pre-computed Lookup Table Setup)
// ============================================================================
void initialize_spatial_lut(const CameraConfig& cam, int bev_width, int bev_height,
                            double min_x, double max_x, double min_y, double max_y,
                            int img_cols, int img_rows)
{
    map_x.create(bev_height, bev_width, CV_32FC1);
    map_y.create(bev_height, bev_width, CV_32FC1);

    Eigen::Matrix4d E_veh2cam = cam.E_cam2veh.inverse();
    Eigen::Matrix3d R = E_veh2cam.block<3,3>(0,0);
    Eigen::Vector3d T = E_veh2cam.block<3,1>(0,3);

    for (int v_bev = 0; v_bev < bev_height; ++v_bev) {
        for (int u_bev = 0; u_bev < bev_width; ++u_bev) {
            double X_world = max_x - (static_cast<double>(v_bev) / bev_height) * (max_x - min_x);
            double Y_world = min_y + (static_cast<double>(u_bev) / bev_width) * (max_y - min_y);
            double Z_world = 0.0;

            Eigen::Vector3d P_world(X_world, Y_world, Z_world);
            Eigen::Vector3d P_cam = R * P_world + T;

            if (P_cam.x() > 0.1) {
                Eigen::Vector2i pixel = SpatialMath::project_3d_to_2d(
                    P_cam, cam.fx, cam.fy, cam.cx, cam.cy, cam.type
                );

                if (pixel.x() >= 0 && pixel.x() < img_cols && pixel.y() >= 0 && pixel.y() < img_rows) {
                    map_x.at<float>(v_bev, u_bev) = static_cast<float>(pixel.x());
                    map_y.at<float>(v_bev, u_bev) = static_cast<float>(pixel.y());
                    continue;
                }
            }
            map_x.at<float>(v_bev, u_bev) = -1.0f;
            map_y.at<float>(v_bev, u_bev) = -1.0f;
        }
    }
}

// ============================================================================
// MAIN RUNTIME & VERIFICATION PIPELINE
// ============================================================================
int main() {
    std::cout << "📋 Starting Real-Time Spatial BEV Integration Pipeline..." << std::endl;

    CameraConfig cam = Config::load_runtime_geometry("config/runtime_geometry.json");

    int bev_width = 400;
    int bev_height = 500;
    double min_x = 5.0,  max_x = 50.0;
    double min_y = -20.0, max_y = 20.0;
    int img_width = 1920;
    int img_height = 1280;

    // Generate simulated camera input frame
    cv::Mat simulated_input_frame = cv::Mat::zeros(img_height, img_width, CV_8UC3);
    simulated_input_frame.setTo(cv::Vec3b(40, 30, 30)); // Dark background

    // ========================================================================
    // [UNCOMMENT TO RE-RUN OLD SIDE-BY-SIDE PERFORMANCE PROFILING]
    // ========================================================================
    /*
     *   const int ITERATIONS = 10;
     *   std::vector<double> old_times_ms;
     *   std::vector<double> new_times_us;
     *
     *   std::cout << "🏃 Running baseline loop-based method (" << ITERATIONS << " passes)..." << std::endl;
     *   for (int i = 0; i < ITERATIONS; ++i) {
     *       auto start = std::chrono::high_resolution_clock::now();
     *       cv::Mat res = run_old_loop_projection(simulated_input_frame, cam, bev_width, bev_height, min_x, max_x, min_y, max_y);
     *       auto end = std::chrono::high_resolution_clock::now();
     *       double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
     *       old_times_ms.push_back(elapsed_ms);
}
*/

    // --- Production Implementation: LUT Initialization ---
    std::cout << "⚙️ Pre-computing Spatial Look-Up Tables..." << std::endl;
    initialize_spatial_lut(cam, bev_width, bev_height, min_x, max_x, min_y, max_y, img_width, img_height);

    // --- Production Implementation: Fast Real-Time Warping ---
    std::cout << "⚡ Executing high-throughput spatial remap..." << std::endl;
    cv::Mat final_bev_map;
    auto start_remap = std::chrono::high_resolution_clock::now();
    cv::remap(simulated_input_frame, final_bev_map, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    auto end_remap = std::chrono::high_resolution_clock::now();
    double remap_us = std::chrono::duration<double, std::micro>(end_remap - start_remap).count();
    std::cout << "🚀 Latency achieved: " << remap_us << " µs" << std::endl;

    /*
     *   // [UNCOMMENT TO CALCULATE SPEEDUP METRICS WITH ACCUMULATORS]
     *   double avg_old_ms = std::accumulate(old_times_ms.begin(), old_times_ms.end(), 0.0) / ITERATIONS;
     *   double avg_new_us = std::accumulate(new_times_us.begin(), new_times_us.end(), 0.0) / ITERATIONS;
     *   double avg_new_ms = avg_new_us / 1000.0;
     *   double speedup_factor = avg_old_ms / avg_new_ms;
     *   // ... Output printing statements ...
     */

    // ========================================================================
    // PHASE 4 INTEGRATION: TEMPORAL STATE ESTIMATION VERIFICATION DRILL
    // ========================================================================
    std::cout << "\n🔮 Initializing Temporal Verification (4D Kalman Filter)..." << std::endl;
    KalmanFilter4D kf;

    double dt = 0.1;             // 10 Hz simulated step frequency
    double process_noise = 4.0; // Model dynamics covariance
    double meas_noise = 0.4;     // Simulated detector variance bounds

    // Physical vehicle kinematics trajectory simulation passed down to Ego Center
    double true_x = 5.0;
    double true_y = 2.0;
    double true_v_x = 15.0;
    double true_v_y = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise(0.0, std::sqrt(meas_noise));

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "========================================================================\n";
    std::cout << " TIME   │     TRUE POSITION     │    NOISY MEASUREMENT  │    KALMAN FILTER      \n";
    std::cout << " (s)    │   X (m)   │   Y (m)   │   X (m)   │   Y (m)   │   X (m)   │   Y (m)   \n";
    std::cout << "========================================================================\n";

    for (int step = 0; step < 20; ++step) {
        double t = step * dt;

        true_x += true_v_x * dt;
        true_y += true_v_y * dt;

        Eigen::Vector2d measurement;
        measurement << true_x + noise(gen), true_y + noise(gen);

        if (!kf.isInitialized()) {
            kf.init(measurement);
        } else {
            kf.predict(dt, process_noise);

            // Simulate visual dropout / sensor occlusion between frames 10 and 14
            if (step >= 10 && step <= 14) {
                // Skip update phase - rely entirely on motion prediction kinematics!
            } else {
                kf.update(measurement, meas_noise);
            }
        }

        Eigen::Vector4d state = kf.getState();

        std::cout << std::setw(6) << t << "  │ "
        << std::setw(9) << true_x << " │ " << std::setw(9) << true_y << " │ "
        << std::setw(9) << measurement(0) << " │ " << std::setw(9) << measurement(1) << " │ ";

        if (step >= 10 && step <= 14) {
            std::cout << std::setw(9) << state(0) << "*│ " << std::setw(9) << state(1) << "* [OCCLUSION]\n";
        } else {
            std::cout << std::setw(9) << state(0) << " │ " << std::setw(9) << state(1) << "\n";
        }
    }
    std::cout << "========================================================================\n";
    std::cout << "(* Asterisk indicates pure velocity-predicted estimations during a blind spot)\n" << std::endl;

    cv::imwrite("bev_grid_test.png", final_bev_map);
    return 0;
}
