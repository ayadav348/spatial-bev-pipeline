#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <numeric>
#include "config.hpp"

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
// MAIN RUNTIME PROFILER
// ============================================================================
int main() {
    std::cout << "📋 Starting Side-by-Side Performance Profiling Harness..." << std::endl;

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

    const int ITERATIONS = 10;
    std::vector<double> old_times_ms;
    std::vector<double> new_times_us;

    // --- Benchmark Method 1 (Old Dynamic Execution) ---
    std::cout << "🏃 Running baseline loop-based method (" << ITERATIONS << " passes)..." << std::endl;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat res = run_old_loop_projection(simulated_input_frame, cam, bev_width, bev_height, min_x, max_x, min_y, max_y);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        old_times_ms.push_back(elapsed_ms);
    }

    // --- Benchmark Method 2 (New LUT-cached Execution) ---
    std::cout << "⚙️ Pre-computing Look-Up Tables (Excluded from streaming metrics)..." << std::endl;
    initialize_spatial_lut(cam, bev_width, bev_height, min_x, max_x, min_y, max_y, img_width, img_height);

    std::cout << "⚡ Running optimized cv::remap method (" << ITERATIONS << " passes)..." << std::endl;
    cv::Mat final_bev_map;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::remap(simulated_input_frame, final_bev_map, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        new_times_us.push_back(elapsed_us);
    }

    // Compute statistical averages
    double avg_old_ms = std::accumulate(old_times_ms.begin(), old_times_ms.end(), 0.0) / ITERATIONS;
    double avg_new_us = std::accumulate(new_times_us.begin(), new_times_us.end(), 0.0) / ITERATIONS;
    double avg_new_ms = avg_new_us / 1000.0; // convert to ms for direct comparison

    double speedup_factor = avg_old_ms / avg_new_ms;

    // --- REPORT THE RESUME METRICS ---
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "📊 RESUME PERFORMANCE METRICS GENERATED SUCCESSFULLY" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "❌ Baseline Loop Processing Latency: " << avg_old_ms << " ms" << std::endl;
    std::cout << "🚀 Optimized LUT Processing Latency: " << avg_new_us << " µs (" << avg_new_ms << " ms)" << std::endl;
    std::cout << "📈 Total Throughput Acceleration:    " << speedup_factor << "x FASTER" << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    cv::imwrite("bev_grid_test.png", final_bev_map);
    return 0;
}
