#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "config.hpp"

int main() {
    std::cout << "🚀 Initializing Adaptive Spatial BEV Projection Engine..." << std::endl;

    // 1. Ingest parsed metadata attributes generated from the Python parser layer
    CameraConfig cam = Config::load_runtime_geometry("config/runtime_geometry.json");

    // Convert Camera-to-Vehicle mapping into Vehicle-to-Camera tracking space
    Eigen::Matrix4d E_veh2cam = cam.E_cam2veh.inverse();
    Eigen::Matrix3d R = E_veh2cam.block<3,3>(0,0);
    Eigen::Vector3d T = E_veh2cam.block<3,1>(0,3);

    // 2. Define our Output BEV Grid Specs (Real-world Meters)
    double min_x = 5.0,  max_x = 50.0;
    double min_y = -20.0, max_y = 20.0;

    int bev_width = 400;
    int bev_height = 500;
    cv::Mat bev_map = cv::Mat::zeros(bev_height, bev_width, CV_8UC3);

    std::cout << "📐 Computing Multi-Vendor Inverse Perspective Mapping arrays..." << std::endl;

    for (int v_bev = 0; v_bev < bev_height; ++v_bev) {
        for (int u_bev = 0; u_bev < bev_width; ++u_bev) {

            // Map pixel coordinate (u_bev, v_bev) to physical world ground coordinates
            double X_world = max_x - (static_cast<double>(v_bev) / bev_height) * (max_x - min_x);
            double Y_world = min_y + (static_cast<double>(u_bev) / bev_width) * (max_y - min_y);
            double Z_world = 0.0; // Flat-ground assumption

            // Reconstruct 3D vehicle vector and project to Camera Frame space
            Eigen::Vector3d P_world(X_world, Y_world, Z_world);
            Eigen::Vector3d P_cam = R * P_world + T;

            // Route point projection using our structural layout switchboard
            Eigen::Vector2i pixel = SpatialMath::project_3d_to_2d(
                P_cam, cam.fx, cam.fy, cam.cx, cam.cy, cam.type
            );

            int u_img = pixel.x();
            int v_img = pixel.y();

            // Check boundaries: Ensure the point falls within the physical lens view dimensions
            // Hardcoding standard high-res boundaries for tracking validation limits
            if (u_img >= 0 && u_img < 1920 && v_img >= 0 && v_img < 1280) {

                // Base background environment color
                bev_map.at<cv::Vec3b>(v_bev, u_bev) = cv::Vec3b(40, 30, 30);

                // Draw distance cross-sections every 10 meters forward
                if (std::abs(std::fmod(X_world, 10.0)) < 0.5 || std::abs(std::fmod(X_world, 10.0)) > 9.5) {
                    bev_map.at<cv::Vec3b>(v_bev, u_bev) = cv::Vec3b(0, 255, 255); // Yellow Bars
                }

                // Draw lane markers at Y = -3.5m and Y = +3.5m
                if (std::abs(Y_world - 3.5) < 0.3 || std::abs(Y_world + 3.5) < 0.3) {
                    bev_map.at<cv::Vec3b>(v_bev, u_bev) = cv::Vec3b(255, 255, 255); // White Lanes
                }
            }
        }
    }

    std::string output_path = "bev_grid_test.png";
    cv::imwrite(output_path, bev_map);
    std::cout << "✅ Matrix transformations complete! Visualized grid saved to: " << output_path << std::endl;

    return 0;
}
