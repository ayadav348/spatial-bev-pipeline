#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

enum class DatasetType {
    WAYMO,
    NUSCENES,
    STANDARD_OPENCV
};

struct CameraConfig {
    DatasetType type;
    double fx;
    double fy;
    double cx;
    double cy;
    Eigen::Matrix4d E_cam2veh;
};

namespace SpatialMath {
    // A universal projection function that swaps vector axes based on the dataset origin
    inline Eigen::Vector2i project_3d_to_2d(const Eigen::Vector3d& P_cam,
                                            double fx, double fy, double cx, double cy,
                                            DatasetType type)
    {
        double depth = 0.0;
        double u_normalized = 0.0;
        double v_normalized = 0.0;

        switch (type) {
            case DatasetType::WAYMO:
                // Waymo: X is forward down lens barrel, Y is left, Z is up
                depth = P_cam.x();
                if (depth <= 0.1) return Eigen::Vector2i(-1, -1);
                u_normalized = -P_cam.y() / depth;
            v_normalized = -P_cam.z() / depth;
            break;

            case DatasetType::NUSCENES:
            case DatasetType::STANDARD_OPENCV:
                // Standard CV / nuScenes: Z is forward depth, X is right, Y is down
                depth = P_cam.z();
                if (depth <= 0.1) return Eigen::Vector2i(-1, -1);
                u_normalized = P_cam.x() / depth;
            v_normalized = P_cam.y() / depth;
            break;
        }

        // Apply pinhole calibration map parameters
        int u = static_cast<int>(cx + fx * u_normalized);
        int v = static_cast<int>(cy + fy * v_normalized);

        return Eigen::Vector2i(u, v);
    }
}

namespace Config {
    // Fast inline string token parser to bypass heavy external library link chains
    inline CameraConfig load_runtime_geometry(const std::string& filepath) {
        CameraConfig config;
        config.type = DatasetType::STANDARD_OPENCV; // Safe Default

        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "⚠️ Warning: Runtime geometry configuration missing. Defaulting to standard tracking values." << std::endl;
            return config;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();

        // Parse Provider Flag
        if (content.find("\"dataset_type\": \"WAYMO\"") != std::string::npos) config.type = DatasetType::WAYMO;
        if (content.find("\"dataset_type\": \"NUSCENES\"") != std::string::npos) config.type = DatasetType::NUSCENES;

        // Helper closures to extract floating-point parameters out of simple tokens
        auto extract_val = [&](const std::string& key) -> double {
            size_t pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0.0;
            size_t colon = content.find(":", pos);
            size_t comma = content.find(",", colon);
            return std::stod(content.substr(colon + 1, comma - colon - 1));
        };

        config.fx = extract_val("fx");
        config.fy = extract_val("fy");
        config.cx = extract_val("cx");
        config.cy = extract_val("cy");

        // Parse flat transform array directly into Eigen 4x4 matrix block
        size_t matrix_start = content.find("[");
        // Simple sequential number extraction scanner
        size_t current_pos = matrix_start;
        config.E_cam2veh = Eigen::Matrix4d::Identity();

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                while (current_pos < content.size() &&
                    content[current_pos] != '-' &&
                    content[current_pos] != '.' &&
                    (content[current_pos] < '0' || content[current_pos] > '9')) {
                    current_pos++;
                    }
                    if (current_pos >= content.size()) break;
                    size_t next_char;
                config.E_cam2veh(i, j) = std::stod(content.substr(current_pos), &next_char);
                current_pos += next_char;
            }
        }
        return config;
    }
}

#endif // CONFIG_HPP
