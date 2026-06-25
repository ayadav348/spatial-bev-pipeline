#pragma once
#include <Eigen/Dense>

class KalmanFilter4D {
private:
    Eigen::Vector4d x_; // State: [X, Y, X_dot, Y_dot]^T
    Eigen::Matrix4d F_; // State transition
    Eigen::Matrix4d P_; // State covariance
    Eigen::Matrix4d Q_; // Process noise covariance
    Eigen::Matrix<double, 2, 4> H_; // Measurement matrix
    Eigen::Matrix2d R_; // Measurement noise covariance

    bool is_initialized_ = false;

public:
    KalmanFilter4D();

    void init(const Eigen::Vector2d& initial_pos);
    void predict(double dt, double process_noise);
    void update(const Eigen::Vector2d& measurement, double meas_noise);

    Eigen::Vector4d getState() const;
    Eigen::Matrix4d getCovariance() const;
    bool isInitialized() const;
};
