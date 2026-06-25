#include "kalman_filter.hpp"
#include <cmath>

KalmanFilter4D::KalmanFilter4D() {
    F_.setIdentity();
    H_.setZero();
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    P_.setIdentity() * 10.0; // Initial state uncertainty bound
}

void KalmanFilter4D::init(const Eigen::Vector2d& initial_pos) {
    x_ << initial_pos(0), initial_pos(1), 0.0, 0.0;
    is_initialized_ = true;
}

void KalmanFilter4D::predict(double dt, double process_noise) {
    if (!is_initialized_) return;

    // Inject step-time into State Transition
    F_(0, 2) = dt;
    F_(1, 3) = dt;

    // Piecewise continuous white noise model for Q
    double q = process_noise;
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;

    Q_ << dt3/3.0,   0.0, dt2/2.0,   0.0,
    0.0, dt3/3.0,   0.0, dt2/2.0,
    dt2/2.0,   0.0,      dt,   0.0,
    0.0, dt2/2.0,   0.0,      dt;
    Q_ *= q;

    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter4D::update(const Eigen::Vector2d& measurement, double meas_noise) {
    if (!is_initialized_) return;

    R_ = Eigen::Matrix2d::Identity() * meas_noise;

    Eigen::Vector2d y = measurement - H_ * x_; // Innovation residual
    Eigen::Matrix2d S = H_ * P_ * H_.transpose() + R_; // Innovation covariance
    Eigen::Matrix<double, 4, 2> K = P_ * H_.transpose() * S.inverse(); // Kalman Gain

    x_ = x_ + K * y;
    Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
    P_ = (I - K * H_) * P_;
}

Eigen::Vector4d KalmanFilter4D::getState() const { return x_; }
bool KalmanFilter4D::isInitialized() const { return is_initialized_; }
