// Deterministic unit tests for KalmanFilter4D.
//
// Self-contained: no external test framework dependency. Each CHECK records a
// failure and the process exits non-zero if any assertion fails, so the test
// integrates cleanly with CTest / add_test().

#include "kalman_filter.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>

static int g_failures = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        std::cout << "[ PASS ] " << name << "\n";
    } else {
        std::cout << "[ FAIL ] " << name << "\n";
        ++g_failures;
    }
}

// Symmetric positive semi-definite check for a 4x4 covariance matrix.
static bool is_symmetric_psd(const Eigen::Matrix4d& M, double tol = 1e-9) {
    // Symmetry
    if ((M - M.transpose()).cwiseAbs().maxCoeff() > 1e-6) return false;
    // PSD: all eigenvalues >= -tol (self-adjoint solver for real symmetric)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(M);
    return es.eigenvalues().minCoeff() >= -tol;
}

// ---------------------------------------------------------------------------
// Test 1: Initial covariance is 10*I (regression for the setIdentity()*10 bug).
// ---------------------------------------------------------------------------
static void test_initial_state() {
    KalmanFilter4D kf;
    check(!kf.isInitialized(), "uninitialized filter reports not initialized");

    // predict/update before init() must be safe no-ops.
    kf.predict(0.1, 1.0);
    kf.update(Eigen::Vector2d(1.0, 2.0), 0.1);
    check(!kf.isInitialized(), "predict/update before init() are no-ops");

    kf.init(Eigen::Vector2d(3.0, 4.0));
    Eigen::Vector4d s = kf.getState();
    check(std::abs(s(0) - 3.0) < 1e-12 && std::abs(s(1) - 4.0) < 1e-12,
          "init() seeds position from measurement");
    check(std::abs(s(2)) < 1e-12 && std::abs(s(3)) < 1e-12,
          "init() seeds zero velocity");
}

// ---------------------------------------------------------------------------
// Test 2: Zero-noise identity. With Q=0 and a perfect measurement (R~0), the
// filter must lock exactly onto the measured position.
// ---------------------------------------------------------------------------
static void test_zero_noise_lock() {
    KalmanFilter4D kf;
    kf.init(Eigen::Vector2d(0.0, 0.0));

    Eigen::Vector2d z(5.0, -3.0);
    for (int i = 0; i < 50; ++i) {
        kf.predict(0.1, 0.0);
        kf.update(z, 1e-9);
    }
    Eigen::Vector4d s = kf.getState();
    check(std::abs(s(0) - 5.0) < 1e-3 && std::abs(s(1) - (-3.0)) < 1e-3,
          "zero-noise filter converges onto a static measurement");
}

// ---------------------------------------------------------------------------
// Test 3: Covariance stays symmetric PSD across many predict/update cycles
// (validates the Joseph-form update).
// ---------------------------------------------------------------------------
static void test_covariance_psd() {
    KalmanFilter4D kf;
    kf.init(Eigen::Vector2d(0.0, 0.0));

    std::mt19937 gen(1234);                       // fixed seed -> reproducible
    std::normal_distribution<double> noise(0.0, 0.5);

    double tx = 0.0, ty = 0.0, vx = 12.0, vy = 1.0, dt = 0.1;
    bool all_psd = true;

    for (int i = 0; i < 200; ++i) {
        tx += vx * dt;
        ty += vy * dt;
        kf.predict(dt, 2.0);
        kf.update(Eigen::Vector2d(tx + noise(gen), ty + noise(gen)), 0.25);
        if (!is_symmetric_psd(kf.getCovariance())) {
            all_psd = false;
            break;
        }
    }
    check(all_psd, "covariance remains symmetric PSD over 200 cycles");
}

// ---------------------------------------------------------------------------
// Test 4: Constant-velocity convergence. The filter should track a constant
// velocity target and recover the true velocity within a tolerance.
// ---------------------------------------------------------------------------
static void test_velocity_convergence() {
    KalmanFilter4D kf;

    std::mt19937 gen(42);                          // fixed seed -> reproducible
    std::normal_distribution<double> noise(0.0, 0.3);

    double tx = 5.0, ty = 2.0, vx = 15.0, vy = 0.0, dt = 0.1;

    for (int i = 0; i < 60; ++i) {
        tx += vx * dt;
        ty += vy * dt;
        Eigen::Vector2d z(tx + noise(gen), ty + noise(gen));
        if (!kf.isInitialized()) {
            kf.init(z);
        } else {
            kf.predict(dt, 1.0);
            kf.update(z, 0.09);
        }
    }
    Eigen::Vector4d s = kf.getState();
    check(std::abs(s(2) - vx) < 1.5,
          "estimated vx converges to true velocity (+/- 1.5 m/s)");
    check(std::abs(s(0) - tx) < 1.0,
          "estimated position tracks ground truth (+/- 1.0 m)");
}

// ---------------------------------------------------------------------------
// Test 5: Occlusion coast. After skipping updates for N steps, the position
// error must stay bounded by velocity-prediction error.
// ---------------------------------------------------------------------------
static void test_occlusion_coast() {
    KalmanFilter4D kf;

    std::mt19937 gen(7);                            // fixed seed -> reproducible
    std::normal_distribution<double> noise(0.0, 0.2);

    double tx = 0.0, ty = 0.0, vx = 10.0, vy = 0.0, dt = 0.1;

    // Lock on with 30 normal steps.
    for (int i = 0; i < 30; ++i) {
        tx += vx * dt;
        ty += vy * dt;
        Eigen::Vector2d z(tx + noise(gen), ty + noise(gen));
        if (!kf.isInitialized()) kf.init(z);
        else { kf.predict(dt, 1.0); kf.update(z, 0.04); }
    }

    // 5-step occlusion: predict only, no measurement updates.
    for (int i = 0; i < 5; ++i) {
        tx += vx * dt;
        ty += vy * dt;
        kf.predict(dt, 1.0);
    }
    Eigen::Vector4d s = kf.getState();
    double pos_err = std::abs(s(0) - tx);
    check(pos_err < 1.0, "occluded coast keeps position error bounded (< 1.0 m)");
}

int main() {
    std::cout << "Running KalmanFilter4D unit tests...\n";
    test_initial_state();
    test_zero_noise_lock();
    test_covariance_psd();
    test_velocity_convergence();
    test_occlusion_coast();

    if (g_failures == 0) {
        std::cout << "\nAll tests passed.\n";
        return 0;
    }
    std::cout << "\n" << g_failures << " test(s) FAILED.\n";
    return 1;
}
