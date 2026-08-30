#include "Sharemem.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {
constexpr int MOTOR_NUM = 16;
constexpr key_t SHM_KEY = 13563267;
constexpr int RIGHT_BEGIN = 8;
constexpr int RIGHT_DOF = 7;
constexpr int CMD_PORT = 5005;
constexpr int FB_PORT = 5006;
// Safety/smoothing parameters.
// Incoming IK targets are NOT dropped just because they are far away.
// Instead, the bridge moves the hardware command toward the latest target
// at a bounded joint-space rate.
constexpr double MAX_COMMAND_RATE_RAD_S = 0.60;       // max commanded joint speed
constexpr double MAX_COMMAND_STEP_RAD = 0.03;         // extra per-packet hard cap
constexpr double MAX_TRACKING_ERROR_RAD = 0.35;       // q_cmd may not run farther than this from measured q
constexpr double ABS_COMMAND_LIMIT_RAD = 2.0;         // final hardware-wide absolute guard

// Coordinate convention:
//   model-space: URDF / Placo / areumii_real_teleop.py
//   hardware-space: Areum2_can shared-memory motor convention
//
// Right wrist_yaw (joint index 5, SHM index 13) is physically reversed
// relative to the URDF/model convention, so flip it at this boundary.
// Keep all safety comparisons and UDP traffic in model-space.
constexpr std::array<double, RIGHT_DOF> RIGHT_JOINT_SIGN = {
    +1.0,  // 0 shoulder_pitch
    +1.0,  // 1 shoulder_roll
    +1.0,  // 2 shoulder_yaw
    +1.0,  // 3 elbow
    +1.0,  // 4 wrist_roll
    +1.0,  // 5 wrist_yaw  <-- reversed on real hardware
    +1.0,  // 6 wrist_pitch
};

double hardware_to_model_q(int i, double q_hw) {
    return RIGHT_JOINT_SIGN[i] * q_hw;
}

double model_to_hardware_q(int i, double q_model) {
    return RIGHT_JOINT_SIGN[i] * q_model;
}

bool parse_csv7(const std::string& s, std::array<double, RIGHT_DOF>& q) {
    std::stringstream ss(s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, ',')) {
        if (i >= RIGHT_DOF) return false;
        try { q[i] = std::stod(item); }
        catch (...) { return false; }
        if (!std::isfinite(q[i])) return false;
        ++i;
    }
    return i == RIGHT_DOF;
}

std::string csv7_from_fb(const Feedback_Param* fb) {
    std::ostringstream os;
    os.precision(12);
    for (int i = 0; i < RIGHT_DOF; ++i) {
        if (i) os << ',';
        os << hardware_to_model_q(
            i, static_cast<double>(fb[RIGHT_BEGIN + i].pos));
    }
    return os.str();
}

bool read_ctrl_consistent(const Control_Shm<MOTOR_NUM>::Control_Struct* shm, Control_param* ctrl) {
    for (int tries = 0; tries < 20; ++tries) if (shm->try_read_ctrl(ctrl)) return true;
    return false;
}

bool read_fb_consistent(const Control_Shm<MOTOR_NUM>::Control_Struct* shm, Feedback_Param* fb) {
    for (int tries = 0; tries < 20; ++tries) if (shm->try_read_fb(fb)) return true;
    return false;
}
}

int main() {
    Control_Shm<MOTOR_NUM> shared(SHM_KEY);
    auto* shm = shared.get();
    Control_param ctrl[MOTOR_NUM]{};
    Feedback_Param fb[MOTOR_NUM]{};

    int rx = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (rx < 0) { perror("socket(rx)"); return 1; }

    sockaddr_in rx_addr{};
    rx_addr.sin_family = AF_INET;
    rx_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    rx_addr.sin_port = htons(CMD_PORT);
    if (::bind(rx, reinterpret_cast<sockaddr*>(&rx_addr), sizeof(rx_addr)) < 0) {
        perror("bind(command)"); return 1;
    }
    int flags = fcntl(rx, F_GETFL, 0);
    fcntl(rx, F_SETFL, flags | O_NONBLOCK);

    int tx = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (tx < 0) { perror("socket(tx)"); return 1; }
    sockaddr_in fb_addr{};
    fb_addr.sin_family = AF_INET;
    fb_addr.sin_port = htons(FB_PORT);
    inet_pton(AF_INET, "127.0.0.1", &fb_addr.sin_addr);

    bool have_command_state = false;
    std::array<double, RIGHT_DOF> q_command_model{};
    auto last_command_update = std::chrono::steady_clock::now();
    auto last_rate_log = std::chrono::steady_clock::time_point{};
    auto next_feedback = std::chrono::steady_clock::now();

    std::cout << "[bridge] SHM key=" << SHM_KEY
              << " right indices=" << RIGHT_BEGIN << ".." << (RIGHT_BEGIN + RIGHT_DOF - 1)
              << "\n[bridge] UDP command 127.0.0.1:" << CMD_PORT
              << " feedback 127.0.0.1:" << FB_PORT
              << "\n[bridge] model<->hardware sign: wrist_yaw(q[5], SHM13) = -1"
              << "\n[bridge] smooth rate limit=" << MAX_COMMAND_RATE_RAD_S
              << " rad/s, tracking gap=" << MAX_TRACKING_ERROR_RAD << " rad"
              << "\n";

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_feedback) {
            if (read_fb_consistent(shm, fb)) {
                const std::string msg = csv7_from_fb(fb);
                ::sendto(tx, msg.data(), msg.size(), 0,
                         reinterpret_cast<const sockaddr*>(&fb_addr), sizeof(fb_addr));
            }
            next_feedback = now + std::chrono::milliseconds(10);
        }

        char buf[512];
        sockaddr_in src{};
        socklen_t src_len = sizeof(src);
        const ssize_t n = ::recvfrom(rx, buf, sizeof(buf) - 1, 0,
                                     reinterpret_cast<sockaddr*>(&src), &src_len);

        if (n > 0) {
            buf[n] = '\0';
            std::array<double, RIGHT_DOF> q{};
            if (!parse_csv7(std::string(buf), q)) {
                std::cerr << "[bridge] malformed command dropped\n";
            } else if (!read_fb_consistent(shm, fb)) {
                std::cerr << "[bridge] no consistent feedback; command dropped\n";
            } else {
                // Reject only clearly invalid/out-of-envelope desired targets.
                bool safe = true;
                for (int i = 0; i < RIGHT_DOF; ++i) {
                    if (!std::isfinite(q[i]) || std::abs(q[i]) > ABS_COMMAND_LIMIT_RAD) {
                        std::cerr << "[bridge] joint " << i << " desired target outside +/-"
                                  << ABS_COMMAND_LIMIT_RAD << " rad; packet dropped\n";
                        safe = false;
                        break;
                    }
                }

                if (safe) {
                    const auto command_now = std::chrono::steady_clock::now();

                    // On the first valid packet, start exactly from the REAL measured
                    // robot pose. This avoids a jump to the first IK target.
                    if (!have_command_state) {
                        for (int i = 0; i < RIGHT_DOF; ++i) {
                            q_command_model[i] = hardware_to_model_q(
                                i, static_cast<double>(fb[RIGHT_BEGIN + i].pos));
                        }
                        have_command_state = true;
                        last_command_update = command_now;
                        std::cout << "[bridge] command state initialized from measured q\n";
                    }

                    double dt = std::chrono::duration<double>(
                        command_now - last_command_update).count();
                    // Protect against timer anomalies / long pauses.
                    if (!std::isfinite(dt) || dt <= 0.0) dt = 0.001;
                    if (dt > 0.05) dt = 0.05;

                    const double rate_step =
                        std::min(MAX_COMMAND_STEP_RAD, MAX_COMMAND_RATE_RAD_S * dt);

                    bool any_rate_limited = false;
                    int first_limited_joint = -1;
                    double first_desired = 0.0;
                    double first_before = 0.0;
                    double first_after = 0.0;

                    for (int i = 0; i < RIGHT_DOF; ++i) {
                        const double measured = hardware_to_model_q(
                            i, static_cast<double>(fb[RIGHT_BEGIN + i].pos));

                        const double before = q_command_model[i];
                        const double err_to_desired = q[i] - before;
                        const double step = std::max(
                            -rate_step, std::min(rate_step, err_to_desired));
                        double candidate = before + step;

                        if (std::abs(err_to_desired) > rate_step + 1e-12) {
                            any_rate_limited = true;
                            if (first_limited_joint < 0) {
                                first_limited_joint = i;
                                first_desired = q[i];
                                first_before = before;
                                first_after = candidate;
                            }
                        }

                        // Do not let the command "run away" from the actual arm if
                        // low gains / gravity make the hardware lag behind.
                        const double lo_track = measured - MAX_TRACKING_ERROR_RAD;
                        const double hi_track = measured + MAX_TRACKING_ERROR_RAD;
                        candidate = std::max(lo_track, std::min(hi_track, candidate));

                        // Final absolute hardware envelope.
                        candidate = std::max(
                            -ABS_COMMAND_LIMIT_RAD,
                            std::min(ABS_COMMAND_LIMIT_RAD, candidate));

                        q_command_model[i] = candidate;
                    }

                    if (any_rate_limited &&
                        (last_rate_log.time_since_epoch().count() == 0 ||
                         command_now - last_rate_log > std::chrono::milliseconds(250))) {
                        std::cout << "[bridge] rate limiting joint " << first_limited_joint
                                  << ": desired=" << first_desired
                                  << " cmd " << first_before << " -> " << first_after
                                  << " max_step=" << rate_step << "\n";
                        last_rate_log = command_now;
                    }

                    if (!read_ctrl_consistent(shm, ctrl)) {
                        std::cerr << "[bridge] no consistent control block; command not written\n";
                    } else {
                        for (int i = 0; i < RIGHT_DOF; ++i) {
                            ctrl[RIGHT_BEGIN + i].pos =
                                model_to_hardware_q(i, q_command_model[i]);
                            ctrl[RIGHT_BEGIN + i].vel = 0.0;
                        }
                        shm->write_ctrl(ctrl);
                        last_command_update = command_now;
                    }
                }
            }
        } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            perror("recvfrom");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ::close(rx);
    ::close(tx);
    return 0;
}
