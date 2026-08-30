#include "Sharemem.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {

constexpr int MOTOR_NUM = 16;
constexpr key_t SHM_KEY = 13563267;

constexpr int LEFT_BEGIN = 0;    // arm SHM 0..6
constexpr int LEFT_GRIPPER = 7;
constexpr int RIGHT_BEGIN = 8;   // arm SHM 8..14
constexpr int RIGHT_GRIPPER = 15;
constexpr int ARM_DOF = 7;

// Measured real-hardware gripper motor endpoints [rad].
// constexpr double LEFT_GRIPPER_OPEN_RAD = -1.24;
// constexpr double LEFT_GRIPPER_CLOSE_RAD = 0.92;
// constexpr double RIGHT_GRIPPER_OPEN_RAD = -1.22;
// constexpr double RIGHT_GRIPPER_CLOSE_RAD = 0.82;

constexpr double LEFT_GRIPPER_OPEN_RAD = -1.1;
constexpr double LEFT_GRIPPER_CLOSE_RAD = 0.7;
constexpr double RIGHT_GRIPPER_OPEN_RAD = -1.1;
constexpr double RIGHT_GRIPPER_CLOSE_RAD = 0.7;

constexpr int CMD_PORT = 5005;
constexpr int FB_PORT = 5006;

// Model-space -> real-hardware sign.
// LEFT has not been physically sign-calibrated yet: keep all +1 for now.
// RIGHT wrist_yaw was already confirmed reversed.
constexpr std::array<double, ARM_DOF> LEFT_SIGN = {
    +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0
};

constexpr std::array<double, ARM_DOF> RIGHT_SIGN = {
    +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0
};

double hw_to_model(
    const std::array<double, ARM_DOF>& sign,
    int joint,
    double q_hw
) {
    return sign[joint] * q_hw;
}

double model_to_hw(
    const std::array<double, ARM_DOF>& sign,
    int joint,
    double q_model
) {
    return sign[joint] * q_model;
}

// Command packet:
//   L,q0,q1,...,q6
//   R,q0,q1,...,q6
bool parse_arm_command(
    const std::string& s,
    char& side,
    std::array<double, ARM_DOF>& q
) {
    std::stringstream ss(s);
    std::string item;

    if (!std::getline(ss, item, ',')) return false;
    if (item != "L" && item != "R") return false;
    side = item[0];

    int i = 0;
    while (std::getline(ss, item, ',')) {
        if (i >= ARM_DOF) return false;
        try {
            q[i] = std::stod(item);
        } catch (...) {
            return false;
        }
        if (!std::isfinite(q[i])) return false;
        ++i;
    }

    return i == ARM_DOF;
}

// Gripper command packet:
//   GL,<motor_angle_rad>
//   GR,<motor_angle_rad>
bool parse_gripper_command(
    const std::string& s,
    bool& left,
    double& q
) {
    std::stringstream ss(s);
    std::string tag;
    std::string value;
    std::string extra;

    if (!std::getline(ss, tag, ',')) return false;
    if (tag != "GL" && tag != "GR") return false;

    if (!std::getline(ss, value, ',')) return false;
    if (std::getline(ss, extra, ',')) return false;

    try {
        q = std::stod(value);
    } catch (...) {
        return false;
    }

    if (!std::isfinite(q)) return false;

    left = (tag == "GL");
    return true;
}

// Feedback packet (32 CSV values):
//   measured: LEFT7,RIGHT7
//   control : LEFT7,RIGHT7
//   gripper : measured LEFT,RIGHT, control LEFT,RIGHT
//
// This lets Python use:
//   measured arm q -> FK / IK origin
//   control arm q  -> preserve the current CAN holding target at clutch engage
//   measured/control gripper q -> safe rate-limited trigger control.
std::string csv32_from_shm(
    const Feedback_Param* fb,
    const Control_param* ctrl
) {
    std::ostringstream os;
    os.precision(12);

    bool first = true;
    auto append = [&](double v) {
        if (!first) os << ',';
        os << v;
        first = false;
    };

    for (int i = 0; i < ARM_DOF; ++i) {
        append(hw_to_model(
            LEFT_SIGN, i,
            static_cast<double>(fb[LEFT_BEGIN + i].pos)
        ));
    }
    for (int i = 0; i < ARM_DOF; ++i) {
        append(hw_to_model(
            RIGHT_SIGN, i,
            static_cast<double>(fb[RIGHT_BEGIN + i].pos)
        ));
    }
    for (int i = 0; i < ARM_DOF; ++i) {
        append(hw_to_model(
            LEFT_SIGN, i,
            static_cast<double>(ctrl[LEFT_BEGIN + i].pos)
        ));
    }
    for (int i = 0; i < ARM_DOF; ++i) {
        append(hw_to_model(
            RIGHT_SIGN, i,
            static_cast<double>(ctrl[RIGHT_BEGIN + i].pos)
        ));
    }

    // Gripper values are already real motor angles [rad], so no arm sign mapping.
    // Order must match the Python teleop:
    //   [28] measured left gripper
    //   [29] measured right gripper
    //   [30] control left gripper
    //   [31] control right gripper
    append(static_cast<double>(fb[LEFT_GRIPPER].pos));
    append(static_cast<double>(fb[RIGHT_GRIPPER].pos));
    append(static_cast<double>(ctrl[LEFT_GRIPPER].pos));
    append(static_cast<double>(ctrl[RIGHT_GRIPPER].pos));

    return os.str();
}

bool read_ctrl_consistent(
    const Control_Shm<MOTOR_NUM>::Control_Struct* shm,
    Control_param* ctrl
) {
    for (int tries = 0; tries < 20; ++tries) {
        if (shm->try_read_ctrl(ctrl)) return true;
    }
    return false;
}

bool read_fb_consistent(
    const Control_Shm<MOTOR_NUM>::Control_Struct* shm,
    Feedback_Param* fb
) {
    for (int tries = 0; tries < 20; ++tries) {
        if (shm->try_read_fb(fb)) return true;
    }
    return false;
}


// ============================================================================
// BRIDGE SAFETY / SMOOTHING — TEMPORARILY DISABLED
//
// User requested that CAN-side safety be used for now.
// Keep this code here instead of deleting it, so it can be restored later.
// ============================================================================
#if 0

constexpr double MAX_COMMAND_RATE_RAD_S = 0.60;
constexpr double MAX_COMMAND_STEP_RAD = 0.10;
constexpr double MAX_TRACKING_ERROR_RAD = 0.80;
constexpr double ABS_COMMAND_LIMIT_RAD = 2.0;

struct ArmSafetyState {
    bool initialized = false;
    std::array<double, ARM_DOF> q_command{};
    std::chrono::steady_clock::time_point last_update =
        std::chrono::steady_clock::now();
};

bool apply_bridge_safety(
    const std::array<double, ARM_DOF>& desired,
    const Feedback_Param* fb,
    int begin,
    const std::array<double, ARM_DOF>& sign,
    ArmSafetyState& state,
    std::array<double, ARM_DOF>& safe_command
) {
    const auto now = std::chrono::steady_clock::now();

    for (int i = 0; i < ARM_DOF; ++i) {
        if (!std::isfinite(desired[i]) ||
            std::abs(desired[i]) > ABS_COMMAND_LIMIT_RAD) {
            return false;
        }
    }

    if (!state.initialized) {
        for (int i = 0; i < ARM_DOF; ++i) {
            state.q_command[i] = hw_to_model(
                sign,
                i,
                static_cast<double>(fb[begin + i].pos)
            );
        }
        state.initialized = true;
        state.last_update = now;
    }

    double dt =
        std::chrono::duration<double>(now - state.last_update).count();
    if (!std::isfinite(dt) || dt <= 0.0) dt = 0.001;
    if (dt > 0.05) dt = 0.05;

    const double rate_step = std::min(
        MAX_COMMAND_STEP_RAD,
        MAX_COMMAND_RATE_RAD_S * dt
    );

    for (int i = 0; i < ARM_DOF; ++i) {
        const double measured = hw_to_model(
            sign,
            i,
            static_cast<double>(fb[begin + i].pos)
        );

        const double error = desired[i] - state.q_command[i];
        const double step = std::clamp(error, -rate_step, +rate_step);

        double candidate = state.q_command[i] + step;

        candidate = std::clamp(
            candidate,
            measured - MAX_TRACKING_ERROR_RAD,
            measured + MAX_TRACKING_ERROR_RAD
        );

        candidate = std::clamp(
            candidate,
            -ABS_COMMAND_LIMIT_RAD,
            +ABS_COMMAND_LIMIT_RAD
        );

        state.q_command[i] = candidate;
        safe_command[i] = candidate;
    }

    state.last_update = now;
    return true;
}

#endif
// ============================================================================

}  // namespace


int main() {
    Control_Shm<MOTOR_NUM> shared(SHM_KEY);
    auto* shm = shared.get();

    Control_param ctrl[MOTOR_NUM]{};
    Feedback_Param fb[MOTOR_NUM]{};

    int rx = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (rx < 0) {
        perror("socket(rx)");
        return 1;
    }

    sockaddr_in rx_addr{};
    rx_addr.sin_family = AF_INET;
    rx_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    rx_addr.sin_port = htons(CMD_PORT);

    if (::bind(
            rx,
            reinterpret_cast<sockaddr*>(&rx_addr),
            sizeof(rx_addr)
        ) < 0) {
        perror("bind(command)");
        return 1;
    }

    int flags = fcntl(rx, F_GETFL, 0);
    fcntl(rx, F_SETFL, flags | O_NONBLOCK);

    int tx = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (tx < 0) {
        perror("socket(tx)");
        return 1;
    }

    sockaddr_in fb_addr{};
    fb_addr.sin_family = AF_INET;
    fb_addr.sin_port = htons(FB_PORT);
    inet_pton(AF_INET, "127.0.0.1", &fb_addr.sin_addr);

    auto next_feedback = std::chrono::steady_clock::now();

    std::cout
        << "[bridge] BIMANUAL + GRIPPERS"
        << "\n[bridge] left arm SHM  = 0..6, left gripper = 7"
        << "\n[bridge] right arm SHM = 8..14, right gripper = 15"
        << "\n[bridge] command 127.0.0.1:" << CMD_PORT
        << "  feedback 127.0.0.1:" << FB_PORT
        << "\n[bridge] arm command: L,q0..q6 or R,q0..q6"
        << "\n[bridge] gripper command: GL,<rad> or GR,<rad>"
        << "\n[bridge] gripper range L=["
        << LEFT_GRIPPER_OPEN_RAD << ", " << LEFT_GRIPPER_CLOSE_RAD << "] rad"
        << " R=[" << RIGHT_GRIPPER_OPEN_RAD << ", "
        << RIGHT_GRIPPER_CLOSE_RAD << "] rad"
        << "\n[bridge] feedback: measured L7,R7, control L7,R7, "
           "measured grip L,R, control grip L,R (32 values)"
        << "\n[bridge] arm bridge safety/smoothing DISABLED (#if 0)"
        << "\n";

    while (true) {
        const auto now = std::chrono::steady_clock::now();

        // Actual encoder feedback -> Python, ~100 Hz.
        if (now >= next_feedback) {
            if (read_fb_consistent(shm, fb) &&
                read_ctrl_consistent(shm, ctrl)) {
                const std::string msg = csv32_from_shm(fb, ctrl);
                ::sendto(
                    tx,
                    msg.data(),
                    msg.size(),
                    0,
                    reinterpret_cast<const sockaddr*>(&fb_addr),
                    sizeof(fb_addr)
                );
            }
            next_feedback = now + std::chrono::milliseconds(10);
        }

        char buf[512];
        sockaddr_in src{};
        socklen_t src_len = sizeof(src);

        const ssize_t n = ::recvfrom(
            rx,
            buf,
            sizeof(buf) - 1,
            0,
            reinterpret_cast<sockaddr*>(&src),
            &src_len
        );

        if (n > 0) {
            buf[n] = '\0';
            const std::string packet(buf);

            char side = '\0';
            std::array<double, ARM_DOF> q{};
            bool gripper_left = false;
            double gripper_q = 0.0;

            if (parse_arm_command(packet, side, q)) {
                if (!read_ctrl_consistent(shm, ctrl)) {
                    std::cerr
                        << "[bridge] no consistent control block; "
                        << "arm command not written\n";
                } else {
                    const bool left = side == 'L';
                    const int begin = left ? LEFT_BEGIN : RIGHT_BEGIN;
                    const auto& sign = left ? LEFT_SIGN : RIGHT_SIGN;

                    // ACTIVE ARM PATH: direct q forwarding.
                    // Only the requested arm is modified.
                    for (int i = 0; i < ARM_DOF; ++i) {
                        ctrl[begin + i].pos =
                            model_to_hw(sign, i, q[i]);
                        ctrl[begin + i].vel = 0.0;
                    }

                    shm->write_ctrl(ctrl);

#if 0
                    // To restore bridge safety later, replace the direct-forward
                    // loop above with this block.
                    static ArmSafetyState left_safety;
                    static ArmSafetyState right_safety;

                    if (!read_fb_consistent(shm, fb)) {
                        std::cerr << "[bridge] no feedback; command dropped\n";
                    } else {
                        auto& state = left ? left_safety : right_safety;
                        std::array<double, ARM_DOF> safe_q{};

                        if (apply_bridge_safety(
                                q,
                                fb,
                                begin,
                                sign,
                                state,
                                safe_q
                            )) {
                            for (int i = 0; i < ARM_DOF; ++i) {
                                ctrl[begin + i].pos =
                                    model_to_hw(sign, i, safe_q[i]);
                                ctrl[begin + i].vel = 0.0;
                            }
                            shm->write_ctrl(ctrl);
                        }
                    }
#endif
                }

            } else if (parse_gripper_command(
                           packet,
                           gripper_left,
                           gripper_q
                       )) {
                if (!read_ctrl_consistent(shm, ctrl)) {
                    std::cerr
                        << "[bridge] no consistent control block; "
                        << "gripper command not written\n";
                } else {
                    const int index =
                        gripper_left ? LEFT_GRIPPER : RIGHT_GRIPPER;

                    const double q_min = gripper_left
                        ? LEFT_GRIPPER_OPEN_RAD
                        : RIGHT_GRIPPER_OPEN_RAD;
                    const double q_max = gripper_left
                        ? LEFT_GRIPPER_CLOSE_RAD
                        : RIGHT_GRIPPER_CLOSE_RAD;

                    // The Python side already rate-limits the trigger command.
                    // This bridge adds a hard endpoint guard from the measured
                    // real gripper motor range.
                    if (gripper_q < q_min || gripper_q > q_max) {
                        std::cerr
                            << "[bridge] "
                            << (gripper_left ? "LEFT" : "RIGHT")
                            << " gripper command outside measured range: "
                            << gripper_q
                            << " not in [" << q_min << ", " << q_max
                            << "] rad; dropped\n";
                    } else {
                        ctrl[index].pos = gripper_q;
                        ctrl[index].vel = 0.0;
                        shm->write_ctrl(ctrl);
                    }
                }

            } else {
                std::cerr
                    << "[bridge] malformed command dropped: "
                    << packet << "\n";
            }

        } else if (
            n < 0 &&
            errno != EAGAIN &&
            errno != EWOULDBLOCK
        ) {
            perror("recvfrom");
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ::close(rx);
    ::close(tx);
    return 0;
}
