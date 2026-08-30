#ifndef SHAREMEM_HPP
#define SHAREMEM_HPP

#include <sys/ipc.h>
#include <sys/shm.h>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <new>

//#include "RobstrideMotor.hpp"

typedef struct
{
    double pos=0;
    double vel=0;
    double Kp=0;
    double Kd=0;
    double ffTorque=0;
}Control_param;

typedef struct{
    float pos=0;
    float vel=0;
    float torque=0;
    float temp=0;
}Feedback_Param;


static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "uint32_t atomic must be lock-free for shm seqlock");

template<int Motor_number>  // 생각해보니까 모터객체하고 연결을 우째하지
class Control_Shm {

public:
    Control_Shm(key_t key) : SHM_KEY(key) { open_shm(); }

    struct Control_Struct {

        alignas(64) std::atomic<int> refcount{0};

        // Written by upper controller, read by motion thread
        struct alignas(64) CtrlBlock {
            std::atomic<uint32_t> seq{0};
            Control_param data[Motor_number];
        } ctrl;

        // Written by motion thread, read by upper controller
        struct alignas(64) FbBlock {
            std::atomic<uint32_t> seq{0};
            Feedback_Param data[Motor_number];
        } fb;

        // Seqlock write: control params (upper controller side)
        void write_ctrl(const Control_param* src) {
            ctrl.seq.fetch_add(1, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            std::memcpy(ctrl.data, src, sizeof(Control_param) * Motor_number);
            std::atomic_thread_fence(std::memory_order_release);
            ctrl.seq.fetch_add(1, std::memory_order_relaxed);
        }

        // Returns false if write in progress or torn read — caller retries
        bool try_read_ctrl(Control_param* dst) const {
            uint32_t s1 = ctrl.seq.load(std::memory_order_acquire);
            if (s1 & 1) return false;
            std::memcpy(dst, ctrl.data, sizeof(Control_param) * Motor_number);
            std::atomic_thread_fence(std::memory_order_acquire);
            return ctrl.seq.load(std::memory_order_acquire) == s1;
        }

        // Seqlock write: feedback (motion thread side)
        void write_fb(const Feedback_Param* src) {
            fb.seq.fetch_add(1, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            std::memcpy(fb.data, src, sizeof(Feedback_Param) * Motor_number);
            std::atomic_thread_fence(std::memory_order_release);
            fb.seq.fetch_add(1, std::memory_order_relaxed);
        }

        bool try_read_fb(Feedback_Param* dst) const {
            uint32_t s1 = fb.seq.load(std::memory_order_acquire);
            if (s1 & 1) return false;
            std::memcpy(dst, fb.data, sizeof(Feedback_Param) * Motor_number);
            std::atomic_thread_fence(std::memory_order_acquire);
            return fb.seq.load(std::memory_order_acquire) == s1;
        }
    };

    Control_Struct* get() { return ptr_; }

    ~Control_Shm() {
        if (ptr_) close_shm();
    }

    private:
    const key_t SHM_KEY;
    int id_ = -1;
    Control_Struct* ptr_ = nullptr;

    void open_shm() {
        int id = shmget(SHM_KEY, sizeof(Control_Struct), IPC_CREAT | IPC_EXCL | 0666);
        bool is_creator = (id >= 0);
        if (!is_creator) {
            id = shmget(SHM_KEY, sizeof(Control_Struct), 0666);
            if (id < 0) { perror("shmget"); exit(EXIT_FAILURE); }
        }

        auto* ptr = static_cast<Control_Struct*>(shmat(id, nullptr, 0));
        if (ptr == reinterpret_cast<Control_Struct*>(-1)) {
            perror("shmat"); exit(EXIT_FAILURE);
        }

        if (is_creator) new (ptr) Control_Struct();

        ptr->refcount.fetch_add(1, std::memory_order_relaxed);
        id_ = id;
        ptr_ = ptr;
    }

    void close_shm() {
        if (ptr_->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            shmdt(ptr_);
            shmctl(id_, IPC_RMID, nullptr);
        } else {
            shmdt(ptr_);
        }
        ptr_ = nullptr;
    }

};

#endif