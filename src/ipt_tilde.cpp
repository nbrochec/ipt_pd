// ipt~ for Pure Data: real-time Instrumental Playing Technique (IPT) recognition
// with a TorchScript classifier.
//
// Port of the Max ipt~ (ipt_tilde/ipt_tilde.cpp in the ipt_tilde repository),
// with the same arguments, settings and outlets. Everything model-related goes
// through libipt's C ABI (libipt/include/ipt.h); no torch call is made here.
//
// Threads
// -------
// In Pd, messages, clocks and DSP all run on one thread, the scheduler. So the
// "main thread" of Max is the audio thread here, and nothing that can block may
// run on it. libipt's classifier holds an internal mutex for the whole forward
// pass (ipt_process) and the whole model load (ipt_initialize_model), and
// ipt_set_energy_threshold / ipt_set_threshold_window / ipt_init_buffers take
// that same mutex. Those three are therefore called only by the worker thread:
// the Pd side stores the values in atomics and the worker applies them between
// two ipt_process calls.
//
//   perform (DSP)  pushes samples into a lock-free FIFO and arms a clock when
//                  results are waiting. No allocation, no torch, no lock.
//   worker thread  loads the model, applies pending settings, drains the FIFO,
//                  calls ipt_process, pushes raw distributions into a second
//                  lock-free FIFO. It never calls into Pd (post() is not
//                  thread-safe); errors go through a one-slot mailbox.
//   clocks (Pd)    the deliver clock smooths with ipt_smooth (no mutex in
//                  libipt) and sends to the outlets; the poll clock reports the
//                  end of the model load, even with DSP off.

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <condition_variable>
#include <mutex>
#include <system_error>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "m_pd.h"
#include "ipt.h"

#ifndef IPT_PACKAGE_VERSION
#define IPT_PACKAGE_VERSION "dev"
#endif

namespace {

constexpr double      kDefaultThresholdDb   = -80.0;  // <= -80 disables the gate
constexpr int         kDefaultWindowMs      = 20;
constexpr double      kDefaultSensitivity   = 1.0;
constexpr int         kDefaultRangeMs       = 2000;
constexpr double      kDefaultConfidence    = 0.0;
constexpr int         kDefaultPeriodMs      = 0;
constexpr int         kMaxClasses           = 256;     // same cap as the Max ipt~
constexpr std::size_t kAudioFifoSize        = 16384;   // samples, as in the Max ipt~
constexpr std::size_t kResultFifoSize       = 128;     // Max: 100
constexpr double      kLoadPollMs           = 20.0;
constexpr double      kErrorIntervalMs      = 1000.0;  // worker error rate limit

// ---------------------------------------------------------------------------
// Lock-free single-producer / single-consumer FIFOs, preallocated.
// ---------------------------------------------------------------------------

// Audio samples: producer is the DSP perform routine, consumer the worker.
template <std::size_t N>
class AudioFifo {
    static_assert((N & (N - 1)) == 0, "AudioFifo size must be a power of two");
public:
    // Producer. Copies as many samples as fit; the rest are counted as dropped.
    void push(const t_sample* in, std::size_t n) {
        const std::size_t w = m_write.load(std::memory_order_relaxed);
        const std::size_t r = m_read.load(std::memory_order_acquire);
        const std::size_t k = std::min(n, N - (w - r));
        for (std::size_t i = 0; i < k; ++i) {
            m_buf[(w + i) & (N - 1)] = static_cast<float>(in[i]);
        }
        m_write.store(w + k, std::memory_order_release);
        if (k < n) {
            m_dropped.fetch_add(n - k, std::memory_order_relaxed);
        }
    }

    // Consumer. Returns the number of samples written to out.
    std::size_t pop(double* out, std::size_t max) {
        const std::size_t r = m_read.load(std::memory_order_relaxed);
        const std::size_t w = m_write.load(std::memory_order_acquire);
        const std::size_t k = std::min(max, w - r);
        for (std::size_t i = 0; i < k; ++i) {
            out[i] = static_cast<double>(m_buf[(r + i) & (N - 1)]);
        }
        m_read.store(r + k, std::memory_order_release);
        return k;
    }

    // Consumer. Samples dropped since the last call (FIFO was full).
    std::size_t take_dropped() { return m_dropped.exchange(0, std::memory_order_relaxed); }

private:
    std::array<float, N> m_buf{};
    alignas(64) std::atomic<std::size_t> m_write{0};
    alignas(64) std::atomic<std::size_t> m_read{0};
    std::atomic<std::size_t> m_dropped{0};
};

// Raw classification result, stamped with its audio time so that results
// drained together are smoothed with their real spacing.
struct Result {
    int    n          = 0;
    double latency_ms = 0.0;
    double time_ms    = 0.0;
    std::array<float, kMaxClasses> dist{};
};

// Results: producer is the worker, consumer the deliver clock.
template <typename T, std::size_t N>
class SpscQueue {
    static_assert((N & (N - 1)) == 0, "SpscQueue size must be a power of two");
public:
    bool push(const T& v) {
        const std::size_t w = m_write.load(std::memory_order_relaxed);
        const std::size_t r = m_read.load(std::memory_order_acquire);
        if (w - r >= N) return false;
        m_buf[w & (N - 1)] = v;
        m_write.store(w + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& v) {
        const std::size_t r = m_read.load(std::memory_order_relaxed);
        const std::size_t w = m_write.load(std::memory_order_acquire);
        if (r == w) return false;
        v = m_buf[r & (N - 1)];
        m_read.store(r + 1, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return m_read.load(std::memory_order_acquire) == m_write.load(std::memory_order_acquire);
    }

private:
    std::array<T, N> m_buf{};
    alignas(64) std::atomic<std::size_t> m_write{0};
    alignas(64) std::atomic<std::size_t> m_read{0};
};

// One-slot mailbox for worker error messages, read on the Pd thread.
class Mailbox {
public:
    // Worker. Returns false (message dropped) if the previous one is unread.
    bool post(const char* text) {
        if (m_full.load(std::memory_order_acquire)) return false;
        std::snprintf(m_text, sizeof(m_text), "%s", text);
        m_full.store(true, std::memory_order_release);
        return true;
    }

    bool pending() const { return m_full.load(std::memory_order_acquire); }

    // Pd thread. Copies the message into out and frees the slot.
    bool take(std::string& out) {
        if (!m_full.load(std::memory_order_acquire)) return false;
        out = m_text;
        m_full.store(false, std::memory_order_release);
        return true;
    }

private:
    std::atomic<bool> m_full{false};
    char m_text[2048] = {};
};

enum LoadState : int { kLoading = 0, kReady = 1, kFailed = 2 };

// ---------------------------------------------------------------------------
// Object state. Allocated with new in ipt_new. ipt_free hands it to a
// detached cleanup thread, which joins the worker, destroys the handle and
// deletes it, so that freeing never blocks the Pd thread.
// ---------------------------------------------------------------------------
struct IptState {
    // --- set once in ipt_new, then read-only ---
    ipt_classifier* clf    = nullptr;
    ipt_device      device = IPT_DEVICE_CPU;
    std::string     model_path;

    // --- shared between threads ---
    AudioFifo<kAudioFifoSize>             audio;
    SpscQueue<Result, kResultFifoSize>    results;
    Mailbox                               errors;
    std::atomic<int>                      load_state{kLoading};
    std::atomic<bool>                     stop{false};
    std::atomic<bool>                     enabled{true};
    std::atomic<bool>                     verbose{false};
    // settings that take libipt's internal mutex: written by Pd, applied by the worker
    std::atomic<double>                   threshold_db{kDefaultThresholdDb};
    std::atomic<int>                      window_ms{kDefaultWindowMs};
    std::atomic<int>                      sample_rate{0};
    std::atomic<int>                      block_size{0};

    // --- written by the worker before load_state leaves kLoading, then read-only ---
    std::string              load_error;
    std::vector<std::string> class_names;

    // --- worker thread ---
    std::thread worker;
    // Protected by live_mutex(): set by whoever takes charge of joining the
    // worker (the cleanup thread started by ipt_free, or stop_all_workers when
    // Pd exits), so that it is joined exactly once.
    bool join_claimed = false;

    // --- Pd thread only ---
    double                  sensitivity = kDefaultSensitivity;
    int                     range_ms    = kDefaultRangeMs;
    double                  confidence  = kDefaultConfidence;
    int                     period_ms   = kDefaultPeriodMs;
    bool                    armed        = false;  // deliver clock scheduled
    bool                    period_armed = false;  // period clock scheduled
    bool                    pending      = false;  // smoothed result not sent yet
    double                  latency_ms   = 0.0;    // latency of the last result
    int                     n_dist       = 0;      // size of the smoothed distribution
    double                  last_output  = 0.0;    // logical time of the last output
    std::vector<t_symbol*>  class_syms;
    Result                  scratch;
    std::array<float, kMaxClasses>  smoothed{};
    std::array<t_atom, kMaxClasses> atoms{};
};

static_assert(std::atomic<double>::is_always_lock_free, "atomic<double> must be lock-free");

// ---------------------------------------------------------------------------
// Quitting Pd. "pd quit" calls exit() without freeing the patches, so ipt_free
// never runs and the workers are still inside ipt_process while exit()
// destroys torch's static objects: a crash (seen with MPS) or undefined
// behaviour. An atexit handler stops and joins every live worker first.
//
// atexit handlers and static destructors run in reverse order of
// registration, and torch creates some statics lazily on the worker, while
// loading a model (the TorchScript builtin registry) or running it (the MPS
// allocator, the CPU thread pool). So the handler is registered at setup and
// again by every worker after its model load and after its first inference,
// which puts it after those statics and makes it run before they are
// destroyed. The registry is allocated once and never freed, so it outlives
// every destructor.
//
// Known limit: statics created by a load that is still running when exit()
// starts are registered after the handler and destroyed before it runs. So
// quitting during the first load of a model kind in the process (a patch just
// opened, or an object just created or deleted) can still crash on exit. Once
// a model of that kind has loaded, later loads create nothing new and are
// waited for.
//
// Freeing an object. ipt_free must not block the Pd thread: in plugdata the
// scheduler runs in the audio callback under sys_lock, and joining a worker
// that is loading a model stalls the audio for the whole load (measured: about
// 150 ms on CPU, up to 750 ms on MPS with a real model). So ipt_free only
// raises the stop flag and starts a detached cleanup thread that joins the
// worker, destroys the handle and deletes the state. The live registry keeps
// the state until then, and stop_all_workers waits for every cleanup thread
// before torch's statics are destroyed.
// ---------------------------------------------------------------------------

std::mutex& live_mutex() {
    static auto* m = new std::mutex;
    return *m;
}

std::vector<IptState*>& live_states() {
    static auto* v = new std::vector<IptState*>;
    return *v;
}

// Detached cleanup threads still running, protected by live_mutex().
int& cleanups_running() {
    static auto* n = new int(0);
    return *n;
}

std::condition_variable& cleanups_done() {
    static auto* cv = new std::condition_variable;
    return *cv;
}

void stop_all_workers() {
    std::unique_lock<std::mutex> lock(live_mutex());
    // Stop every worker; join those no cleanup thread is in charge of.
    std::vector<IptState*> to_join;
    for (IptState* st : live_states()) {
        st->stop.store(true, std::memory_order_release);
        if (!st->join_claimed) {
            st->join_claimed = true;
            to_join.push_back(st);
        }
    }
    // Joined without the lock, so that cleanup threads can finish meanwhile.
    // These states cannot be deleted: only a cleanup thread deletes a state,
    // and only one whose join it claimed.
    lock.unlock();
    for (IptState* st : to_join) {
        if (st->worker.joinable() && st->worker.get_id() != std::this_thread::get_id()) {
            st->worker.join();
        }
    }
    // Wait for the objects freed just before exit to finish their cleanup.
    lock.lock();
    cleanups_done().wait(lock, [] { return cleanups_running() == 0; });
    // The handles of the live objects are not destroyed: the process is
    // exiting and torch is about to be torn down.
}

// Cleanup thread started by ipt_free (or called directly if no thread could
// be started). Never touches the t_ipt, which Pd has already freed.
void cleanup_state(IptState* st) {
    if (st->worker.joinable()) {
        st->worker.join();
    }
    ipt_destroy(st->clf);

    std::lock_guard<std::mutex> lock(live_mutex());
    auto& live = live_states();
    live.erase(std::remove(live.begin(), live.end(), st), live.end());
    delete st;
    if (--cleanups_running() == 0) {
        cleanups_done().notify_all();
    }
}

// Called by every worker after its model load and after its first inference,
// so that the handler is registered after the statics torch created on the
// way (stop_all_workers is idempotent; two registrations per object).
void register_exit_hook() {
    std::atexit(stop_all_workers);
}

} // namespace

// ---------------------------------------------------------------------------
// Pd object
// ---------------------------------------------------------------------------

static t_class* ipt_class = nullptr;

struct t_ipt {
    t_object   x_obj;
    t_float    x_f;               // CLASS_MAINSIGNALIN scalar
    IptState*  x_st;
    t_clock*   x_deliver_clock;   // drains and smooths results
    t_clock*   x_period_clock;    // sends the pending result at the end of a period
    t_clock*   x_poll_clock;      // reports the end of the model load
    t_outlet*  x_out_index;
    t_outlet*  x_out_name;
    t_outlet*  x_out_dist;
    t_outlet*  x_out_info;
};

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

static void worker_report(IptState* st, std::chrono::steady_clock::time_point& last,
                          bool& reported, const char* text) {
    const auto now = std::chrono::steady_clock::now();
    if (reported && std::chrono::duration<double, std::milli>(now - last).count() < kErrorIntervalMs) {
        return;
    }
    if (st->errors.post(text)) {
        last = now;
        reported = true;
    }
}

static void worker_main(IptState* st) {
    ipt_classifier* clf = st->clf;

    // Model load. ipt_last_error() is thread-local, so it is read here.
    if (ipt_initialize_model(clf) != IPT_OK) {
        st->load_error = ipt_last_error();
        st->load_state.store(kFailed, std::memory_order_release);
        return;
    }
    const int n_classes = ipt_num_classes(clf);
    for (int i = 0; i < n_classes; ++i) {
        char buf[256];
        ipt_get_class_name(clf, i, buf, sizeof(buf));
        st->class_names.emplace_back(buf);
    }
    register_exit_hook();  // after the statics this load created
    st->load_state.store(kReady, std::memory_order_release);

    std::vector<double> buffer(kAudioFifoSize);
    Result result;
    double applied_threshold = std::numeric_limits<double>::quiet_NaN();
    int    applied_window    = std::numeric_limits<int>::min();
    int    applied_sr        = 0;
    int    applied_block     = 0;
    bool   buffers_ok        = false;
    double audio_ms          = 0.0;   // audio time of the samples consumed so far
    auto   last_error        = std::chrono::steady_clock::time_point{};
    bool   error_reported    = false;
    bool   first_inference   = true;

    while (!st->stop.load(std::memory_order_acquire)) {
        // Apply pending settings, between two ipt_process calls.
        const double threshold = st->threshold_db.load(std::memory_order_relaxed);
        if (!(threshold == applied_threshold)) {
            ipt_set_energy_threshold(clf, threshold);
            applied_threshold = threshold;
        }
        const int window = st->window_ms.load(std::memory_order_relaxed);
        if (window != applied_window) {
            ipt_set_threshold_window(clf, window);
            applied_window = window;
        }
        // Pd calls the dsp method on every DSP graph rebuild (any patch edit
        // while DSP is on), so buffers are reinitialised only on a real change.
        const int sr    = st->sample_rate.load(std::memory_order_relaxed);
        const int block = st->block_size.load(std::memory_order_relaxed);
        if (sr > 0 && block > 0 && (sr != applied_sr || block != applied_block)) {
            applied_sr    = sr;
            applied_block = block;
            buffers_ok    = ipt_init_buffers(clf, sr, block) == IPT_OK;
            if (!buffers_ok) {
                std::string msg = std::string("cannot initialise buffers: ") + ipt_last_error();
                worker_report(st, last_error, error_reported, msg.c_str());
            }
        }

        if (buffers_ok && st->enabled.load(std::memory_order_relaxed)) {
            // Samples lost to a full FIFO still count as elapsed audio time.
            audio_ms += 1000.0 * static_cast<double>(st->audio.take_dropped()) / applied_sr;

            const std::size_t k = st->audio.pop(buffer.data(), buffer.size());
            if (k > 0) {
                audio_ms += 1000.0 * static_cast<double>(k) / applied_sr;
                double latency_ms = 0.0;
                const int n = ipt_process(clf, buffer.data(), static_cast<int>(k),
                                          result.dist.data(), kMaxClasses, &latency_ms);
                if (n < 0) {
                    worker_report(st, last_error, error_reported,
                                  st->verbose.load(std::memory_order_relaxed)
                                      ? ipt_last_error()
                                      : "model architecture is not compatible");
                } else if (n > 0) {
                    // Raw distribution; smoothing happens on the Pd thread.
                    result.n          = std::min(n, kMaxClasses);
                    result.latency_ms = latency_ms;
                    result.time_ms    = audio_ms;
                    st->results.push(result);  // dropped if the Pd side is 128 results behind
                    if (first_inference) {
                        first_inference = false;
                        register_exit_hook();
                    }
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// ---------------------------------------------------------------------------
// Pd thread helpers
// ---------------------------------------------------------------------------

static void ensure_class_symbols(IptState* st) {
    if (!st->class_syms.empty()) return;
    if (st->load_state.load(std::memory_order_acquire) != kReady) return;
    for (const auto& name : st->class_names) {
        st->class_syms.push_back(gensym(name.c_str()));
    }
}

static void report_worker_errors(t_ipt* x) {
    std::string msg;
    if (x->x_st->errors.take(msg)) {
        pd_error(x, "ipt~: %s", msg.c_str());
    }
}

static void update_smoothing(IptState* st, double tau_ms) {
    ipt_set_smoothing_tau(st->clf, tau_ms);
}

// ---------------------------------------------------------------------------
// Clocks
// ---------------------------------------------------------------------------

static void ipt_output(t_ipt* x) {
    IptState* st = x->x_st;
    const int n = st->n_dist;
    st->pending     = false;
    st->last_output = clock_getlogicaltime();
    if (n <= 0) return;

    ensure_class_symbols(st);

    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (st->smoothed[i] > st->smoothed[best]) best = i;
    }
    for (int i = 0; i < n; ++i) {
        SETFLOAT(&st->atoms[i], st->smoothed[i]);
    }

    // Right to left, the Pd convention.
    t_atom latency;
    SETFLOAT(&latency, static_cast<t_float>(st->latency_ms));
    outlet_anything(x->x_out_info, gensym("latency"), 1, &latency);
    outlet_list(x->x_out_dist, &s_list, n, st->atoms.data());

    if (st->smoothed[best] >= st->confidence) {
        t_symbol* name = best < static_cast<int>(st->class_syms.size())
                             ? st->class_syms[best] : gensym("unknown");
        outlet_symbol(x->x_out_name, name);
        outlet_float(x->x_out_index, static_cast<t_float>(best));
    } else {
        outlet_symbol(x->x_out_name, gensym("no_confidence"));
        outlet_float(x->x_out_index, -1);
    }
}

// Drain all pending results, smoothing each at its own audio time, then send
// the last smoothed distribution, at most once per period when period > 0.
// Draining never waits for the period, so the result FIFO cannot overflow and
// every result of a period goes through the leaky integrator.
static void ipt_drain(t_ipt* x) {
    IptState* st = x->x_st;

    report_worker_errors(x);

    while (st->results.pop(st->scratch)) {
        st->n_dist = st->scratch.n;
        ipt_smooth(st->clf, st->scratch.dist.data(), st->n_dist, st->scratch.time_ms,
                   st->smoothed.data(), kMaxClasses);
        st->latency_ms = st->scratch.latency_ms;
        st->pending    = true;
    }
    if (!st->pending) return;

    if (st->period_ms > 0) {
        const double remaining = st->period_ms - clock_gettimesince(st->last_output);
        if (remaining > 1e-6) {
            if (!st->period_armed) {
                st->period_armed = true;
                clock_delay(x->x_period_clock, remaining);
            }
            return;
        }
    }
    ipt_output(x);
}

static void ipt_deliver(t_ipt* x) {
    x->x_st->armed = false;
    ipt_drain(x);
}

static void ipt_period_tick(t_ipt* x) {
    x->x_st->period_armed = false;
    ipt_drain(x);
}

static void ipt_poll(t_ipt* x) {
    IptState* st = x->x_st;
    const int state = st->load_state.load(std::memory_order_acquire);
    if (state == kLoading) {
        clock_delay(x->x_poll_clock, kLoadPollMs);
        return;
    }
    const bool verbose = st->verbose.load(std::memory_order_relaxed);
    if (state == kReady) {
        ensure_class_symbols(st);
        logpost(x, verbose ? PD_NORMAL : PD_DEBUG, "ipt~: model loaded, %d classes (%s)",
                static_cast<int>(st->class_names.size()), st->model_path.c_str());
    } else if (verbose) {
        pd_error(x, "ipt~: error during loading: %s", st->load_error.c_str());
    } else {
        pd_error(x, "ipt~: error during loading (create the object with -verbose 1 for details)");
    }
}

// ---------------------------------------------------------------------------
// DSP
// ---------------------------------------------------------------------------

static t_int* ipt_perform(t_int* w) {
    auto*          x  = reinterpret_cast<t_ipt*>(w[1]);
    const t_sample* in = reinterpret_cast<t_sample*>(w[2]);
    const auto     n  = static_cast<std::size_t>(w[3]);
    IptState*      st = x->x_st;

    if (st->load_state.load(std::memory_order_acquire) == kReady
        && st->enabled.load(std::memory_order_relaxed)) {
        st->audio.push(in, n);
    }

    // clock_delay is allowed here (Pd's own env~ does it): same thread as the
    // clocks, no allocation.
    if (!st->armed && (!st->results.empty() || st->errors.pending())) {
        st->armed = true;
        clock_delay(x->x_deliver_clock, 0.0);
    }

    return w + 4;
}

static void ipt_dsp(t_ipt* x, t_signal** sp) {
    x->x_st->sample_rate.store(static_cast<int>(sp[0]->s_sr), std::memory_order_relaxed);
    x->x_st->block_size.store(sp[0]->s_n, std::memory_order_relaxed);
    dsp_add(ipt_perform, 3, x, sp[0]->s_vec, static_cast<t_int>(sp[0]->s_n));
}

// ---------------------------------------------------------------------------
// Settings (same names, defaults and meaning as the Max attributes)
// ---------------------------------------------------------------------------

static bool single_number(t_ipt* x, const char* name, int argc, const t_atom* argv, double& out) {
    if (argc == 1 && argv[0].a_type == A_FLOAT) {
        out = argv[0].a_w.w_float;
        return true;
    }
    pd_error(x, "ipt~: bad argument for message \"%s\" (expects one number)", name);
    return false;
}

static void set_enabled(t_ipt* x, double v) {
    x->x_st->enabled.store(v != 0.0, std::memory_order_relaxed);
}

static void set_verbose(t_ipt* x, double v) {
    x->x_st->verbose.store(v != 0.0, std::memory_order_relaxed);
}

// smoothing time constant = (1 - sensitivity) * sensitivityrange, as in Max
static void set_sensitivity(t_ipt* x, double v) {
    IptState* st = x->x_st;
    st->sensitivity = std::clamp(v, 0.0, 1.0);
    update_smoothing(st, (1.0 - st->sensitivity) * st->range_ms);
}

static void set_sensitivityrange(t_ipt* x, double v) {
    IptState* st = x->x_st;
    const int range = static_cast<int>(v);
    if (range <= 0) {
        pd_error(x, "ipt~: bad argument for message \"sensitivityrange\" (expects ms > 0)");
        return;
    }
    st->range_ms = range;
    update_smoothing(st, std::max((1.0 - st->sensitivity) * range, 1e-6));
}

static void set_threshold(t_ipt* x, double v) {
    x->x_st->threshold_db.store(v, std::memory_order_relaxed);
}

static void set_window(t_ipt* x, double v) {
    x->x_st->window_ms.store(static_cast<int>(v), std::memory_order_relaxed);
}

static void set_confidence(t_ipt* x, double v) {
    x->x_st->confidence = std::clamp(v, 0.0, 1.0);
}

static void set_period(t_ipt* x, double v) {
    x->x_st->period_ms = std::max(0, static_cast<int>(v));
}

struct Setting {
    const char* name;
    void (*set)(t_ipt*, double);
};

static const Setting kSettings[] = {
    {"enabled",          set_enabled},
    {"sensitivity",      set_sensitivity},
    {"sensitivityrange", set_sensitivityrange},
    {"threshold",        set_threshold},
    {"window",           set_window},
    {"confidence",       set_confidence},
    {"period",           set_period},
    {"verbose",          set_verbose},
};

static const Setting* find_setting(const char* name) {
    for (const auto& s : kSettings) {
        if (std::strcmp(s.name, name) == 0) return &s;
    }
    return nullptr;
}

// One message handler for all settings: Pd passes the selector as s.
static void ipt_setting(t_ipt* x, t_symbol* s, int argc, t_atom* argv) {
    const Setting* setting = find_setting(s->s_name);
    double v = 0.0;
    if (setting && single_number(x, s->s_name, argc, argv, v)) {
        setting->set(x, v);
    }
}

static void ipt_classnames(t_ipt* x) {
    IptState* st = x->x_st;
    const int state = st->load_state.load(std::memory_order_acquire);
    if (state == kLoading) {
        pd_error(x, "ipt~: cannot get classnames: the model is still loading");
        return;
    }
    if (state != kReady) {
        pd_error(x, "ipt~: cannot get classnames: no model has been loaded");
        return;
    }
    ensure_class_symbols(st);
    const int n = std::min(static_cast<int>(st->class_syms.size()), kMaxClasses);
    for (int i = 0; i < n; ++i) {
        SETSYMBOL(&st->atoms[i], st->class_syms[i]);
    }
    outlet_anything(x->x_out_info, gensym("classnames"), n, st->atoms.data());
}

// ---------------------------------------------------------------------------
// Creation arguments
// ---------------------------------------------------------------------------

static bool is_flag(const t_atom& a) {
    return a.a_type == A_SYMBOL
        && (a.a_w.w_symbol->s_name[0] == '-' || a.a_w.w_symbol->s_name[0] == '@')
        && a.a_w.w_symbol->s_name[1] != '\0';
}

static ipt_device parse_device(const t_atom& a) {
    if (a.a_type == A_SYMBOL) {
        std::string d = a.a_w.w_symbol->s_name;
        std::transform(d.begin(), d.end(), d.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (d == "CPU")  return IPT_DEVICE_CPU;
        if (d == "CUDA") return IPT_DEVICE_CUDA;
        if (d == "MPS")  return IPT_DEVICE_MPS;
        pd_error(nullptr, "ipt~: unknown device type \"%s\", defaulting to CPU", a.a_w.w_symbol->s_name);
        return IPT_DEVICE_CPU;
    }
    // integer form, same mapping as the Max ipt~: 0 = CPU, 1 = CUDA, 2 = MPS
    switch (static_cast<int>(a.a_w.w_float)) {
        case 0: return IPT_DEVICE_CPU;
        case 1: return IPT_DEVICE_CUDA;
        case 2: return IPT_DEVICE_MPS;
        default:
            pd_error(nullptr, "ipt~: unknown device type \"%d\", defaulting to CPU",
                     static_cast<int>(a.a_w.w_float));
            return IPT_DEVICE_CPU;
    }
}

static bool is_regular_file(const std::string& path) {
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && S_ISREG(sb.st_mode);
}

// Folder ipt~ was loaded from (e.g. ~/Documents/Pd/externals/ipt~), or "".
static std::string external_dir() {
    const char* dir = ipt_class ? class_gethelpdir(ipt_class) : nullptr;
    return dir ? dir : "";
}

// Absolute path, "~/..." or a name found next to the patch, through [declare],
// in Pd's search paths or, last, in the ipt~ folder itself (as models in the
// Max package's help folder are found by name). ".ts" is appended when missing.
static bool resolve_model_path(t_canvas* canvas, std::string name, std::string& out) {
    if (name.size() < 3 || name.compare(name.size() - 3, 3, ".ts") != 0) {
        name += ".ts";
    }
    if (name.size() > 1 && name[0] == '~' && name[1] == '/') {
        if (const char* home = std::getenv("HOME")) {
            name = std::string(home) + name.substr(1);
        }
    }
    if (sys_isabsolutepath(name.c_str())) {
        if (!is_regular_file(name)) return false;
        out = name;
        return true;
    }
    char dir[MAXPDSTRING];
    char* base = nullptr;
    const int fd = canvas_open(canvas, name.c_str(), "", dir, &base, MAXPDSTRING, 1);
    if (fd >= 0) {
        sys_close(fd);
        out = std::string(dir) + "/" + base;
        return true;
    }
    const std::string own_dir = external_dir();
    if (!own_dir.empty() && is_regular_file(own_dir + "/" + name)) {
        out = own_dir + "/" + name;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// new / free
// ---------------------------------------------------------------------------

static void* ipt_new(t_symbol*, int argc, t_atom* argv) {
    // ipt~ <model> [device] [-setting value ...]   (@setting is accepted too)
    if (argc < 1 || argv[0].a_type != A_SYMBOL || is_flag(argv[0])) {
        pd_error(nullptr, "ipt~: missing argument: TorchScript model file (.ts)");
        return nullptr;
    }

    std::string model_path;
    t_canvas*   canvas = canvas_getcurrent();
    const char* model_arg = argv[0].a_w.w_symbol->s_name;
    if (!resolve_model_path(canvas, model_arg, model_path)) {
        if (sys_isabsolutepath(model_arg) || (model_arg[0] == '~' && model_arg[1] == '/')) {
            pd_error(nullptr, "ipt~: model file not found: %s", model_arg);
        } else {
            // Say where it looked: an unsaved patch's folder is often not the expected one.
            const t_symbol* patch_dir = canvas ? canvas_getdir(canvas) : nullptr;
            pd_error(nullptr, "ipt~: model file not found: %s (looked in the patch folder %s, "
                              "in Pd's search paths and in the ipt~ folder %s)",
                     model_arg, patch_dir ? patch_dir->s_name : "?", external_dir().c_str());
        }
        return nullptr;
    }

    int i = 1;
    ipt_device device = IPT_DEVICE_CPU;
    if (i < argc && !is_flag(argv[i])) {
        device = parse_device(argv[i]);
        ++i;
    }
    // Patches move between Macs and PCs: a device this platform cannot have
    // runs on the CPU instead of failing the model load.
#if defined(__APPLE__)
    if (device == IPT_DEVICE_CUDA) {
        pd_error(nullptr, "ipt~: CUDA is not available on macOS, using CPU");
        device = IPT_DEVICE_CPU;
    }
#else
    if (device == IPT_DEVICE_MPS) {
        pd_error(nullptr, "ipt~: MPS is only available on macOS, using CPU");
        device = IPT_DEVICE_CPU;
    }
#endif

    ipt_classifier* clf = ipt_create(model_path.c_str(), device, kDefaultThresholdDb, kDefaultWindowMs);
    if (!clf) {
        pd_error(nullptr, "ipt~: %s", ipt_last_error());
        return nullptr;
    }

    auto* x = reinterpret_cast<t_ipt*>(pd_new(ipt_class));
    x->x_st = new IptState;
    x->x_st->clf        = clf;
    x->x_st->device     = device;
    x->x_st->model_path = model_path;
    x->x_st->last_output = clock_getlogicaltime();
    update_smoothing(x->x_st, (1.0 - x->x_st->sensitivity) * x->x_st->range_ms);

    x->x_out_index = outlet_new(&x->x_obj, &s_float);
    x->x_out_name  = outlet_new(&x->x_obj, &s_symbol);
    x->x_out_dist  = outlet_new(&x->x_obj, &s_list);
    x->x_out_info  = outlet_new(&x->x_obj, nullptr);
    x->x_deliver_clock = clock_new(x, reinterpret_cast<t_method>(ipt_deliver));
    x->x_period_clock  = clock_new(x, reinterpret_cast<t_method>(ipt_period_tick));
    x->x_poll_clock    = clock_new(x, reinterpret_cast<t_method>(ipt_poll));

    // Creation flags, applied through the same setters as the messages.
    while (i < argc) {
        if (!is_flag(argv[i])) {
            pd_error(x, "ipt~: unexpected creation argument, expected -setting value");
            ++i;
            continue;
        }
        const char* name = argv[i].a_w.w_symbol->s_name + 1;
        const Setting* setting = find_setting(name);
        if (!setting) {
            pd_error(x, "ipt~: unknown setting \"%s\"", name);
            ++i;
            continue;
        }
        if (i + 1 >= argc || argv[i + 1].a_type != A_FLOAT) {
            pd_error(x, "ipt~: setting \"%s\" expects a number", name);
            ++i;
            continue;
        }
        setting->set(x, argv[i + 1].a_w.w_float);
        i += 2;
    }

    // The model loads on the worker; this returns before it is ready.
    {
        std::lock_guard<std::mutex> lock(live_mutex());
        x->x_st->worker = std::thread(worker_main, x->x_st);
        live_states().push_back(x->x_st);
    }
    clock_delay(x->x_poll_clock, kLoadPollMs);

    return x;
}

static void ipt_free(t_ipt* x) {
    if (IptState* st = x->x_st) {
        x->x_st = nullptr;
        // Returns at once: the worker is joined and the handle destroyed on a
        // detached thread (see "Freeing an object" above).
        st->stop.store(true, std::memory_order_release);
        bool cleanup_here = false;
        {
            std::lock_guard<std::mutex> lock(live_mutex());
            if (!st->join_claimed) {  // otherwise Pd is exiting and stop_all_workers has it
                st->join_claimed = true;
                ++cleanups_running();
                try {
                    std::thread(cleanup_state, st).detach();
                } catch (const std::system_error&) {
                    cleanup_here = true;  // no thread available: fall back to blocking
                }
            }
        }
        if (cleanup_here) {
            cleanup_state(st);
        }
    }
    if (x->x_deliver_clock) clock_free(x->x_deliver_clock);
    if (x->x_period_clock)  clock_free(x->x_period_clock);
    if (x->x_poll_clock)    clock_free(x->x_poll_clock);
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

extern "C" __attribute__((visibility("default"))) void ipt_tilde_setup(void) {
    ipt_class = class_new(gensym("ipt~"),
                          reinterpret_cast<t_newmethod>(ipt_new),
                          reinterpret_cast<t_method>(ipt_free),
                          sizeof(t_ipt), CLASS_DEFAULT, A_GIMME, 0);
    CLASS_MAINSIGNALIN(ipt_class, t_ipt, x_f);
    class_addmethod(ipt_class, reinterpret_cast<t_method>(ipt_dsp), gensym("dsp"), A_CANT, 0);
    for (const auto& s : kSettings) {
        class_addmethod(ipt_class, reinterpret_cast<t_method>(ipt_setting), gensym(s.name), A_GIMME, 0);
    }
    class_addmethod(ipt_class, reinterpret_cast<t_method>(ipt_classnames), gensym("classnames"), A_NULL);

    // Registered after libipt and libtorch are loaded: runs before their
    // load-time statics are destroyed. See stop_all_workers.
    std::atexit(stop_all_workers);

    post(" ipt~ v%s (2026) for Pure Data, by Nicolas Brochec", IPT_PACKAGE_VERSION);
    post(" based on original work by Nicolas Brochec, Joakim Borg, and Marco Fiorini");
    post(" IRCAM, RepMus REACH team");
}
