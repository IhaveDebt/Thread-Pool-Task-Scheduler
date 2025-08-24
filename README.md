#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

// -----------------------------
// Thread-safe queue
// -----------------------------
template <class T>
class TSQueue {
public:
    void push(T v) {
        {
            std::lock_guard<std::mutex> lk(m_);
            q_.push(std::move(v));
        }
        cv_.notify_one();
    }

    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    bool wait_pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.empty();
    }

private:
    mutable std::mutex m_;
    std::queue<T> q_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// -----------------------------
// ThreadPool with submit()
// -----------------------------
class ThreadPool {
public:
    explicit ThreadPool(std::size_t nthreads = std::thread::hardware_concurrency())
        : stop_(false)
    {
        if (nthreads == 0) nthreads = 1;
        workers_.reserve(nthreads);
        for (std::size_t i = 0; i < nthreads; ++i) {
            workers_.emplace_back([this]{
                for (;;) {
                    std::function<void()> task;
                    if (!tasks_.wait_pop(task)) return; // stopped
                    try {
                        task();
                    } catch (const std::exception& e) {
                        // You can hook a logger here
                        std::lock_guard<std::mutex> lk(io_m_);
                        std::cerr << "[ThreadPool] Task threw: " << e.what() << "\n";
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(io_m_);
                        std::cerr << "[ThreadPool] Task threw unknown exception\n";
                    }
                }
            });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        shutdown();
    }

    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using R = std::invoke_result_t<F, Args...>;
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<R> fut = task_ptr->get_future();
        tasks_.push([task_ptr]{ (*task_ptr)(); });
        return fut;
    }

    void shutdown() {
        bool expected = false;
        if (!stop_.compare_exchange_strong(expected, true)) return;
        tasks_.stop();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

private:
    TSQueue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_;
    std::mutex io_m_;
};

// -----------------------------
// Simple Scheduler
//  - schedule_after: run once after delay
//  - schedule_every: run periodically (until stop_token set)
// -----------------------------
class Scheduler {
public:
    explicit Scheduler(ThreadPool& pool)
        : pool_(pool), stop_(false)
    {}

    ~Scheduler() { stop(); }

    template <class F>
    void schedule_after(std::chrono::milliseconds delay, F&& f) {
        std::lock_guard<std::mutex> lk(m_);
        timers_.emplace(Clock::now() + delay,
                        wrap_(std::forward<F>(f)));
        cv_.notify_one();
        ensure_worker_();
    }

    template <class F>
    // returns a token you can use to stop the periodic task
    std::shared_ptr<std::atomic<bool>> schedule_every(std::chrono::milliseconds period, F&& f, std::chrono::milliseconds initial_delay = 0ms) {
        auto token = std::make_shared<std::atomic<bool>>(false);
        auto periodic = [this, period, fn = wrap_(std::forward<F>(f)), token]() mutable {
            if (token->load()) return; // cancelled
            fn();
            if (token->load()) return;
            schedule_after(period, [this, fn, period, token]{ 
                if (!token->load()) fn(); 
                if (!token->load()) schedule_after(period, [this, fn, period, token]{ if (!token->load()) fn(); });
            });
        };
        schedule_after(initial_delay == 0ms ? period : initial_delay, periodic);
        return token;
    }

    void stop() {
        bool expected = false;
        if (!stop_.compare_exchange_strong(expected, true)) return;
        {
            std::lock_guard<std::mutex> lk(m_);
            while (!timers_.empty()) timers_.pop();
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    using Clock = std::chrono::steady_clock;
    using Task = std::function<void()>;
    using TimedTask = std::pair<Clock::time_point, Task>;

    struct Cmp {
        bool operator()(const TimedTask& a, const TimedTask& b) const {
            return a.first > b.first;
        }
    };

    Task wrap_(Task t) {
        return [this, t=std::move(t)](){ pool_.submit(t); };
    }

    template <class F>
    Task wrap_(F&& f) {
        return [this, fn = std::forward<F>(f)](){ pool_.submit(fn); };
    }

    void ensure_worker_() {
        if (has_worker_) return;
        has_worker_ = true;
        worker_ = std::thread([this]{
            std::unique_lock<std::mutex> lk(m_);
            while (!stop_.load()) {
                if (timers_.empty()) {
                    cv_.wait(lk, [&]{ return stop_.load() || !timers_.empty(); });
                    if (stop_.load()) break;
                } else {
                    auto now = Clock::now();
                    auto [tp, task] = timers_.top();
                    if (tp <= now) {
                        timers_.pop();
                        lk.unlock();
                        // execute without holding lock
                        task();
                        lk.lock();
                    } else {
                        cv_.wait_until(lk, tp);
                    }
                }
            }
        });
    }

    ThreadPool& pool_;
    std::priority_queue<TimedTask, std::vector<TimedTask>, Cmp> timers_;
    std::thread worker_;
    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
    bool has_worker_ = false;
};

// -----------------------------
// Demo / Benchmark
// -----------------------------
int main() {
    std::cout << "ThreadPoolScheduler demo\n";
    const std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "Starting with " << threads << " worker threads\n";

    ThreadPool pool(threads);
    Scheduler sched(pool);

    // 1) Basic submit with futures
    auto f1 = pool.submit([]{
        std::this_thread::sleep_for(200ms);
        return std::string("Task 1 complete");
    });

    auto f2 = pool.submit([](int x){ return x * x; }, 12);

    // 2) Delayed one-shot task
    std::promise<void> delayed_done;
    auto delayed_fut = delayed_done.get_future();
    sched.schedule_after(500ms, [p = std::move(delayed_done)]() mutable {
        std::cout << "[after 500ms] delayed task ran\n";
        p.set_value();
    });

    // 3) Periodic task (prints heartbeat)
    auto token = sched.schedule_every(300ms, []{
        static std::atomic<int> count{0};
        int c = ++count;
        std::cout << "[tick] " << c << "\n";
    }, 100ms);

    // 4) Mini benchmark: sum of heavy computations
    auto heavy = [](int n){
        // trivial CPU work
        std::uint64_t s = 0;
        for (int i = 1; i <= n * 100000; ++i) s += (i % 101);
        return s;
    };

    const int jobs = 12;
    std::vector<std::future<std::uint64_t>> futs;
    futs.reserve(jobs);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < jobs; ++i) {
        futs.push_back(pool.submit(heavy, 500 + (i % 5)));
    }
    std::uint64_t total = 0;
    for (auto& f : futs) total += f.get();
    auto t1 = std::chrono::steady_clock::now();

    // Wait for basics
    std::cout << f1.get() << "\n";
    std::cout << "Task 2 result: " << f2.get() << "\n";
    delayed_fut.wait();

    // Stop periodic after a moment
    std::this_thread::sleep_for(1s);
    token->store(true);

    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Heavy jobs total: " << total << " in " << dt << " ms\n";

    // Clean shutdown
    sched.stop();
    pool.shutdown();
    std::cout << "Done.\n";
    return 0;
}
