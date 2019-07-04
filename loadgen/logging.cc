/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Implements a logging system with a central IO thread that handles
// all stringification and IO.
// Log-producing threads only submit lambdas to be executed on the IO thread.
// All producers and consumers use lock-free operations that guarantee
// forward progress independent of a) other stalled threads and b) where
// those threads are stalled.
// Each thread uses a double-buffering scheme to queue its logs. One buffer
// is always reserved for writes and the other is reserved for reads.
// A producing thread sends requests to the IOThread to swap the buffers
// and the IOThread does the actual read/write swap after it has finished
// reading the buffer it was working on.

#include "logging.h"

#include <cassert>
#include <future>
#include <iostream>
#include <sstream>

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#include <process.h>
#define MLPERF_GET_PID() _getpid()
#else
#include <unistd.h>
#define MLPERF_GET_PID() getpid()
#endif

#include "utils.h"

namespace mlperf {

namespace {

uintptr_t SwapRequestSlotIsWritableValue(size_t id) {
  // LSB of 1 indicates that this isn't a pointer.
  // MSBs encode the id to detect collisions when a slot in
  // |thread_swap_request_slots_| is reused for a different id and the request
  // for the previous id is very slow.
  return (id << 1) | 0x1;
}

bool SwapRequestSlotIsReadable(uintptr_t value) {
  // Valid pointers will not have their lsb set.
  return (value & 0x1) != 0x1;
}

constexpr size_t kMaxThreadsToLog = 1024;
constexpr std::chrono::milliseconds kLogPollPeriod(10);

constexpr auto kInvalidLatency = std::numeric_limits<QuerySampleLatency>::min();

}  // namespace

const std::string& ArgValueTransform(const bool& value) {
  static const std::string v_true("true");
  static const std::string v_false("false");
  return value ? v_true : v_false;
}

char Bin2Hex(uint8_t four_bits) {
  char number = '0' + four_bits;
  char letter = ('A' - 10) + four_bits;
  return four_bits < 10 ? number : letter;
}

const std::string ArgValueTransform(const LogBinaryAsHexString& value) {
  if (value.data == nullptr) {
    return "\"\"";
  }
  std::string hex;
  hex.reserve(value.data->size() + 2);
  hex.push_back('"');
  for (auto b : *value.data) {
    hex.push_back(Bin2Hex(b >> 4));
    hex.push_back(Bin2Hex(b & 0x0F));
  }
  hex.push_back('"');
  return hex;
}

void AsyncLog::RecordLatency(uint64_t sample_sequence_id,
                             QuerySampleLatency latency) {
  // Relaxed memory order since the early-out checks are inherently racy.
  // The final check will be ordered by locks on the latencies_mutex.
  max_latency_.store(
      std::max(max_latency_.load(std::memory_order_relaxed), latency),
      std::memory_order_relaxed);

  std::unique_lock<std::mutex> lock(latencies_mutex_);

  if (sample_sequence_id < latencies_first_sample_sequence_id_) {
    // Call LogErrorSync here since this kind of error could result in a
    // segfault in the near future.
    GlobalLogger().LogErrorSync(
        "Received completion for an old sample.", "Min expected id",
        latencies_first_sample_sequence_id_, "Actual id", sample_sequence_id);
    return;
  }

  const size_t i = sample_sequence_id - latencies_first_sample_sequence_id_;

  if (latencies_.size() <= i) {
    // TODO: Reserve in advance.
    latencies_.resize(i + 1, kInvalidLatency);
  } else if (latencies_[i] != kInvalidLatency) {
    // Call LogErrorSync here since this kind of error could result in a
    // segfault in the near future.
    GlobalLogger().LogErrorSync("Attempted to complete a sample twice.");

    // Return without recording the latency again to avoid potentially
    // ending the test before the SUT is actually done, which could result
    // in a segfault.
    // If the SUT recorded the wrong sample, the test will hang and see
    // the error above.
    return;
  }

  latencies_[i] = latency;
  latencies_recorded_++;
  if (AllLatenciesRecorded()) {
    all_latencies_recorded_.notify_all();
  }
}

void AsyncLog::RestartLatencyRecording(uint64_t first_sample_sequence_id) {
  std::unique_lock<std::mutex> lock(latencies_mutex_);
  assert(latencies_.empty());
  assert(latencies_recorded_ == latencies_expected_);
  latencies_recorded_ = 0;
  latencies_expected_ = 0;
  max_latency_ = 0;
  latencies_first_sample_sequence_id_ = first_sample_sequence_id;
}

std::vector<QuerySampleLatency> AsyncLog::GetLatenciesBlocking(
    size_t expected_count) {
  std::vector<QuerySampleLatency> latencies;
  {
    std::unique_lock<std::mutex> lock(latencies_mutex_);
    latencies_expected_ = expected_count;
    all_latencies_recorded_.wait(lock, [&] { return AllLatenciesRecorded(); });
    latencies.swap(latencies_);
  }

  if (latencies.size() != expected_count) {
    // Call LogErrorSync here since this kind of error could result in a
    // segfault in the near future.
    GlobalLogger().LogErrorSync("Received SequenceId that was too large.",
                                "expectted_size", expected_count, "actual_size",
                                latencies.size());
  }

  size_t invalid_latency_count = 0;
  for (auto l : latencies) {
    if (l == kInvalidLatency) {
      invalid_latency_count++;
    }
  }
  if (invalid_latency_count != 0) {
    // Call LogErrorSync here since this kind of error could result in a
    // segfault in the near future.
    GlobalLogger().LogErrorSync(
        "Encountered incomplete samples at the end of a series of queries.",
        "count", invalid_latency_count);
  }

  return latencies;
}

QuerySampleLatency AsyncLog::GetMaxLatencySoFar() {
  return max_latency_.load(std::memory_order_release);
}

// TlsLogger logs a single thread using thread-local storage.
// Submits logs to the central Logger:
//   * With forward-progress guarantees. (i.e.: no locking or blocking
//       operations even if other threads have stalled.
//   * Without expensive syscalls or I/O operations.
class TlsLogger {
 public:
  TlsLogger(std::function<void()> forced_detatch);
  ~TlsLogger();
  void ForcedDetatchFromThread() { forced_detatch_(); }

  void Log(AsyncLogEntry&& entry);
  void SwapBuffers();

  std::vector<AsyncLogEntry>* StartReadingEntries();
  void FinishReadingEntries();
  bool ReadBufferHasBeenConsumed();

  const std::string* TracePidTidString() const { return &trace_pid_tid_; }

  const std::string* TidAsString() const { return &tid_as_string_; }

  void RequestSwapBuffersSlotRetried() {
    swap_buffers_slot_retry_count_.fetch_add(1, std::memory_order_relaxed);
  }

  size_t ReportLogCasFailCount() {
    size_t c = log_cas_fail_count_.load(std::memory_order_relaxed);
    log_cas_fail_count_.fetch_sub(c, std::memory_order_relaxed);
    return c;
  }

  size_t ReportSwapBuffersSlotRetryCount() {
    size_t c = swap_buffers_slot_retry_count_.load(std::memory_order_relaxed);
    swap_buffers_slot_retry_count_.fetch_sub(c, std::memory_order_relaxed);
    return c;
  }

  void TraceCounters();

 private:
  using EntryVector = std::vector<AsyncLogEntry>;
  enum class EntryState { Unlocked, ReadLock, WriteLock };

  // Accessed by producer only.
  size_t i_read_ = 0;

  // Accessed by producer and consumer atomically.
  EntryVector entries_[2];
  std::atomic<EntryState> entry_states_[2]{{EntryState::ReadLock},
                                           {EntryState::Unlocked}};
  std::atomic<size_t> i_write_{1};

  std::atomic<size_t> log_cas_fail_count_{0};
  std::atomic<size_t> swap_buffers_slot_retry_count_{0};

  // Accessed by consumer only.
  size_t unread_swaps_ = 0;
  size_t i_write_prev_ = 0;
  std::string trace_pid_tid_;  // Cached as string.
  std::string tid_as_string_;  // Cached as string.

  std::function<void()> forced_detatch_;
};

Logger::Logger(std::chrono::duration<double> poll_period,
               size_t max_threads_to_log)
    : poll_period_(poll_period),
      max_threads_to_log_(max_threads_to_log),
      thread_swap_request_slots_(max_threads_to_log * 2) {
  const size_t kSlotCount = max_threads_to_log * 2;
  for (size_t i = 0; i < kSlotCount; i++) {
    std::atomic_init(&thread_swap_request_slots_[i],
                     SwapRequestSlotIsWritableValue(i));
  }
}

Logger::~Logger() {
  // TlsLoggers might outlive this Logger when loaded as a python module.
  // Forcefully make all currently registered TlsLoggers orphans.
  std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
  TlsLogger* tls_logger_prev = nullptr;
  (void)tls_logger_prev;  // Avoid unused error in release builds.
  while (!tls_loggers_registerd_.empty()) {
    TlsLogger* tls_logger = *tls_loggers_registerd_.begin();
    // Otherwise, this is an infinite loop.
    assert(tls_logger != tls_logger_prev);
    tls_loggers_registerd_mutex_.unlock();
    tls_logger->ForcedDetatchFromThread();
    tls_loggers_registerd_mutex_.lock();
    tls_logger_prev = tls_logger;
  }
}

void Logger::RequestSwapBuffers(TlsLogger* tls_logger) {
  auto tls_logger_as_uint = reinterpret_cast<uintptr_t>(tls_logger);
  assert(SwapRequestSlotIsReadable(tls_logger_as_uint));
  size_t id, slot;
  uintptr_t slot_is_writeable_value;
  // The compare_exchange below should almost always succeed.
  // The compare_exchange may fail if a recycled slot is still actively used
  // by another thread, so we retry with subsequent slots here if needed.
  // Since the slot count is 2x the expected number of threads to log,
  // the CAS should only fail at most 50% of the time when all logging threads
  // happen to be descheduled between the fetch_add and CAS below, which is
  // very unlikely.
  id = swap_request_id_.fetch_add(1, std::memory_order_relaxed);
  slot = id % thread_swap_request_slots_.size();
  slot_is_writeable_value = SwapRequestSlotIsWritableValue(id);
  while (!thread_swap_request_slots_[slot].compare_exchange_strong(
      slot_is_writeable_value, tls_logger_as_uint, std::memory_order_release)) {
    id = swap_request_id_.fetch_add(1, std::memory_order_relaxed);
    slot = id % thread_swap_request_slots_.size();
    slot_is_writeable_value = SwapRequestSlotIsWritableValue(id);
    tls_logger->RequestSwapBuffersSlotRetried();
  }
}

void Logger::RegisterTlsLogger(TlsLogger* tls_logger) {
  std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
  if (tls_loggers_registerd_.size() >= max_threads_to_log_) {
    LogErrorSync(
        "Warning: More TLS loggers registerd than can "
        "be active simultaneously.\n");
  }
  tls_loggers_registerd_.insert(tls_logger);
}

// This moves ownership of the tls_logger data to Logger so the
// exiting thread can exit immediately, even if all the logs of the
// exiting thread haven't been processed.
void Logger::UnRegisterTlsLogger(std::unique_ptr<TlsLogger> tls_logger) {
  OrphanContainer::iterator orphan;
  {
    std::unique_lock<std::mutex> lock(tls_logger_orphans_mutex_);
    tls_logger_orphans_.emplace_front(std::move(tls_logger));
    orphan = tls_logger_orphans_.begin();
  }

  // Only remove the TlsLogger from the registry after adding to orphans so
  // CollectTlsLoggerStats doesn't have any gaps in coverage.
  {
    std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
    tls_loggers_registerd_.erase(orphan->get());
  }

  // This will flush the logs of |tls_logger| and mark it for destruction.
  // Deferring destruction via orphans_to_destroy helps avoid use-after-frees
  // when the IOThread calls FinishReadingEntries.
  (*orphan)->Log([this, orphan](AsyncLog&) {
    CollectTlsLoggerStats(orphan->get());
    orphans_to_destroy_.push_back(orphan);
  });
}

void Logger::CollectTlsLoggerStats(TlsLogger* tls_logger) {
  tls_total_log_cas_fail_count_ += tls_logger->ReportLogCasFailCount();
  tls_total_swap_buffers_slot_retry_count_ +=
      tls_logger->ReportSwapBuffersSlotRetryCount();
}

void Logger::StartIOThread() {
  {
    std::unique_lock<std::mutex> lock(io_thread_mutex_);
    keep_io_thread_alive_ = true;
  }
  io_thread_ = std::thread(&Logger::IOThread, this);
}

void Logger::StopIOThread() {
  {
    std::unique_lock<std::mutex> lock(io_thread_mutex_);
    keep_io_thread_alive_ = false;
    io_thread_cv_.notify_all();
  }
  io_thread_.join();
}

void Logger::StartLogging(std::ostream* summary, std::ostream* detail,
                          std::ostream* accuracy, bool copy_detail_to_stdout,
                          bool copy_summary_to_stdout) {
  async_logger_.SetLogFiles(summary, detail, accuracy, copy_detail_to_stdout,
                            copy_summary_to_stdout, PerfClock::now());
}

void Logger::StopLogging() {
  if (std::this_thread::get_id() == io_thread_.get_id()) {
    LogErrorSync("StopLogging() not supported from IO thread.");
    return;
  }

  // Flush logs from this thread.
  std::promise<void> io_thread_flushed_this_thread;
  Log([&](AsyncLog&) { io_thread_flushed_this_thread.set_value(); });
  io_thread_flushed_this_thread.get_future().wait();
  async_logger_.SetLogFiles(&std::cerr, &std::cerr, &std::cerr, false, false,
                            PerfClock::now());
}

void Logger::StartNewTrace(std::ostream* trace_out,
                           PerfClock::time_point origin) {
  async_logger_.StartNewTrace(trace_out, origin);
}

void Logger::StopTracing() {
  // Flush traces from this thread.
  std::promise<void> io_thread_flushed_this_thread;
  Log([&](AsyncLog&) { io_thread_flushed_this_thread.set_value(); });
  io_thread_flushed_this_thread.get_future().wait();
  async_logger_.StartNewTrace(nullptr, PerfClock::now());
}

void Logger::LogContentionCounters() {
  LogDetail([&](AsyncLog& log) {
    {
      std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
      for (auto tls_logger : tls_loggers_registerd_) {
        CollectTlsLoggerStats(tls_logger);
      }
    }

    {
      std::unique_lock<std::mutex> lock(tls_logger_orphans_mutex_);
      for (auto& orphan : tls_logger_orphans_) {
        CollectTlsLoggerStats(orphan.get());
      }
    }

    log.LogDetail("Log Contention Counters:");
    log.LogDetail(std::to_string(swap_request_slots_retry_count_) +
                  " : swap_request_slots_retry_count");
    log.LogDetail(std::to_string(swap_request_slots_retry_retry_count_) +
                  " : swap_request_slots_retry_retry_count");
    log.LogDetail(std::to_string(swap_request_slots_retry_reencounter_count_) +
                  " : swap_request_slots_retry_reencounter_count");
    log.LogDetail(std::to_string(start_reading_entries_retry_count_) +
                  " : start_reading_entries_retry_count");
    log.LogDetail(std::to_string(tls_total_log_cas_fail_count_) +
                  " : tls_total_log_cas_fail_count");
    log.LogDetail(std::to_string(tls_total_swap_buffers_slot_retry_count_) +
                  " : tls_total_swap_buffers_slot_retry_count");

    swap_request_slots_retry_count_ = 0;
    swap_request_slots_retry_retry_count_ = 0;
    swap_request_slots_retry_reencounter_count_ = 0;
    start_reading_entries_retry_count_ = 0;
    tls_total_log_cas_fail_count_ = 0;
    tls_total_swap_buffers_slot_retry_count_ = 0;
  });
}

void Logger::RestartLatencyRecording(uint64_t first_sample_sequence_id) {
  async_logger_.RestartLatencyRecording(first_sample_sequence_id);
}

std::vector<QuerySampleLatency> Logger::GetLatenciesBlocking(
    size_t expected_count) {
  return async_logger_.GetLatenciesBlocking(expected_count);
}

QuerySampleLatency Logger::GetMaxLatencySoFar() {
  return async_logger_.GetMaxLatencySoFar();
}

TlsLogger* Logger::GetTlsLoggerThatRequestedSwap(size_t slot, size_t next_id) {
  uintptr_t slot_value = thread_swap_request_slots_[slot].load();
  if (SwapRequestSlotIsReadable(slot_value)) {
    // TODO: Convert this block to a simple write once we are confidient
    // that we don't need to check for success.
    bool success = thread_swap_request_slots_[slot].compare_exchange_strong(
        slot_value, SwapRequestSlotIsWritableValue(next_id));
    if (!success) {
      GlobalLogger().LogErrorSync("CAS failed.", "line", __LINE__);
      assert(success);
    }
    return reinterpret_cast<TlsLogger*>(slot_value);
  }
  return nullptr;
}

void Logger::GatherRetrySwapRequests(std::vector<TlsLogger*>* threads_to_swap) {
  if (swap_request_slots_to_retry_.empty()) {
    return;
  }

  std::vector<SlotRetry> retry_slots;
  retry_slots.swap(swap_request_slots_to_retry_);
  for (auto& slot_retry : retry_slots) {
    TlsLogger* tls_logger =
        GetTlsLoggerThatRequestedSwap(slot_retry.slot, slot_retry.next_id);
    if (tls_logger) {
      threads_to_swap->push_back(tls_logger);
    } else {
      swap_request_slots_to_retry_.push_back(slot_retry);
      swap_request_slots_retry_retry_count_++;
    }
  }
}

void Logger::GatherNewSwapRequests(std::vector<TlsLogger*>* threads_to_swap) {
  auto swap_request_end = swap_request_id_.load(std::memory_order_acquire);
  for (; swap_request_id_read_ < swap_request_end; swap_request_id_read_++) {
    size_t slot = swap_request_id_read_ % thread_swap_request_slots_.size();
    size_t next_id = swap_request_id_read_ + thread_swap_request_slots_.size();
    TlsLogger* tls_logger = GetTlsLoggerThatRequestedSwap(slot, next_id);
    if (tls_logger) {
      threads_to_swap->push_back(tls_logger);
    } else {
      swap_request_slots_retry_count_++;
      // A thread is in the middle of its call to RequestSwapBuffers.
      // Retry later once it's done.
      auto it = std::find_if(swap_request_slots_to_retry_.begin(),
                             swap_request_slots_to_retry_.end(),
                             [=](SlotRetry& s) { return s.slot == slot; });
      if (it == swap_request_slots_to_retry_.end()) {
        // This is the first time we are retrying the slot.
        swap_request_slots_to_retry_.push_back({slot, next_id});
      } else {
        // Whoa. We've been retrying this slot since the last time it was
        // encountered. Just update the next_id.
        it->next_id = next_id;
        swap_request_slots_retry_reencounter_count_++;
      }
    };
  }
}

void Logger::IOThread() {
  while (keep_io_thread_alive_) {
    auto trace1 = MakeScopedTracer(
        [](AsyncLog& log) { log.ScopedTrace("IOThreadLoop"); });
    {
      auto trace2 =
          MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("Wait"); });
      std::unique_lock<std::mutex> lock(io_thread_mutex_);
      io_thread_cv_.wait_for(lock, poll_period_,
                             [&] { return !keep_io_thread_alive_; });
    }

    {
      auto trace3 =
          MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("Gather"); });
      std::vector<TlsLogger*> threads_to_swap;
      threads_to_swap.swap(threads_to_swap_deferred_);
      GatherRetrySwapRequests(&threads_to_swap);
      GatherNewSwapRequests(&threads_to_swap);
      for (TlsLogger* thread : threads_to_swap) {
        if (thread->ReadBufferHasBeenConsumed()) {
          thread->SwapBuffers();
          // After swapping a thread, it's ready to be read.
          threads_to_read_.push_back(thread);
        } else {
          // Don't swap buffers again until we've finish reading the
          // previous swap.
          threads_to_swap_deferred_.push_back(thread);
        }
      }
    }

    {
      auto trace4 =
          MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("Process"); });
      // Read from the threads we are confident have activity.
      for (std::vector<TlsLogger*>::iterator thread = threads_to_read_.begin();
           thread != threads_to_read_.end(); thread++) {
        auto trace5 =
            MakeScopedTracer([tid = *(*thread)->TidAsString()](AsyncLog& log) {
              log.ScopedTrace("Thread", "tid", tid);
            });
        std::vector<AsyncLogEntry>* entries = (*thread)->StartReadingEntries();
        if (!entries) {
          start_reading_entries_retry_count_++;
          continue;
        }

        async_logger_.SetCurrentTracePidTidString(
            (*thread)->TracePidTidString());
        for (auto& entry : *entries) {
          // Execute the entry to perform the serialization and I/O.
          entry(async_logger_);
        }
        (*thread)->FinishReadingEntries();
        // Mark for removal by the call to RemoveValue below.
        *thread = nullptr;
      }

      // Only remove threads where reading succeeded so we retry the failed
      // threads the next time around.
      RemoveValue(&threads_to_read_, nullptr);
    }

    {
      auto trace6 =
          MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("FlushAll"); });
      async_logger_.Flush();
    }

    if (!orphans_to_destroy_.empty()) {
      auto trace7 = MakeScopedTracer(
          [](AsyncLog& log) { log.ScopedTrace("Abandoning Orphans"); });
      std::unique_lock<std::mutex> lock(tls_logger_orphans_mutex_);
      for (auto orphan : orphans_to_destroy_) {
        tls_logger_orphans_.erase(orphan);
      }
      orphans_to_destroy_.clear();
    }
  }
}

TlsLogger::TlsLogger(std::function<void()> forced_detatch)
    : forced_detatch_(std::move(forced_detatch)) {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  tid_as_string_ = ss.str();
  trace_pid_tid_ = "\"pid\": " + std::to_string(MLPERF_GET_PID()) + ", " +
                   "\"tid\": " + tid_as_string_ + ", ";
}

TlsLogger::~TlsLogger() {}

// Log always makes forward progress since it can unconditionally obtain a
// "lock" on at least one of the buffers for writting.
// Notificiation is also lock free.
void TlsLogger::Log(AsyncLogEntry&& entry) {
  size_t cas_fail_count = 0;
  auto unlocked = EntryState::Unlocked;
  size_t i_write = i_write_.load(std::memory_order_relaxed);
  while (!entry_states_[i_write].compare_exchange_strong(
      unlocked, EntryState::WriteLock, std::memory_order_acquire,
      std::memory_order_relaxed)) {
    unlocked = EntryState::Unlocked;
    i_write ^= 1;
    // We may need to try 3 times, since there could be a race with a
    // previous SwapBuffers request and we use memory_order_relaxed when
    // loading i_write_ above.
    cas_fail_count++;
    if (cas_fail_count >= 3) {
      GlobalLogger().LogErrorSync("CAS failed.", "times", cas_fail_count,
                                  "line", __LINE__);
      assert(cas_fail_count < 3);
    }
    log_cas_fail_count_.fetch_add(1, std::memory_order_relaxed);
  }
  entries_[i_write].emplace_back(std::forward<AsyncLogEntry>(entry));

  // TODO: Convert this block to a simple write once we are confidient
  // that we don't need to check for success.
  auto write_lock = EntryState::WriteLock;
  bool success = entry_states_[i_write].compare_exchange_strong(
      write_lock, EntryState::Unlocked, std::memory_order_release);
  if (!success) {
    GlobalLogger().LogErrorSync("CAS failed.", "line", __LINE__);
    assert(success);
  }

  bool write_buffer_swapped = i_write_prev_ != i_write;
  if (write_buffer_swapped) {
    GlobalLogger().RequestSwapBuffers(this);
    i_write_prev_ = i_write;
  }
}

void TlsLogger::SwapBuffers() {
  // TODO: Convert this block to a simple write once we are confidient
  // that we don't need to check for success.
  auto read_lock = EntryState::ReadLock;
  bool success = entry_states_[i_read_].compare_exchange_strong(
      read_lock, EntryState::Unlocked, std::memory_order_release);
  if (!success) {
    GlobalLogger().LogErrorSync("CAS failed.", "line", __LINE__);
    assert(success);
  }

  i_write_.store(i_read_, std::memory_order_relaxed);
  i_read_ ^= 1;
  unread_swaps_++;
}

// Returns nullptr if read lock fails.
std::vector<AsyncLogEntry>* TlsLogger::StartReadingEntries() {
  auto unlocked = EntryState::Unlocked;
  if (entry_states_[i_read_].compare_exchange_strong(
          unlocked, EntryState::ReadLock, std::memory_order_acquire,
          std::memory_order_relaxed)) {
    return &entries_[i_read_];
  }
  return nullptr;
}

void TlsLogger::FinishReadingEntries() {
  entries_[i_read_].clear();
  unread_swaps_--;
}

bool TlsLogger::ReadBufferHasBeenConsumed() { return unread_swaps_ == 0; }

void TlsLogger::TraceCounters() {
  auto trace = MakeScopedTracer(
      [lcfc = log_cas_fail_count_.load(std::memory_order_relaxed),
       sbsrc = swap_buffers_slot_retry_count_.load(std::memory_order_relaxed)](
          AsyncLog& log) {
        log.ScopedTrace("TlsLogger:ContentionCounters", "log_cas_fail_count",
                        lcfc, "swap_buffers_slot_retry_count", sbsrc);
      });
}

Logger& GlobalLogger() {
  static Logger g_logger(kLogPollPeriod, kMaxThreadsToLog);
  return g_logger;
}

// TlsLoggerWrapper moves ownership of the TlsLogger to Logger on thread exit
// so no round-trip synchronization with the IO thread is required.
struct TlsLoggerWrapper {
  TlsLoggerWrapper(std::function<void()> forced_detatch)
      : tls_logger(std::make_unique<TlsLogger>(std::move(forced_detatch))) {
    GlobalLogger().RegisterTlsLogger(tls_logger.get());
  }
  ~TlsLoggerWrapper() {
    tls_logger->TraceCounters();
    GlobalLogger().UnRegisterTlsLogger(std::move(tls_logger));
  }
  std::unique_ptr<TlsLogger> tls_logger;
};

TlsLoggerWrapper* InitializeMyTlsLoggerWrapper() {
  thread_local std::unique_ptr<TlsLoggerWrapper> tls_logger_wrapper;
  // forced_detatch lets the global Logger forcefully detatch TlsLoggers
  // from the thread in the Logger's destructor, which may run before
  // thread-local variables are destroyed when the loadgen is used as a python
  // module and dynamically unloaded.
  auto forced_detatch = [&]() { tls_logger_wrapper.reset(); };
  tls_logger_wrapper = std::make_unique<TlsLoggerWrapper>(forced_detatch);
  return tls_logger_wrapper.get();
}

TlsLogger* InitializeMyTlsLogger() {
  thread_local TlsLoggerWrapper* wrapper = InitializeMyTlsLoggerWrapper();
  return wrapper->tls_logger.get();
}

void Log(AsyncLogEntry&& entry) {
  thread_local TlsLogger* const tls_logger = InitializeMyTlsLogger();
  tls_logger->Log(std::forward<AsyncLogEntry>(entry));
}

}  // namespace mlperf
