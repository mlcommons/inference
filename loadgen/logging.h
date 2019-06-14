#ifndef MLPERF_LOADGEN_LOGGING_H_
#define MLPERF_LOADGEN_LOGGING_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "query_sample.h"

namespace mlperf {

class AsyncLog;
class Logger;
class TlsLogger;
class TlsLoggerWrapper;

using AsyncLogEntry = std::function<void(AsyncLog&)>;
using PerfClock = std::chrono::high_resolution_clock;

struct LogBinaryAsHexString {
  std::vector<uint8_t>* data;
};

const std::string& ArgValueTransform(const bool& value);
const std::string ArgValueTransform(const LogBinaryAsHexString& value);

template <typename T>
const T& ArgValueTransform(const T& value) {
  return value;
}

// AsyncLog is passed as an argument to the log lambda on the
// recording thread to serialize the data captured by the lambda and
// forward it to the output stream.
// TODO: Move non-templated methods to the cc file.
class AsyncLog {
 public:
  ~AsyncLog() { StartNewTrace(nullptr, PerfClock::now()); }

  void SetLogFiles(std::ostream* summary, std::ostream* detail,
                   std::ostream* accuracy, PerfClock::time_point log_origin) {
    std::unique_lock<std::mutex> lock(log_mutex_);
    if (summary_out_ != &std::cerr) {
      if (log_error_count_ == 0) {
        *summary_out_ << "\nNo errors encountered during test.\n";
      } else if (log_error_count_ == 1) {
        *summary_out_ << "\n1 ERROR encountered. See detailed log.\n";
      } else if (log_error_count_ != 0) {
        *summary_out_ << "\n"
                      << log_error_count_
                      << " ERRORS encountered. See detailed log.\n";
      }
    }
    if (summary_out_) {
      summary_out_->flush();
    }
    if (detail_out_) {
      detail_out_->flush();
    }
    if (accuracy_out_) {
      WriteAccuracyFooterLocked();
      accuracy_out_->flush();
    }
    summary_out_ = summary;
    detail_out_ = detail;
    accuracy_out_ = accuracy;
    if (accuracy_out_) {
      WriteAccuracyHeaderLocked();
    }
    log_origin_ = log_origin;
    log_error_count_ = 0;
  }

  void StartNewTrace(std::ostream* trace_out, PerfClock::time_point origin) {
    std::unique_lock<std::mutex> lock(trace_mutex_);
    // Cleanup previous trace.
    if (trace_out_) {
      WriteTraceEventFooterLocked();
      trace_out_->flush();
    }

    // Setup new trace.
    trace_out_ = trace_out;
    trace_origin_ = origin;
    if (trace_out_) {
      WriteTraceEventHeaderLocked();
    }
  }

  void Flush() {
    {
      std::unique_lock<std::mutex> lock(log_mutex_);
      if (summary_out_) {
        summary_out_->flush();
      }
      if (detail_out_) {
        detail_out_->flush();
      }
      if (accuracy_out_) {
        accuracy_out_->flush();
      }
    }

    {
      std::unique_lock<std::mutex> lock(trace_mutex_);
      if (trace_out_) {
        trace_out_->flush();
      }
    }
  }

  void SetCurrentTracePidTidString(const std::string* pid_tid) {
    current_pid_tid_ = pid_tid;
  }

  template <typename... Args>
  void LogSummary(const std::string& message, const Args... args) {
    auto trace = MakeScopedTracer([message](AsyncLog& log) {
      std::string sanitized_message = message;
      std::replace(sanitized_message.begin(), sanitized_message.end(), '"',
                   '\'');
      std::replace(sanitized_message.begin(), sanitized_message.end(), '\n',
                   ';');
      log.ScopedTrace("LogSummary", "message", "\"" + sanitized_message + "\"");
    });
    std::unique_lock<std::mutex> lock(log_mutex_);
    *summary_out_ << message;
    LogArgs(summary_out_, args...);
    *summary_out_ << "\n";
  }

  void SetLogDetailTime(PerfClock::time_point time) { log_detail_time_ = time; }

  // TODO: Warnings.
  void FlagError() {
    std::unique_lock<std::mutex> lock(log_mutex_);
    log_error_count_++;
    error_flagged_ = true;
  }

  template <typename... Args>
  void LogDetail(const std::string& message, const Args... args) {
    auto trace = MakeScopedTracer([message](AsyncLog& log) {
      std::string sanitized_message = message;
      std::replace(sanitized_message.begin(), sanitized_message.end(), '"',
                   '\'');
      std::replace(sanitized_message.begin(), sanitized_message.end(), '\n',
                   ';');
      log.ScopedTrace("LogDetail", "message", "\"" + sanitized_message + "\"");
    });
    std::unique_lock<std::mutex> lock(log_mutex_);
    *detail_out_ << *current_pid_tid_
                 << "\"ts\": " << (log_detail_time_ - log_origin_).count()
                 << "ns : ";
    if (error_flagged_) {
      *detail_out_ << "ERROR : ";
      error_flagged_ = false;
    }
    *detail_out_ << message;
    LogArgs(detail_out_, args...);
    *detail_out_ << "\n";
  }

  void LogAccuracy(uint64_t seq_id, const QuerySampleIndex qsl_idx,
                   const LogBinaryAsHexString& response) {
    std::unique_lock<std::mutex> lock(log_mutex_);
    if (!accuracy_out_) {
      return;
    }
    *accuracy_out_ << (accuracy_needs_comma_ ? ",\n{ " : "\n{ ");
    LogArgs(accuracy_out_, "seq_id", seq_id, "qsl_idx", qsl_idx, "data",
            response);
    *accuracy_out_ << " }";
    accuracy_needs_comma_ = true;
  }

  template <typename... Args>
  void Trace(const std::string& trace_name, PerfClock::time_point start,
             PerfClock::time_point end, const Args... args) {
    std::unique_lock<std::mutex> lock(trace_mutex_);
    if (!trace_out_) {
      return;
    }
    *trace_out_ << "{ \"name\": \"" << trace_name << "\", "
                << "\"ph\": \"X\", " << *current_pid_tid_
                << "\"ts\": " << (start - trace_origin_).count() << ", "
                << "\"dur\": " << (end - start).count() << ", "
                << "\"args\": { ";
    LogArgs(trace_out_, args...);
    *trace_out_ << " }},\n";
  }

  template <typename... Args>
  void TraceAsyncInstant(const std::string& trace_name, uint64_t id,
                         PerfClock::time_point instant_time,
                         const Args... args) {
    std::unique_lock<std::mutex> lock(trace_mutex_);
    if (!trace_out_) {
      return;
    }
    *trace_out_ << "{\"name\": \"" << trace_name << "\", "
                << "\"cat\": \"default\", "
                << "\"ph\": \"n\", "
                << "\"id\": " << id << ", " << *current_pid_tid_
                << "\"ts\": " << (instant_time - trace_origin_).count() << ", "
                << "\"args\": { ";
    LogArgs(trace_out_, args...);
    *trace_out_ << " }},\n";
  }

  void SetScopedTraceTimes(PerfClock::time_point start,
                           PerfClock::time_point end) {
    scoped_start_ = start;
    scoped_end_ = end;
    ;
  }

  template <typename... Args>
  void ScopedTrace(const std::string& trace_name, const Args... args) {
    std::unique_lock<std::mutex> lock(trace_mutex_);
    if (!trace_out_) {
      return;
    }
    *trace_out_ << "{ \"name\": \"" << trace_name << "\", "
                << "\"ph\": \"X\", " << *current_pid_tid_
                << "\"ts\": " << (scoped_start_ - trace_origin_).count() << ", "
                << "\"dur\": " << (scoped_end_ - scoped_start_).count() << ", "
                << "\"args\": { ";
    LogArgs(trace_out_, args...);
    *trace_out_ << " }},\n";
  }

  template <typename... Args>
  void TraceSample(const std::string& trace_name, uint64_t id,
                   PerfClock::time_point start, PerfClock::time_point end,
                   const Args... args) {
    std::unique_lock<std::mutex> lock(trace_mutex_);
    if (!trace_out_) {
      return;
    }
    *trace_out_ << "{\"name\": \"" << trace_name << "\", "
                << "\"cat\": \"default\", "
                << "\"ph\": \"b\", "
                << "\"id\": " << id << ", " << *current_pid_tid_
                << "\"ts\": " << (start - trace_origin_).count() << ", "
                << "\"args\": { ";
    LogArgs(trace_out_, args...);
    *trace_out_ << " }},\n";

    *trace_out_ << "{ \"name\": \"" << trace_name << "\", "
                << "\"cat\": \"default\", "
                << "\"ph\": \"e\", "
                << "\"id\": " << id << ", " << *current_pid_tid_
                << "\"ts\": " << (end - trace_origin_).count() << " },\n";
  }

  void RecordLatency(uint64_t sample_sequence_id, QuerySampleLatency latency) {
    std::unique_lock<std::mutex> lock(latencies_mutex_);
    if (latencies_.size() < sample_sequence_id + 1) {
      // TODO: Reserve in advance.
      latencies_.resize(sample_sequence_id + 1,
                        std::numeric_limits<QuerySampleLatency>::min());
    }
    latencies_[sample_sequence_id] = latency;
    latencies_recorded_++;
    if (AllLatenciesRecorded()) {
      all_latencies_recorded_.notify_all();
    }
    // Relaxed memory order since the early-out checks can be racy.
    // The final check will be ordered by locks on the latencies_mutex.
    max_latency_.store(
        std::max(max_latency_.load(std::memory_order_relaxed), latency),
        std::memory_order_relaxed);
  }

  void RestartLatencyRecording() {
    std::unique_lock<std::mutex> lock(latencies_mutex_);
    assert(latencies_.empty());
    assert(latencies_recorded_ == latencies_expected_);
    latencies_recorded_ = 0;
    latencies_expected_ = 0;
    max_latency_ = 0;
  }

  std::vector<QuerySampleLatency> GetLatenciesBlocking(size_t expected_count) {
    std::vector<QuerySampleLatency> latencies;
    std::unique_lock<std::mutex> lock(latencies_mutex_);
    latencies_expected_ = expected_count;
    all_latencies_recorded_.wait(lock, [&] { return AllLatenciesRecorded(); });
    latencies.swap(latencies_);
    return latencies;
  }

  QuerySampleLatency GetMaxLatencySoFar() {
    return max_latency_.load(std::memory_order_release);
  }

 private:
  void WriteAccuracyHeaderLocked() {
    *accuracy_out_ << "[";
    accuracy_needs_comma_ = false;
  }

  void WriteAccuracyFooterLocked() { *accuracy_out_ << "\n]\n"; }

  void WriteTraceEventHeaderLocked() {
    *trace_out_ << "{ \"traceEvents\": [\n";
  }

  void WriteTraceEventFooterLocked() {
    *trace_out_ << "{ \"name\": \"LastTrace\" }\n"
                << "],\n"
                << "\"displayTimeUnit\": \"ns\",\n"
                << "\"otherData\": {\n"
                << "\"version\": \"MLPerf LoadGen v0.5a0\"\n"
                << "}\n"
                << "}\n";
  }

  void LogArgs(std::ostream*) {}

  template <typename T>
  void LogArgs(std::ostream* out, const T& value_only) {
    *out << ArgValueTransform(value_only);
  }

  template <typename T>
  void LogArgs(std::ostream* out, const std::string& arg_name,
               const T& arg_value) {
    *out << "\"" << arg_name << "\" : " << ArgValueTransform(arg_value);
  }

  template <typename T, typename... Args>
  void LogArgs(std::ostream* out, const std::string& arg_name,
               const T& arg_value, const Args... args) {
    *out << "\"" << arg_name << "\" : " << ArgValueTransform(arg_value) << ", ";
    LogArgs(out, args...);
  }

  std::mutex log_mutex_;
  std::ostream* summary_out_ = &std::cerr;
  std::ostream* detail_out_ = &std::cerr;
  std::ostream* accuracy_out_ = &std::cerr;
  bool accuracy_needs_comma_ = false;
  PerfClock::time_point log_origin_;
  uint32_t log_error_count_ = 0;
  bool error_flagged_ = false;

  std::mutex trace_mutex_;
  std::ostream* trace_out_ = nullptr;
  PerfClock::time_point trace_origin_;

  const std::string* current_pid_tid_ = nullptr;
  PerfClock::time_point log_detail_time_;
  PerfClock::time_point scoped_start_;
  PerfClock::time_point scoped_end_;

  std::mutex latencies_mutex_;
  std::condition_variable all_latencies_recorded_;
  std::vector<QuerySampleLatency> latencies_;
  std::atomic<QuerySampleLatency> max_latency_{0};
  size_t latencies_recorded_ = 0;
  size_t latencies_expected_ = 0;
  // Must be called with latencies_mutex_ held.
  bool AllLatenciesRecorded() {
    return latencies_recorded_ == latencies_expected_;
  }
};

template <typename LambdaT>
class ScopedTracer {
 public:
  ScopedTracer(LambdaT&& lambda)
      : start_(PerfClock::now()), lambda_(std::forward<LambdaT>(lambda)) {}

  ~ScopedTracer() {
    Log([start = start_, lambda = std::move(lambda_),
         end = PerfClock::now()](AsyncLog& log) {
      log.SetScopedTraceTimes(start, end);
      lambda(log);
    });
  }

 private:
  PerfClock::time_point start_;
  LambdaT lambda_;
};

template <typename LambdaT>
auto MakeScopedTracer(LambdaT&& lambda) -> ScopedTracer<LambdaT> {
  return ScopedTracer<LambdaT>(std::forward<LambdaT>(lambda));
}

// Logs all threads belonging to a run.
class Logger {
 public:
  Logger(std::chrono::duration<double> poll_period, size_t max_threads_to_log);
  ~Logger();

  void StartIOThread();
  void StopIOThread();

  void StartLogging(std::ostream* summary, std::ostream* detail,
                    std::ostream* accuracy);
  void StopLogging();

  void StartNewTrace(std::ostream* trace_out, PerfClock::time_point origin);
  void StopTracing();

  void RestartLatencyRecording();
  std::vector<QuerySampleLatency> GetLatenciesBlocking(size_t expected_count);
  QuerySampleLatency GetMaxLatencySoFar();

 private:
  friend TlsLogger;
  friend TlsLoggerWrapper;

  void RegisterTlsLogger(TlsLogger* tls_logger,
                         std::function<void()> destroyer);
  void UnRegisterTlsLogger(std::unique_ptr<TlsLogger> tls_logger);
  void RequestSwapBuffers(TlsLogger* tls_logger);
  void CollectTlsLoggerStats(TlsLogger* tls_logger);

  // Slow synchronous error logging for internals that may prevent
  // async logging from working.
  template <typename... Args>
  void LogErrorSync(const std::string& message, const Args... args) {
    async_logger_.FlagError();
    async_logger_.LogDetail(message, args...);
  }

  TlsLogger* GetTlsLoggerThatRequestedSwap(size_t slot, size_t next_id);
  void GatherRetrySwapRequests(std::vector<TlsLogger*>* threads_to_swap);
  void GatherNewSwapRequests(std::vector<TlsLogger*>* threads_to_swap);

  // The main logging thread function that handles the serialization
  // and I/O to the stream or file.
  void IOThread();

  // Accessed by IOThead only.
  const std::chrono::duration<double> poll_period_;
  AsyncLog async_logger_;

  const size_t max_threads_to_log_;
  std::thread io_thread_;

  // Accessed by producers and IOThead during thread registration and
  // destruction. Protected by io_thread_mutex_.
  std::mutex io_thread_mutex_;
  std::condition_variable io_thread_cv_;
  bool keep_io_thread_alive_ = false;

  std::mutex tls_loggers_registerd_mutex_;
  // |destroyer| is only used in cases where the global Logger is destroyed
  // before the thread-local TlsLoggers. i.e.: With python modules or
  // with detached threads.
  std::unordered_map<TlsLogger*, std::function<void()>> tls_loggers_registerd_;

  // Temporarily stores TlsLogger data for threads that have exited until
  // all their log entries have been processed.
  // Accessed by IOThread and producers as their threads exit.
  std::mutex tls_logger_orphans_mutex_;
  using OrphanContainer = std::list<std::unique_ptr<TlsLogger>>;
  OrphanContainer tls_logger_orphans_;

  // Accessed by producers and IOThead atomically.
  std::atomic<size_t> swap_request_id_{0};
  std::vector<std::atomic<uintptr_t>> thread_swap_request_slots_;

  // Accessed by IOThead only.
  size_t swap_request_id_read_{0};
  struct SlotRetry {
    size_t slot;
    uintptr_t next_id;
  };
  std::vector<SlotRetry> swap_request_slots_to_retry_;
  std::vector<TlsLogger*> threads_to_swap_deferred_;
  std::vector<TlsLogger*> threads_to_read_;
  std::vector<OrphanContainer::iterator> orphans_to_destroy_;

  // Counts for retries related to the lock-free scheme.
  // Abnormally high counts could be an indicator of contention.
  // Access on IOThread only.
  size_t swap_request_slots_retry_count_ = 0;
  size_t swap_request_slots_retry_retry_count_ = 0;
  size_t swap_request_slots_retry_reencounter_count_ = 0;
  size_t start_reading_entries_retry_count_ = 0;
  size_t tls_total_log_cas_fail_count_ = 0;
  size_t tls_total_swap_buffers_slot_retry_count_ = 0;
};

Logger& GlobalLogger();
void Log(AsyncLogEntry&& entry);

template <typename LambdaT>
void LogError(LambdaT&& lambda) {
  Log([lambda = std::forward<LambdaT>(lambda),
       now = PerfClock::now()](AsyncLog& log) {
    log.FlagError();
    log.SetLogDetailTime(now);
    lambda(log);
  });
}

template <typename LambdaT>
void LogDetail(LambdaT&& lambda) {
  Log([lambda = std::forward<LambdaT>(lambda),
       now = PerfClock::now()](AsyncLog& log) {
    log.SetLogDetailTime(now);
    lambda(log);
  });
}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_LOGGING_H_
