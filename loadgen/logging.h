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
#include <string>
#include <thread>
#include <unordered_set>
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
  AsyncLog();
  ~AsyncLog();

  void SetLogFiles(std::ostream* summary, std::ostream* detail,
                   std::ostream* accuracy, bool copy_detail_to_stdout,
                   bool copy_summary_to_stdout,
                   PerfClock::time_point log_origin);
  void StartNewTrace(std::ostream* trace_out, PerfClock::time_point origin);
  void Flush();

  void SetCurrentTracePidTidString(const std::string* pid_tid) {
    current_pid_tid_ = pid_tid;
  }

  void LogAccuracy(uint64_t seq_id, const QuerySampleIndex qsl_idx,
                   const LogBinaryAsHexString& response);

  template <typename... Args>
  void LogSummary(const std::string& message, const Args... args);

  void SetLogDetailTime(PerfClock::time_point time) { log_detail_time_ = time; }

  // TODO: Warnings.
  void FlagError() {
    std::unique_lock<std::mutex> lock(log_mutex_);
    log_error_count_++;
    error_flagged_ = true;
  }

  template <typename... Args>
  void LogDetail(const std::string& message, const Args... args);

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

  template <typename... Args>
  void TraceCounterEvent(const std::string& trace_name,
                         PerfClock::time_point time, const Args... args) {
    std::unique_lock<std::mutex> lock(trace_mutex_);
    if (!trace_out_) {
      return;
    }
    *trace_out_ << "{\"name\": \"" << trace_name << "\", "
                << "\"ph\": \"C\", " << *current_pid_tid_
                << "\"ts\": " << (time - trace_origin_).count() << ", "
                << "\"args\": { ";
    LogArgs(trace_out_, args...);
    *trace_out_ << " }},\n";
  }

  void RestartLatencyRecording(uint64_t first_sample_sequence_id);
  void RecordLatency(uint64_t sample_sequence_id, QuerySampleLatency latency);
  std::vector<QuerySampleLatency> GetLatenciesBlocking(size_t expected_count);
  QuerySampleLatency GetMaxLatencySoFar();

 private:
  void WriteAccuracyHeaderLocked();
  void WriteAccuracyFooterLocked();
  void WriteTraceEventHeaderLocked();
  void WriteTraceEventFooterLocked();

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
  // TODO: Instead of these bools, use a class that forwards to two streams.
  bool copy_detail_to_stdout_ = false;
  bool copy_summary_to_stdout_ = false;
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
  uint64_t latencies_first_sample_sequence_id_ = 0;
  std::vector<QuerySampleLatency> latencies_;
  std::atomic<QuerySampleLatency> max_latency_{0};
  size_t latencies_recorded_ = 0;
  size_t latencies_expected_ = 0;
  // Must be called with latencies_mutex_ held.
  bool AllLatenciesRecorded() {
    return latencies_recorded_ == latencies_expected_;
  }
};

// Logs all threads belonging to a run.
class Logger {
 public:
  Logger(std::chrono::duration<double> poll_period, size_t max_threads_to_log);
  ~Logger();

  void StartIOThread();
  void StopIOThread();

  void StartLogging(std::ostream* summary, std::ostream* detail,
                    std::ostream* accuracy, bool copy_detail_to_stdout,
                    bool copy_summary_to_stdout);
  void StopLogging();

  void StartNewTrace(std::ostream* trace_out, PerfClock::time_point origin);
  void StopTracing();

  void LogContentionCounters();

  void RestartLatencyRecording(uint64_t first_sample_sequence_id);
  std::vector<QuerySampleLatency> GetLatenciesBlocking(size_t expected_count);
  QuerySampleLatency GetMaxLatencySoFar();

 private:
  friend AsyncLog;
  friend TlsLogger;
  friend TlsLoggerWrapper;

  void RegisterTlsLogger(TlsLogger* tls_logger);
  void UnRegisterTlsLogger(std::unique_ptr<TlsLogger> tls_logger);
  void RequestSwapBuffers(TlsLogger* tls_logger);
  void CollectTlsLoggerStats(TlsLogger* tls_logger);

  TlsLogger* GetTlsLoggerThatRequestedSwap(size_t slot, size_t next_id);
  void GatherRetrySwapRequests(std::vector<TlsLogger*>* threads_to_swap);
  void GatherNewSwapRequests(std::vector<TlsLogger*>* threads_to_swap);

  // The main logging thread function that handles the serialization
  // and I/O to the stream or file.
  void IOThread();

  // Slow synchronous error logging for internals that may prevent
  // async logging from working.
  template <typename... Args>
  void LogErrorSync(const std::string& message, Args&&... args) {
    // TODO: Acquire mutex once for FlagError + LogDetail to avoid
    //       races. Better yet, switch to a non-stateful error API.
    //       This is better than nothing though.
    async_logger_.FlagError();
    async_logger_.LogDetail(message, std::forward<Args>(args)...);
  }

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
  std::unordered_set<TlsLogger*> tls_loggers_registerd_;

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

class AsyncSummary {
 public:
  explicit AsyncSummary(AsyncLog& async_log) : async_log_(async_log) {}
  AsyncLog& async_log() { return async_log_; }

  template <typename... Args>
  AsyncLog& operator()(Args&&... args) {
    async_log_.LogSummary(std::forward<Args>(args)...);
    return async_log_;
  }

 private:
  AsyncLog& async_log_;
};

template <typename LambdaT>
void LogSummary(LambdaT&& lambda) {
  Log([lambda = std::forward<LambdaT>(lambda)](AsyncLog& log) mutable {
    AsyncSummary async_summary(log);
    lambda(async_summary);
  });
}

class AsyncDetail {
 public:
  explicit AsyncDetail(AsyncLog& async_log) : async_log_(async_log) {}
  AsyncLog& async_log() { return async_log_; }

  template <typename... Args>
  AsyncLog& operator()(Args&&... args) {
    async_log_.LogDetail(std::forward<Args>(args)...);
    return async_log_;
  }

  template <typename... Args>
  AsyncLog& Error(Args&&... args) {
    async_log_.FlagError();
    async_log_.LogDetail(std::forward<Args>(args)...);
    return async_log_;
  }

 private:
  AsyncLog& async_log_;
};

template <typename LambdaT>
void LogDetail(LambdaT&& lambda) {
  Log([lambda = std::forward<LambdaT>(lambda),
       timestamp = PerfClock::now()](AsyncLog& log) mutable {
    log.SetLogDetailTime(timestamp);
    AsyncDetail async_detail(log);
    lambda(async_detail);
  });
}

class AsyncTrace {
 public:
  explicit AsyncTrace(AsyncLog& async_log) : async_log_(async_log) {}
  AsyncLog& async_log() { return async_log_; }

  template <typename... Args>
  AsyncLog& operator()(Args&&... args) {
    async_log_.ScopedTrace(std::forward<Args>(args)...);
    return async_log_;
  }

 private:
  AsyncLog& async_log_;
};

// ScopedTracer is an RAII object that traces the start and end of its lifetime.
template <typename LambdaT>
class ScopedTracer {
 public:
  ScopedTracer(LambdaT&& lambda)
      : start_(PerfClock::now()), lambda_(std::forward<LambdaT>(lambda)) {}

  ~ScopedTracer() {
    Log([start = start_, lambda = std::move(lambda_),
         end = PerfClock::now()](AsyncLog& log) {
      log.SetScopedTraceTimes(start, end);
      AsyncTrace async_trace(log);
      lambda(async_trace);
    });
  }

 private:
  PerfClock::time_point start_;
  LambdaT lambda_;
};

// MakeScopedTracer helps with automatic template type deduction, which
// has been supported for functions for a long time.
// C++17 will support deduction for classes, which will neutralize the utility
// of a helper function like this.
template <typename LambdaT>
auto MakeScopedTracer(LambdaT&& lambda) -> ScopedTracer<LambdaT> {
  return ScopedTracer<LambdaT>(std::forward<LambdaT>(lambda));
}

template <typename... Args>
void AsyncLog::LogSummary(const std::string& message, const Args... args) {
  auto tracer = MakeScopedTracer([message](AsyncTrace& trace) {
    std::string sanitized_message = message;
    std::replace(sanitized_message.begin(), sanitized_message.end(), '"', '\'');
    std::replace(sanitized_message.begin(), sanitized_message.end(), '\n', ';');
    trace("LogSummary", "message", "\"" + sanitized_message + "\"");
  });
  std::unique_lock<std::mutex> lock(log_mutex_);
  *summary_out_ << message;
  LogArgs(summary_out_, args...);
  *summary_out_ << "\n";

  if (copy_summary_to_stdout_) {
    std::cout << message;
    LogArgs(&std::cout, args...);
    std::cout << "\n";
  }
}

template <typename... Args>
void AsyncLog::LogDetail(const std::string& message, const Args... args) {
  auto tracer = MakeScopedTracer([message](AsyncTrace& trace) {
    std::string sanitized_message = message;
    std::replace(sanitized_message.begin(), sanitized_message.end(), '"', '\'');
    std::replace(sanitized_message.begin(), sanitized_message.end(), '\n', ';');
    trace("LogDetail", "message", "\"" + sanitized_message + "\"");
  });
  std::unique_lock<std::mutex> lock(log_mutex_);
  std::vector<std::ostream*> detail_streams{detail_out_, &std::cout};
  if (!copy_detail_to_stdout_) {
    detail_streams.pop_back();
  }
  for (auto os : detail_streams) {
    *os << *current_pid_tid_
        << "\"ts\": " << (log_detail_time_ - log_origin_).count() << "ns : ";
    if (error_flagged_) {
      *os << "ERROR : ";
    }
    *os << message;
    LogArgs(os, args...);
    *os << "\n";
    if (error_flagged_) {
      os->flush();
    }
  }
  error_flagged_ = false;
}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_LOGGING_H_
