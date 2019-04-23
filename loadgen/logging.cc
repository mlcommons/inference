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
#include <iostream>
#include <future>

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

template <typename T>
void RemoveNulls(T& container) {
  container.erase(
      std::remove_if(container.begin(), container.end(),
                     [](typename T::value_type v) {
                       return v == nullptr;
                     }),
      container.end());
}

}  // namespace

// TlsLogger logs a single thread using thread-local storage.
// Submits logs to the central Logger:
//   * With forward-progress guarantees. (i.e.: no locking or blocking
//       operations even if other threads have stalled.
//   * Without expensive syscalls or I/O operations.
class TlsLogger {
 public:
  TlsLogger(Logger *logger);
  ~TlsLogger();

  void Log(AsyncLogEntry &&entry);
  void SwapBuffers();

  std::vector<AsyncLogEntry>* StartReadingEntries();
  void FinishReadingEntries();
  bool ReadBufferHasBeenConsumed();

 private:
  using EntryVector = std::vector<AsyncLogEntry>;
  enum class EntryState { Unlocked, ReadLock, WriteLock };

  // Accessed by producer only.
  Logger* logger_;
  size_t i_read_ = 0;

  // Accessed by producer and consumer atomically.
  EntryVector entries_[2];
  std::atomic<EntryState> entry_states_[2] {
      {EntryState::ReadLock}, {EntryState::Unlocked} };
  std::atomic<size_t> i_write_ { 1 };

  // Accessed by consumer only.
  size_t unread_swaps_ = 0;
  size_t i_write_prev_ = 0;
};

Logger::Logger(std::ostream *out_stream,
               std::chrono::duration<double> poll_period,
               size_t max_threads_to_log)
    : out_stream_(*out_stream),
      max_threads_to_log_(max_threads_to_log),
      origin_time_(PerfClock::now()),
      poll_period_(poll_period),
      async_logger_(out_stream),
      thread_swap_request_slots_(max_threads_to_log * 2) {
  const size_t kSlotCount = max_threads_to_log * 2;
  for (size_t i = 0; i < kSlotCount; i++) {
    std::atomic_init(&thread_swap_request_slots_[i],
                     SwapRequestSlotIsWritableValue(i));
  }
  io_thread_ = std::thread(&Logger::IOThread, this);
  out_stream_ << "{ \"traceEvents\": [\n";
}

Logger::~Logger() {
  {
    std::unique_lock<std::mutex> lock(io_thread_mutex_);
    keep_io_thread_alive_ = false;
    io_thread_cv_.notify_all();
  }
  io_thread_.join();

  out_stream_ << "{ \"name\": \"LastTrace\" }\n";
  out_stream_ << "],\n";
  out_stream_ << "\"displayTimeUnit\": \"ns\",\n";
  out_stream_ << "\"otherData\": {\n";
  out_stream_ << "\"version\": \"MLPerf LoadGen v0.5a0\"\n";
  out_stream_ << "}\n";
  out_stream_ << "}\n";

  // TODO: Fix lifetime management of TlsLoggers and Loggers.
  {
    std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
    if (!tls_loggers_registerd_.empty()) {
      std::cerr << "Warning: Destroying Logger with TlsLoggers alive: (x" <<
                   tls_loggers_registerd_.size() << ").\n";
    }
  }

  if (swap_request_id_read_ != swap_request_id_.load() ||
      !swap_request_slots_to_retry_.empty() ||
      !threads_to_swap_deferred_.empty() ||
      !threads_to_read_.empty()) {
    std::cerr << "Warning: Logger destroyed before all logs were processed.\n";
    std::cerr << "swap_request_id_read_: " << swap_request_id_read_ << "\n";
    std::cerr << "swap_request_id_.load(): " <<
                 swap_request_id_.load() << "\n";
    std::cerr << "swap_request_slots_to_retry_.empty(): " <<
                 swap_request_slots_to_retry_.empty() << "\n";
    std::cerr << "threads_to_swap_deferred_.empty(): " <<
                 threads_to_swap_deferred_.empty() << "\n";
    std::cerr << "threads_to_read_.empty(): " <<
                 threads_to_read_.empty() << "\n";
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
  do {
    id = swap_request_id_.fetch_add(1);
    slot = id % thread_swap_request_slots_.size();
    slot_is_writeable_value = SwapRequestSlotIsWritableValue(id);
  } while (!thread_swap_request_slots_[slot].compare_exchange_strong(
               slot_is_writeable_value, tls_logger_as_uint));
}

void Logger::RegisterTlsLogger(TlsLogger* tls_logger) {
  std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
  if (tls_loggers_registerd_.size() >= max_threads_to_log_) {
    std::cerr << "Warning: More TLS loggers registerd than can"
                 "be active simultaneously.\n";
  }
  tls_loggers_registerd_.insert(tls_logger);
}

void Logger::UnRegisterTlsLogger(TlsLogger* tls_logger) {
  // TODO(brianderson): Move the TlsLogger data to a struct that we can move
  // ownership of to Logger for later consumption. Then exit this thread
  // immediately, rather than synchronizing with the Logger's consumption here.
  std::promise<void> io_thread_done_with_tls_logger;
  // The AsyncLog lambda runs after the last log submitted by the tls_logger.
  tls_logger->Log([&](AsyncLog&) {
          // Use thread_cleanup_tasks_ to signal completion only after
          // IOThread calls FinishReadingEntries to avoid use-after-free.
          thread_cleanup_tasks_.push_back([&]{
              io_thread_done_with_tls_logger.set_value();
          });
      }
  );

  io_thread_done_with_tls_logger.get_future().wait();
  std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
  tls_loggers_registerd_.erase(tls_logger);
}

std::vector<std::chrono::nanoseconds> Logger::GetLatencies() {
  return async_logger_.GetLatencies();
}

TlsLogger* Logger::GetTlsLoggerThatRequestedSwap(size_t slot, size_t next_id) {
  uintptr_t slot_value = thread_swap_request_slots_[slot].load();
  if (SwapRequestSlotIsReadable(slot_value)) {
    bool success = thread_swap_request_slots_[slot].compare_exchange_strong(
        slot_value, SwapRequestSlotIsWritableValue(next_id));
    assert(success);
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
  for (auto& slot_retry: retry_slots) {
    TlsLogger* tls_logger =
        GetTlsLoggerThatRequestedSwap(slot_retry.slot,
                                      slot_retry.next_id);
    if (tls_logger) {
      threads_to_swap->push_back(tls_logger);
    } else {
      swap_request_slots_to_retry_.push_back(slot_retry);
    }
  }
}

void Logger::GatherNewSwapRequests(std::vector<TlsLogger*>* threads_to_swap) {
  auto swap_request_end = swap_request_id_.load();
  while (swap_request_id_read_ < swap_request_end) {
    size_t slot = swap_request_id_read_ % thread_swap_request_slots_.size();
    size_t next_id = swap_request_id_read_ + thread_swap_request_slots_.size();
    TlsLogger* tls_logger = GetTlsLoggerThatRequestedSwap(slot, next_id);
    if (tls_logger) {
      threads_to_swap->push_back(tls_logger);
    } else {
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
      }
    }
    swap_request_id_read_++;
  }
}

void Logger::IOThread() {
  while(keep_io_thread_alive_) {
    {
      std::unique_lock<std::mutex> lock(io_thread_mutex_);
      io_thread_cv_.wait_for(
            lock, poll_period_, [&] { return !keep_io_thread_alive_; });
    }

    std::vector<TlsLogger*> threads_to_swap;
    threads_to_swap.swap(threads_to_swap_deferred_);
    GatherRetrySwapRequests(&threads_to_swap);
    GatherNewSwapRequests(&threads_to_swap);
    for (TlsLogger* thread : threads_to_swap) {
      if (thread->ReadBufferHasBeenConsumed()) {
        // Don't swap buffers again until wev'e finish reading the
        // previous swap.
        threads_to_swap_deferred_.push_back(thread);
      } else {
        // After swapping a thread, it's ready to be read.
        thread->SwapBuffers();
        threads_to_read_.push_back(thread);
      }
    }

    // Read from the threads we are confident have activity.
    for (std::vector<TlsLogger*>::iterator thread = threads_to_read_.begin();
         thread != threads_to_read_.end();
         thread++) {
      std::vector<AsyncLogEntry>* entries = (*thread)->StartReadingEntries();
      if (entries) {
        for (auto& entry : *entries) {
          // Execute the entry to perform the serialization and I/O.
          entry(async_logger_);
        }
        (*thread)->FinishReadingEntries();
        *thread = nullptr;

        // Clean up tasks
        for (auto& task : thread_cleanup_tasks_) {
          task();
        }
        thread_cleanup_tasks_.clear();
      }
    }

    // Only remove threads where reading succeeded so we retry the failed
    // threads the next time around.
    RemoveNulls(threads_to_read_);
  }
}

TlsLogger::TlsLogger(Logger *logger) : logger_(logger) {
  logger_->RegisterTlsLogger(this);
}

TlsLogger::~TlsLogger() {
  logger_->UnRegisterTlsLogger(this);
}

// Log always makes forward progress since it can unconditionally obtain a
// "lock" on at least one of the buffers for writting.
// Notificiation is also lock free.
void TlsLogger::Log(AsyncLogEntry &&entry) {
  size_t i_write = i_write_.load();
  auto unlocked = EntryState::Unlocked;
  if (!entry_states_[i_write].compare_exchange_strong(
        unlocked, EntryState::WriteLock)) {
    i_write ^= 1;
    // Obtaining a write lock on the second buffer will always succeed since
    // the Logger will not attempt a read lock on it (if it has a write lock on
    // the first buffer) until after the call to RequestSwapBuffers below.
    bool success = entry_states_[i_write].compare_exchange_strong(
             unlocked, EntryState::WriteLock);
    assert(success);
  }
  entries_[i_write].emplace_back(std::forward<AsyncLogEntry>(entry));

  auto write_lock = EntryState::WriteLock;
  bool success = entry_states_[i_write].compare_exchange_strong(
           write_lock, EntryState::Unlocked);
  assert(success);

  bool write_buffer_swapped = i_write_prev_ != i_write;
  if (write_buffer_swapped) {
    logger_->RequestSwapBuffers(this);
    i_write_prev_ = i_write;
  }
}

void TlsLogger::SwapBuffers() {
  auto read_lock = EntryState::ReadLock;
  bool success = entry_states_[i_read_].compare_exchange_strong(
           read_lock, EntryState::Unlocked);
  assert(success);

  i_write_.fetch_xor(1);
  i_read_ ^= 1;
  unread_swaps_++;
}

// Returns nullptr if read lock fails.
std::vector<AsyncLogEntry>* TlsLogger::StartReadingEntries() {
  auto unlocked = EntryState::Unlocked;
  if (!entry_states_[i_read_].compare_exchange_strong(
        unlocked, EntryState::ReadLock)) {
    return nullptr;
  }
  return &entries_[i_read_];
}

void TlsLogger::FinishReadingEntries() {
  entries_[i_read_].clear();
  unread_swaps_--;
}

bool TlsLogger::ReadBufferHasBeenConsumed() {
  return unread_swaps_ != 0;
}

void Log(Logger *logger, AsyncLogEntry &&entry) {
  thread_local TlsLogger tls_logger(logger);
  tls_logger.Log(std::forward<AsyncLogEntry>(entry));
}

}  // namespace mlperf
