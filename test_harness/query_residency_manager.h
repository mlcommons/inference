#ifndef QUERY_RESIDENCY_MANAGER_H
#define QUERY_RESIDENCY_MANAGER_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

namespace mlperf {

class QuerySampleLibrary;
class QueryAllocator;

// TODO(brianderson): If performance mode will run with all queries fitting in
// memory, then we can remove this class since we don't need to manage query
// memory lifetimes.
// QueryResidencyManager coordinates the memory-related aspects of a query's
// lifecycle, including: allocation, staging, issuance, retirement, and
// deallocation.
class QueryResidencyManager {
 public:
  // If a |library_allocator| is provided, all entries will be pre-loaded
  // into system memory at startup.
  // If a |query_allocator| is provided, queries will be copied from the
  // library (pre-loaded or not) when StageQuery() is called.
  // At least one of |library_allocator| or |query_allocator| must be provided.
  QueryResidencyManager(QuerySampleLibrary* qsl,
                        QueryAllocator* library_allocator,
                        QueryAllocator* query_allocator);
  ~QueryResidencyManager();

  size_t LibrarySize() { return pre_loaded_query_library_.size(); }

  // Prepares a query for issuance.
  // Exact behavior depends on the library and query allocators provided to
  // the QueryResidencyManager constructor.
  void StageQuery(size_t query_index, intptr_t query_id);

  // Issues the staged query to the SystemUnderTest.
  void IssueQuery(intptr_t query_id);

  // Signals that the SystemUnderTest no longer needs the query's data
  // and that the query can be recycled or deallocated.
  void RetireQuery(intptr_t query_id);

 private:
  QuerySampleLibrary* qsl_;

  // Allocates and deallocates memory for pre-loaded queries.
  // If null, queries will be loaded on the fly.
  QueryAllocator* library_allocator_;

  // Allocates and deallocates memory for issued queries.
  // If null, queries will be issued directly from the library.
  QueryAllocator* query_allocator_;

  // The library of pre-loaded queries.
  struct QueryLibraryEntry {
    size_t size;
    void* data;
  };
  std::vector<QueryLibraryEntry> pre_loaded_query_library_;

  // Queries staged, but not yet issued.
  struct QueryQueueEntry {
    QueryQueueEntry();

    std::mutex mutex;
    std::condition_variable cv;
    bool ready_to_be_issued = false;
    uint64_t query_index;
    size_t size;
    void* data;
  };
  std::queue<QueryQueueEntry> query_stage_to_issue_queue_;

  // A counter to track when the staging logic doesn't complete in time
  // for the scheduled issuance.
  uint64_t stage_not_complete_at_issue_time_count_ = 0;

  // Make sure not to deallocate or recycle query memory until we are sure
  // the device isn't accessing it.
  // Only used if there's an active |query_allocator_|.
  std::unordered_map<intptr_t, QueryQueueEntry> queries_with_sut_read_lock_;
};

}  // namespace mlperf

#endif  // QUERY_RESIDENCY_MANAGER_H
