#ifndef QUERY_ALLOCATOR_H
#define QUERY_ALLOCATOR_H

#include <stddef.h>

namespace mlperf {

class SystemUnderTest;

// An internal interface used to allocate query and response memory.
class QueryAllocator {
 public:
  virtual ~QueryAllocator() {}
  virtual void* Allocate(size_t size) = 0;
  virtual void Free(void* ptr) = 0;
};

// Allocates memory using malloc and free.
class DefaultQueryAllocator : public QueryAllocator {
 public:
  ~DefaultQueryAllocator() override;
  void* Allocate(size_t size) override;
  void Free(void* ptr) override;
};

// Allocates memory using the SUT's AllocateQuerySample/FreeQuerySample.
class SutQueryAllocator : public QueryAllocator {
 public:
  SutQueryAllocator(SystemUnderTest* su);
  ~SutQueryAllocator() override;
  void* Allocate(size_t size) override;
  void Free(void* ptr) override;
 private:
  SystemUnderTest* sut_;
};

}  // namespace mlperf

#endif  // QUERY_ALLOCATOR_H
