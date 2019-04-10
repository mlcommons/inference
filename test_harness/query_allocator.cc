#include "query_allocator.h"

#include <cstdlib>

#include "system_under_test.h"

namespace mlperf {

// DefaultQueryAllocator
DefaultQueryAllocator::~DefaultQueryAllocator() = default;
void* DefaultQueryAllocator::Allocate(size_t size) { return malloc(size); }
void DefaultQueryAllocator::Free(void* ptr) { free(ptr); }

// SutQueryAllocator
SutQueryAllocator::SutQueryAllocator(SystemUnderTest* sut) : sut_(sut) {}
SutQueryAllocator::~SutQueryAllocator() = default;
void* SutQueryAllocator::Allocate(size_t size) {
  return sut_->AllocateQuerySample(size);
}
void SutQueryAllocator::Free(void* ptr) { return sut_->FreeQuerySample(ptr); }

}  // namespace mlperf
