#ifndef QUERY_SAMPLE_LIBRARY_REGISTRY_H_
#define QUERY_SAMPLE_LIBRARY_REGISTRY_H_

#include <memory>
#include <vector>

namespace mlperf {

class QuerySampleLibrary;
using QslVector = std::vector<QuerySampleLibrary*>;

// QslRegistry lets a query sample library register itself during dynamic
// initialization of static variables (before main starts), so the test
// harness can select it at runtime.
// Usage:
// QuerySampleLibraryDerived::QuerySampleLibraryDerived() {
//   QslRegistry::Register(this);
// }
// QuerySampleLibraryDerived library;
class QslRegistry {
 public:
  static void Register(QuerySampleLibrary* registrant) {
    InitializeIfNeeded();
    libraries_->push_back(registrant);
  }

  static QslVector GetRegistry() {
    InitializeIfNeeded();
    return *libraries_;
  }

  static QuerySampleLibrary* GetQslInstance(std::string name) {
    InitializeIfNeeded();
    for (auto* qsl : *libraries_) {
      if (qsl->Name() == name) {
        return qsl;
      }
    }
    return nullptr;
  }

 private:
  // Use a unique_ptr so libraries_ is part of constant initialization, which
  // occurs before dynamic initialization.
  // This ensures libraries_ is null before the first call to Register().
  static std::unique_ptr<QslVector> libraries_;
  static void InitializeIfNeeded() {
    if (libraries_)
      return;
    libraries_ = std::unique_ptr<QslVector>(new QslVector);
  }
};

}  // namespace mlperf

#endif  // QUERY_SAMPLE_LIBRARY_REGISTRY_H_
