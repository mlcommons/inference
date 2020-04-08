"""
dlrm related classes and methods
the code is adapted from PostProcessCommon
"""

import torch
# import sys

#
# Post processing
#
class DlrmPostProcess:
    def __init__(self):
        self.good = 0
        self.total = 0

    def __call__(self, results, expected=None, result_dict=None):
        processed_results = []
        n = len(results)
        for idx in range(0, n):
            result = results[idx]
            processed_results.append([result])
            # debug prints
            # print(result.__class__)
            # print(result.type())
            # print(result)
            # print(expected[idx].__class__)
            # print(expected[idx].type())
            # print(expected[idx])
            # sys.exit(0)

            if result.round() == expected[idx]:
                self.good += 1
        self.total += n
        return processed_results

    def add_results(self, results):
        pass

    def start(self):
        self.good = 0
        self.total = 0

    def finalize(self, results, ds=False,  output_dir=None):
        results["good"] = self.good
        results["total"] = self.total
