"""
dlrm related classes and methods
"""

import torch

#
# Post processing
#
class DlrmPostProcess:
    def __init__(self):
        self.good = 0
        self.total = 0

    def __call__(self, results, expected=None, result_dict=None):
        processed_results = []
        n = len(results[0])
        for idx in range(0, n):
            result = results[0][idx]
            processed_results.append([result])
            if torch.tensor(result, dtype=torch.float).round() == expected[idx]:
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

