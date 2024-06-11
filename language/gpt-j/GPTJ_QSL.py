import mlperf_loadgen as lg
from dataset import Dataset

class GPTJ_QSL():
    def __init__(self, dataset_path: str, max_examples: int):
        self.dataset_path = dataset_path
        self.max_examples = max_examples

        # creating data object for QSL
        self.data_object = Dataset(
                self.dataset_path, total_count_override=self.max_examples)
        
        # construct QSL from python binding
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        print("Finished constructing QSL.")

def get_GPTJ_QSL(dataset_path: str, max_examples: int):
    return GPTJ_QSL(dataset_path , max_examples)