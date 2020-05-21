from pytorch.parts.manifest import Manifest
from pytorch.parts.segment import AudioSegment

import numpy as np

import mlperf_loadgen as lg

import typing


class AudioSample(typing.NamedTuple):
    waveform: np.ndarray
    waveform_length: np.ndarray


class AudioQSL:
    def __init__(self, dataset_dir, manifest_filepath, labels,
                 sample_rate=16000, perf_count=None):
        m_paths = [manifest_filepath]
        self.manifest = Manifest(dataset_dir, m_paths, labels, len(labels),
                                 normalize=True)
        self.sample_rate = sample_rate
        self.count = len(self.manifest)
        self.perf_count = self.count if perf_count is not None else self.count
        self.sample_id_to_sample = {}
        self.qsl = lg.ConstructQSL(self.count, self.perf_count,
                                   self.load_query_samples,
                                   self.unload_query_samples)
        print("GALVEZ:length=", len(self.manifest))
        print(
            "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours.".format(
                self.manifest.duration / 3600,
                self.manifest.filtered_duration / 3600))

    def load_query_samples(self, sample_list):
        for sample_id in sample_list:
            self.sample_id_to_sample[sample_id] = self._load_sample(sample_id)

    def unload_query_samples(self, sample_list):
        for sample_id in sample_list:
            del self.sample_id_to_sample[sample_id]

    def _load_sample(self, index):
        sample = self.manifest[index]
        segment = AudioSegment.from_file(sample['audio_filepath'][0],
                                         target_sr=self.sample_rate)
        waveform = segment.samples
        assert isinstance(waveform, np.ndarray) and waveform.dtype == np.float32
        return AudioSample(waveform, np.array(waveform.shape[0], dtype=np.int64))

    def __getitem__(self, index):
        return self.sample_id_to_sample[index]

    def __del__(self):
        lg.DestroyQSL(self.qsl)
        print("Finished destroying QSL.")
