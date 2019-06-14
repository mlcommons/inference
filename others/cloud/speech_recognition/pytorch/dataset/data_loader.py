import os
from tempfile import NamedTemporaryFile
from math import floor, ceil

import sox
import librosa
import numpy as np
import scipy.signal
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


def load_audio(path, frame_start=0, frame_end=-1):
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    if frame_end > 0 or frame_start > 0:
        assert frame_start < frame_end, "slicing does not yet support inverting audio"
        if frame_end > sound.shape[0]:
            repeats = ceil((frame_end - sound.shape[0])/float(sound.shape[0]))
            appendage = sound
            for _ in range(int(repeats)):
                sound = np.concatenate((sound,appendage))
        sound = sound[frame_start:frame_end]
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        self.paths = path is not None and librosa.util.find_files(path)
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    @staticmethod
    def inject_noise_sample(data, noise_path, noise_level):
        noise_src = load_audio(noise_path)
        noise_offset_fraction = np.random.rand()
        noise_dst = np.zeros_like(data)
        src_offset = int(len(noise_src) * noise_offset_fraction)
        src_left = len(noise_src) - src_offset
        dst_offset = 0
        dst_left = len(data)
        while dst_left > 0:
            copy_size = min(dst_left, src_left)
            np.copyto(noise_dst[dst_offset:dst_offset + copy_size],
                      noise_src[src_offset:src_offset + copy_size])
            if src_left > dst_left:
                dst_left = 0
            else:
                dst_left -= copy_size
                dst_offset += copy_size
                src_left = len(noise_src)
                src_offset = 0
        data += noise_level * noise_dst
        return data


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'],
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path, frame_start=0, frame_end=-1):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate, frame_start=frame_start, frame_end=frame_end)
        else:
            y = load_audio(audio_path, frame_start=frame_start, frame_end=frame_end)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        d = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(d)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False, force_duration=-1, slice=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        :param force_duration (default -1): force the duration of input audio in seconds
        :param slice (default False): set True if you want to perform slicing before batching
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.access_history = []
	self.force_duration = force_duration
	self.force_duration_frame = int(force_duration*audio_conf['sample_rate']) if force_duration > 0 else -1
	self.slice = slice
        
        # If user chose to slice the input workload, we will update the
        #        self.ids, and self.size
        # class members and create the
        #        self.slice_meta
        # member.
        if slice and force_duration > 0:
            self.slice(force_duration, audio_conf['sample_rate'])
            
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        self.access_history.append(index)
        audio_path, transcript_path = sample[0], sample[1]
        if hasattr(self, 'slice_meta'):
            spect = self.parse_audio(audio_path, self.slice_meta[0], self.slice_meta[1])
        else:
            spect = self.parse_audio(audio_path, 0, self.force_duration_frame)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def get_meta(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        if hasattr(self, 'slice_meta'):
            audio_dur = self.slice_meta[2]
            audio_size = self.slice_meta[3]
        else:
            audio_dur = sox.file_info.duration(audio_path)
            audio_size = os.path.getsize(audio_path)/float(1000)        # returns bytes, so convert to kb
        return audio_path, transcript_path, audio_dur, audio_size
    
    def slice(self, force_duration, sample_rate):
        """
        Look through the entire manifest file and by using the audio duration information and the
        force_duration parameter, we slice using the following strategy:
            let audio duration = L
            let force_duration = s
            L%s = 0:
                perfect slice, choose L/s slices each s duration long
            else:
                L(1-1/floor(L/s)) > L(1/ceil(L/s)-1)            Want the smaller result the the two sides of the inequality
                2 - 1/floor(L/s)   - 1/ceil(L/s) > 0            let floor(L/s) = N
                2 - (N+1)/(N(N+1)) - N/(N(N+1)) > 0
                Thus if
                    (2N+1)/(N(N+1)) < 2, then we should choose ceil(L/s) slices each L/ceil(L/s) duration long
                    (2N+1)/(N(N+1)) >= 2, then we should choose floor(L/s) slices each L/floor(L/s) duration long
        After computing these slice duratations we build a new self.ids (sorted by the slice durations) and give the
        necessary start and end frames for the audio parser to consider.
        Consequentially we will also update the self.size field.
        """ 
        new_input_set = []
        for index in range(self.size):
            s = float(force_duration) 
            audio_path, transcript_path, L, kb = self.get_meta(index)
            if L < s: 
                N = 1
                s = L
            if L % s == 0:
                N = int(L/s)
            else:
                N = floor(L/s)
                if (2*N+1)/(N*(N+1)) < 2:
                    N = ceil(L/s)
                s = L/N
            s_frames = int(s*sample_rate)
            frames_array = range(0,int(N*s_frames),s_frames)
            for i in range(int(N)):
		start = frames_array[i]
		if i+1 >= N:
			end = -1
		else:
	                end = frames_array[i+1]
                new_input_set.append([audio_path, transcript_path, start, end, s, kb/N])
        new_input_set.sort(key=(lambda x: x[4]))
        self.ids = [(each[1],each[2]) for each in new_input_set]
        self.slice_meta = [each[2:] for each in new_input_set]
        print("Pre-slice loader size: {}".format(self.size))
        self.size = len(self.ids)
        print("Post-slice loader size: {}".format(self.size))
        print("Sliced by: {}".format(force_duration))
        del new_input_set

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size


class SpectrogramAndPathDataset(SpectrogramDataset):
    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript, audio_path


class SpectrogramAndLogitsDataset(SpectrogramDataset):
    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        logit_path = os.path.join(
            os.path.split(os.path.split(audio_path)[0])[0],
            "logits",
            os.path.splitext(os.path.split(audio_path)[1])[0] + ".pth"
        )
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        logits = torch.load(logit_path)
        return spect, transcript, audio_path, logits


def _collate_fn_logits(batch):
    longest_sample = max(batch, key=lambda x: x[0].size(1))[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)

    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []

    paths = []

    longest_logit = max(batch, key=lambda p: p[3].size(0))[3]
    logit_len = longest_logit.size(0)
    nclasses = batch[0][3].size(-1)
    logits = torch.FloatTensor(minibatch_size, logit_len, nclasses)
    logits.fill_(float('-inf'))

    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        paths.append(sample[2])
        logit = sample[3]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
        logits[x, :logit.size(0)].copy_(logit)

    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes, paths, logits


def _collate_fn_paths(batch):
    longest_sample = max(batch, key=lambda x: x[0].size(1))[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    paths = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        paths.append(sample[2])
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes, paths, None


def _collate_fn(batch):
    longest_sample = max(batch, key=lambda x: x[0].size(1))[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        if 'with_meta' in kwargs and kwargs['with_meta']:
            self.with_meta = True
        else:
            self.with_meta = False
        kwargs.pop('with_meta', None)
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.item_meta = []
        self.batch_meta = []
        self.iter = 0
        self.collate_fn = self.my_collate_fn
        self.dataset = args[0]

    def my_collate_fn(self, batch):
        # We want to make this collate function such that if the audio lengths are very different,
        # we will not batch things together and instead perform a sub optimal batch
        # this a bit involved and odd.. so we should reconsider
        # and it will prepare meta information for users to pull if they wish
        longest_sample = max(batch, key=lambda x: x[0].size(1))[0]
        freq_size = longest_sample.size(0)
        minibatch_size = len(batch)
        max_seqlength = longest_sample.size(1)
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        input_percentages = torch.FloatTensor(minibatch_size)
        target_sizes = torch.IntTensor(minibatch_size)
        targets = []
        self.item_meta = []
        self.batch_meta = []
        self.iter += 1
        for x in range(minibatch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(1)
            inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
            input_percentages[x] = seq_length / float(max_seqlength)
            target_sizes[x] = len(target)
            targets.extend(target)
            self.item_meta.append(list(self.dataset.get_meta(self.dataset.access_history[x])))
            self.item_meta[-1].append(seq_length)
            if len(self.batch_meta) == 0:
                self.batch_meta = self.item_meta[-1][:]
            else:
                for i, meta in enumerate(self.item_meta[-1]):
                    if i in [2, 3, 4]:
                        self.batch_meta[i] += meta
        targets = torch.IntTensor(targets)
        self.dataset.access_history = []
        if self.with_meta:
            return inputs, targets, input_percentages, target_sizes, self.batch_meta, self.item_meta
        else:
            return inputs, targets, input_percentages, target_sizes


class AudioDataAndLogitsLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataAndLogitsLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_logits


class AudioDataAndPathsLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataAndPathsLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_paths


def augment_audio_with_sox(path, sample_rate, tempo, gain, frame_start=0, frame_end=-1):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                augmented_filename,
                                                                                " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename, frame_start=frame_start, frame_end=frame_end)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8), frame_start=0, frame_end=-1):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value,
                                   frame_start=frame_start, frame_end=frame_end)
    return audio
