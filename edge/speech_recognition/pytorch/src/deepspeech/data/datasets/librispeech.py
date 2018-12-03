import fnmatch
import os
import shutil
import tarfile

import requests
import soundfile
import torch

from deepspeech.data.datasets import utils


class LibriSpeech(torch.utils.data.Dataset):
    """LibriSpeech Dataset - http://openslr.org/12/

    Args:
        root (string): Root directory of the dataset. This should contain the
            LibriSpeech directory which itself contains one directory per
            subset in subsets.  These will be created if they do not exist, or
            are corrupt, and download is True.
        subsets (list of strings): List of subsets to create the dataset from.
            Subset names must be in: `['train-clean-100', 'train-clean-360',
            'train-other-500', 'dev-clean', 'dev-other', 'test-clean',
            'test-other']`.
        transform (callable, optional): A function that returns a transformed
            piece of audiodata.
        target_transform (callable, optional): A function that returns a
            transformed target.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again. See the `download` method
            for more information.
    """

    base_dir = 'LibriSpeech'
    openslr_url = 'http://www.openslr.org/resources/12/'
    data_files = {
        'train-clean-100': {'archive_md5': '2a93770f6d5c6c964bc36631d331a522',
                            'dir_md5':     'b1b762a7384c17c06eee975933c71739'},
        'train-clean-360': {'archive_md5': 'c0e676e450a7ff2f54aeade5171606fa',
                            'dir_md5':     'ef1c8b93522d89ae27116d64a12d1d2f'},
        'train-other-500': {'archive_md5': 'd1a0fd59409feb2c614ce4d30c387708',
                            'dir_md5':     '851de4dff9bdfd1a89d9eab18bc675c6'},
        'dev-clean':       {'archive_md5': '42e2234ba48799c1f50f24a7926300a1',
                            'dir_md5':     '9e3b56b96e2cbbcc941c00f52f2fdcf9'},
        'dev-other':       {'archive_md5': 'c8d0bcc9cca99d4f8b62fcc847357931',
                            'dir_md5':     '1417e91c9f0a1c2c1c61af1178ffa94b'},
        'test-clean':      {'archive_md5': '32fa31d27d2e1cad72775fee3f4849a9',
                            'dir_md5':     '5fad2e72ec7af2659d50e4df720bc22b'},
        'test-other':      {'archive_md5': 'fb5a50374b501bb3bac4815ee91d3135',
                            'dir_md5':     'ddcbdd339bd02c8d2b6c1bbde28c828c'}
    }

    def __init__(self, root, subsets, transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.subsets = self._validate_subsets(subsets)
        self._transform = transform
        self._target_transform = target_transform

        if download:
            self.download()

        if not self.check_integrity():
            raise ValueError('Dataset subset(s) not found or corrupt.'
                             ' Set `download=True` to download.')

        self.load_data()

    def __getitem__(self, index):
        """Returns the sample at index in the dataset.

        The samples are in ascending order by audio sample length.

        Transforms are applied if not None.

        Args:
            index (int): Index.

        Returns:
            tuple: (audiodata, target) where audiodata is a np.int16 numpy
            array and target is a string containing the target transcription.
        """
        path = self.paths[index]
        audio, rate = soundfile.read(path, dtype='int16')

        assert rate == 16000, '%r sample rate != 16000' % path
        assert len(audio) / rate == self.durations[index], \
            '%r sample duration != to expected duration'

        if self._transform is not None:
            audio = self._transform(audio)

        target = self.transcriptions[index]
        if self._target_transform is not None:
            target = self._target_transform(target)

        return audio, target

    def __len__(self):
        return len(self.paths)

    def _validate_subsets(self, subsets):
        """Ensures subsets is non-empty and contains valid subsets only."""
        if not subsets:
            raise ValueError('No subsets specified.')
        for subset in subsets:
            if subset not in self.data_files.keys():
                raise ValueError('%r is not a valid subset.' % subset)
        return subsets

    def download(self):
        """Downloads and extracts self.subsets unless already cached.

        For each subset there are 3 possibilities:
            1. The `LibriSpeech/subset` directory exists and is valid making
               this function a noop.
            2. If not 1. but the `subset.tar.gz` archive file exists and is
               valid then it's contents are extracted.
            3. If not 2. then `subset.tar.gz` is downloaded, checksum verified,
               and it's contents extracted. The archive file is then removed
               leaving behind the `LibriSpeech/subset` directory.
        """
        os.makedirs(self.root, exist_ok=True)

        for subset in self.subsets:
            if self._check_subset_integrity(subset):
                print('%r already downloaded and verified.' % subset)
                continue
            path = os.path.join(self.root, subset + '.tar.gz')

            already_present = os.path.isfile(path)
            if not already_present:
                subset_url = self.openslr_url + subset + '.tar.gz'
                with requests.get(subset_url, stream=True) as r:
                    r.raise_for_status()
                    with open(path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)

            archive_md5 = self.data_files[subset]['archive_md5']
            if utils.checksum_file(path, 'md5') != archive_md5:
                raise utils.DownloadError('Invalid checksum for %r' % path)

            with tarfile.open(path, mode='r|gz') as tar:
                tar.extractall(self.root)

            if not already_present:
                os.remove(path)

    def _check_subset_integrity(self, subset):
        path = os.path.join(self.root, self.base_dir, subset)
        try:
            actual_md5 = utils.checksum_dir(path, 'md5')
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return False
        return actual_md5 == self.data_files[subset]['dir_md5']

    def check_integrity(self):
        """Returns True if each subset is valid."""
        return all([self._check_subset_integrity(s) for s in self.subsets])

    def load_data(self):
        """Loads the data from disk."""
        self.paths = []
        self.durations = []
        self.transcriptions = []

        def raise_(err):
            """raises error if problem during os.walk"""
            raise err

        for subset in self.subsets:
            subset_path = os.path.join(self.root, self.base_dir, subset)
            for root, dirs, files in os.walk(subset_path, onerror=raise_):
                if not files:
                    continue
                matches = fnmatch.filter(files, '*.trans.txt')
                assert len(matches) == 1, '> 1 transcription file found'
                self._parse_transcription_file(root, matches[0])

        self._sort_by_duration()

    def _parse_transcription_file(self, root, name):
        """Parses each sample in a transcription file."""
        trans_path = os.path.join(root, name)
        with open(trans_path, 'r', encoding='utf-8') as trans:
            # Each line has the form "ID THE TARGET TRANSCRIPTION"
            for line in trans:
                id_, transcript = line.split(maxsplit=1)
                self._process_audio(root, id_)
                self._process_transcript(transcript)

    def _process_audio(self, root, id):
        path = os.path.join(root, id + '.flac')
        self.paths.append(path)
        duration = soundfile.info(path).duration
        self.durations.append(duration)

    def _process_transcript(self, transcript):
        transcript = transcript.strip().upper()
        self.transcriptions.append(transcript)

    def _sort_by_duration(self):
        """Orders the loaded data by audio duration, shortest first."""
        total_samples = len(self.paths)
        samples = zip(self.paths, self.durations, self.transcriptions)
        sorted_samples = sorted(samples, key=lambda sample: sample[1])
        sorted_samples = [list(c) for c in zip(*sorted_samples)]
        self.paths, self.durations, self.transcriptions = sorted_samples
        assert (total_samples ==
                len(self.paths) ==
                len(self.durations) ==
                len(self.transcriptions)), '_sort_by_duration len mis-match'
