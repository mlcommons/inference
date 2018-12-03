import warnings
# librosa causes a RuntimeWarning - apparently safe to ignore:
# stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import librosa                  # noqa: E402
import numpy as np              # noqa: E402
import python_speech_features   # noqa: E402


class MFCC:
    """Compute the Mel-frequency cepstral coefficients (MFCC) of audiodata."""

    def __init__(self, numcep, winlen=0.025, winstep=0.02, sample_rate=16000):
        self.numcep = numcep
        self.winlen = winlen
        self.winstep = winstep
        self.sample_rate = sample_rate

    def __call__(self, audiodata):
        return python_speech_features.mfcc(audiodata,
                                           samplerate=self.sample_rate,
                                           winlen=self.winlen,
                                           winstep=self.winstep,
                                           numcep=self.numcep)

    def __repr__(self):
        params = '(numcep={0}, winlen={1}, winstep={2}, sample_rate={3})'
        params = params.format(self.numcep,
                               self.winlen,
                               self.winstep,
                               self.sample_rate)
        return self.__class__.__name__ + params


class LogMagnitudeSTFT:
    """Compute the log of the magnitude of the STFT of audiodata."""

    def __init__(self, winlen=0.02, winstep=0.01, sample_rate=16000):
        self.winlen = winlen
        self.winstep = winstep
        self.sample_rate = sample_rate

        self._n_fft = int(winlen * sample_rate)
        self._hop_length = int(winstep * sample_rate)

    def __call__(self, audiodata):
        """
        Args:
            audiodata: numpy ndarray of audio data with shape (N,).

        Returns:
            numpy ndarray X with shape (N, M). For each step in (0...N-1), the
            array X[step, :] contains the M log(1 + magnitude) components where
            M is equal to (int(winlen * sample_rate) / 2 + 1).
        """
        D = librosa.stft(audiodata,
                         n_fft=self._n_fft,
                         hop_length=self._hop_length,
                         win_length=self._n_fft,
                         window='hamming')
        mag, _ = librosa.magphase(D)
        return np.log1p(mag).T

    def __repr__(self):
        params = '(winlen={0}, winstep={1}, sample_rate={2})'
        params = params.format(self.winlen,
                               self.winstep,
                               self.sample_rate)
        return self.__class__.__name__ + params


class AddContextFrames:
    """Add context frames to each step in the original signal.

    Args:
        n_context: Number of context frames to add to frame in the original
            signal.

    Example:
        >>> signal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> n_context = 2
        >>> print(add_context_frames(signal, n_context))
        [[0 0 0 0 0 0 1 2 3 4 5 6 7 8 9]
         [0 0 0 1 2 3 4 5 6 7 8 9 0 0 0]
         [1 2 3 4 5 6 7 8 9 0 0 0 0 0 0]]
    """

    def __init__(self, n_context):
        self.n_context = n_context

    def __call__(self, signal):
        """
        Args:
            signal: numpy ndarray with shape (steps, features).

        Returns:
            numpy ndarray with shape:
                (steps, features * (n_context + 1 + n_context))
        """
        # Pad to ensure first and last n_context frames in original signal have
        # at least n_context frames to their left and right respectively.
        steps, features = signal.shape
        padding = np.zeros((self.n_context, features), dtype=signal.dtype)
        signal = np.concatenate((padding, signal, padding))

        window_size = self.n_context + 1 + self.n_context
        strided_signal = np.lib.stride_tricks.as_strided(
            signal,
            # Shape of the new array.
            (steps, window_size, features),
            # Strides of the new array (bytes to step in each dim).
            (signal.strides[0], signal.strides[0], signal.strides[1]),
            # Disable write to prevent accidental errors as elems share memory.
            writeable=False)

        # Flatten last dim and return a copy to permit writes.
        return strided_signal.reshape(steps, -1).copy()

    def __repr__(self):
        return self.__class__.__name__ + ('(n_context=%r)' % self.n_context)


class Normalize:
    """Normalize a tensor to have zero mean and one standard deviation."""

    def __call__(self, tensor):
        """
        Args:
            tensor: numpy ndarray
        """
        return (tensor - tensor.mean()) / tensor.std()

    def __repr__(self):
        return self.__class__.__name__ + '()'
