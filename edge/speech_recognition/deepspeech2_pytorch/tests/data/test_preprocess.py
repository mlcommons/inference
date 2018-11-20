import numpy as np

from deepspeech.data import preprocess


def test_add_context_frames():
    """Ensures context frames are correctly added to each step in a signal."""
    steps, features, n_context = 10, 5, 3

    signal_flat = np.arange(steps * features)
    signal = signal_flat.reshape(steps, features)
    add_context_frames = preprocess.AddContextFrames(n_context)
    context_signal = add_context_frames(signal)

    assert len(context_signal) == steps

    # context_signal should be equal to sliding a window of size
    # (features * # (n_context + 1 + n_context)) with a step size of (features)
    # over a flattened, padded version of the original signal.
    pad = np.zeros(features * n_context)
    signal_flat_pad = np.concatenate((pad, signal_flat, pad))
    window_size = features * (n_context + 1 + n_context)

    for i, step in enumerate(context_signal):
        offset = i * features
        assert np.all(step == signal_flat_pad[offset:offset + window_size])


def test_normalize():
    """Ensures normalized tensor has mean zero, std one."""
    normalize = preprocess.Normalize()
    tensor = (np.random.rand(100, 10) - 0.5) * 100

    normed = normalize(tensor)
    assert np.allclose(normed.mean(), 0.0)
    assert np.allclose(normed.std(), 1.0)
