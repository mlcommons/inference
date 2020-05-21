import os
import toml
from pathlib import Path

from dataset import AudioToTextDataLayer
from helpers import add_blank_label
from model_separable_rnnt import RNNT
from preprocessing import AudioPreprocessing

import mlperf_loadgen
import torch
import torchvision
from tqdm import tqdm


def _get_top_dir():
    directory = Path(os.path.dirname(__file__))
    while not (directory / "requirements.txt").exists():
        old_directory = directory
        directory = directory.parent
        assert old_directory != directory, "Must run tests inside project directory"
    return directory


TOP: Path = _get_top_dir()
WORK_DIR: Path = TOP / "my_work_dir"
DATA_DIR: Path = WORK_DIR / "local_data"


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


def instantiate_models():
    model_definition = toml.load(TOP / "configs/rnnt.toml")
    checkpoint_path = WORK_DIR / "rnnt.pt"
    dataset_vocab = model_definition['labels']['labels']
    rnnt_vocab = add_blank_label(dataset_vocab)
    featurizer_config = model_definition['input_eval']

    data_layer = AudioToTextDataLayer(
        dataset_dir=DATA_DIR,
        featurizer_config=featurizer_config,
        manifest_filepath=DATA_DIR / "dev-clean-wav-small.json",
        labels=rnnt_vocab,
        batch_size=1,
        pad_to_max=featurizer_config['pad_to'] == "max",
        shuffle=False)

    audio_preprocessor = AudioPreprocessing(**featurizer_config)

    eval_transforms = []
    eval_transforms.append(lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]])
    # These are just some very confusing transposes, that's all.
    # BxFxT -> TxBxF
    eval_transforms.append(lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]])
    eval_transforms = torchvision.transforms.Compose(eval_transforms)

    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(rnnt_vocab)
    )
    model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path), strict=True)
    model.eval()

    return data_layer, eval_transforms, model


def test_encoder_torchscript_equivalence():
    data_layer, eval_transforms, model = instantiate_models()

    torchscript_encoder = torch.jit.script(model.encoder)

    for i, data in enumerate(tqdm(data_layer.data_iterator)):
        (t_audio_signal_e, t_a_sig_length_e,
         transcript_list, t_transcript_e,
         t_transcript_len_e) = eval_transforms(data)
        f, f_lens = model.encoder(t_audio_signal_e, t_a_sig_length_e)
        script_f, script_f_lens = torchscript_encoder(t_audio_signal_e, t_a_sig_length_e)
        assert torch.all(f_lens == script_f_lens), f"Length mismatch at datapoint {i}"
        assert torch.allclose(f, script_f), f"Mismatch at datapoint {i}"


def load_samples_to_ram(query_samples):
    del query_samples
    return


def unload_samples_from_ram(query_samples):
    del query_samples
    return


class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in len(self.indices))

    def __len__(self):
        return len(self.indices)


# Librispeech dev-clean is small enough for the whole thing to fit
# into RAM easily.
class LibrispeechDevCleanQSL:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._datalayer = None

    def load_query_samples(self, sample_list):
        self._kwargs["sampler"] = SubsetSampler()
        self._datalayer = AudioToTextDataLayer(**self._kwargs)

    def unload_query_samples(self, sample_list):
        # Just recreate self._datalayer next time load_query_samples is called.
        pass

    def get_features(self, sample_id):
        self._datalayer

def test_loadgen_offline_inference():
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.Offline
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.offline_expected_qps = 1000

    sut = mlperf_loadgen.ConstructSUT()


def test_onnxify_encoder(testdir):
    model_definition = toml.load(TOP / "configs/rnnt.toml")
    checkpoint_path = WORK_DIR / "rnnt.pt"
    dataset_vocab = model_definition['labels']['labels']
    rnnt_vocab = add_blank_label(dataset_vocab)
    featurizer_config = model_definition['input_eval']

    data_layer = AudioToTextDataLayer(
        dataset_dir=DATA_DIR,
        featurizer_config=featurizer_config,
        manifest_filepath=DATA_DIR / "dev-clean-wav-small.json",
        labels=rnnt_vocab,
        batch_size=1,
        pad_to_max=featurizer_config['pad_to'] == "max",
        shuffle=False)
    audio_preprocessor = torch.jit.script(AudioPreprocessing(**featurizer_config))

    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(rnnt_vocab)
    )
    model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path), strict=True)
    model.eval()
    model.encoder = torch.jit.script(model.encoder)
    model.encoder = torch.jit._recursive.wrap_cpp_module(
        torch._C._freeze_module(model.encoder._c))

    torch.onnx.export(model,
                      (t_audio_signal_e, t_a_sig_length_e),
                      'greedy_decoder.onnx',
                      verbose=True,
                      input_names=['input', 'input_length'],
                      example_outputs=(logits, logits_lens, t_predictions_e)
    )


def test_onnxify_joint():
    pass


def test_onnxify_featurization():
    model_definition = toml.load(TOP / "configs/rnnt.toml")
    dataset_vocab = model_definition['labels']['labels']
    rnnt_vocab = add_blank_label(dataset_vocab)
    featurizer_config = model_definition['input_eval']

    data_layer = AudioToTextDataLayer(
        dataset_dir=DATA_DIR,
        featurizer_config=featurizer_config,
        manifest_filepath=DATA_DIR / "dev-clean-wav-small.json",
        labels=rnnt_vocab,
        batch_size=1,
        pad_to_max=featurizer_config['pad_to'] == "max",
        shuffle=False)
    audio_preprocessor = torch.jit.script(AudioPreprocessing(**featurizer_config))

    eval_transforms = []
    # How to script lambdas? No idea!
    eval_transforms.append(lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]])
    # These are just some very confusing transposes, that's all.
    # BxFxT -> TxBxF
    eval_transforms.append(lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]])
    eval_transforms = torchvision.transforms.Compose(eval_transforms)

    sample_input = next(iter(data_layer.data_iterator))
    sample_output = eval_transforms(sample_input)
    pass
