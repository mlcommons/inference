# Copyright (c) 2020, Cerebras Systems, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module is used for developing the onnx model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

from pathlib import Path

import numpy as np
import pytest
import toml
import torch

from pytorch.preprocessing import AudioPreprocessing
from QSL import AudioQSLInMemory

def _get_top_dir():
    directory = Path(os.path.dirname(__file__))
    while not (directory / "requirements.txt").exists():
        old_directory = directory
        directory = directory.parent
        assert old_directory != directory, "Must run tests inside project directory"
    return directory


TOP: Path = _get_top_dir()
WORK_DIR: Path = Path("/export/b07/ws15dgalvez/mlperf-rnnt-librispeech")
DATA_DIR: Path = Path("/export/b07/ws15dgalvez/mlperf-rnnt-librispeech/local_data")

def test_export_encoder():
    # sut = PytorchSUT(TOP / "pytorch/configs/rnnt.toml", WORK_DIR / "rnnt.pt",
    #                  DATA_DIR, DATA_DIR / "dev-clean-wav.json", 1)
    # qsl = sut.qsl
    config = toml.load(TOP / "pytorch/configs/rnnt.toml")
    qsl = AudioQSLInMemory(DATA_DIR,
                           DATA_DIR / "dev-clean-wav.json",
                           config['labels']['labels'],
                           config['input_eval']['sample_rate'],
                           1)
    audio_preprocessor = AudioPreprocessing(**config['input_eval'])
    audio_preprocessor.eval()
    audio_preprocessor = torch.jit.script(audio_preprocessor)
    audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
        torch._C._freeze_module(audio_preprocessor._c))
    audio_preprocessor._c._register_attribute("training", torch._C.BoolType.get(), False)
    with open("audio_preprocessor.txt", "w") as fh:
        fh.write(str(audio_preprocessor.forward.inlined_graph))
    # Possible way to convert between ScriptModule and ScriptFunction
    # https://github.com/pytorch/pytorch/issues/27343

    print("GALV: first lint")
    torch._C._jit_pass_lint(audio_preprocessor.forward.graph)

    waveform = qsl._load_sample(0)
    assert waveform.ndim == 1
    waveform_length = np.array(waveform.shape[0], dtype=np.int64)
    waveform = np.expand_dims(waveform, 0)
    waveform_length = np.expand_dims(waveform_length, 0)
    print("GALV:", waveform_length)

    # Do I need this? Does the existing torchscript module have a
    # sense of the backward pass inside it already?
    with torch.no_grad():
        waveform = torch.from_numpy(waveform)
        waveform_length = torch.from_numpy(waveform_length)

        feature, feature_length = audio_preprocessor.forward(waveform, waveform_length)

        print("GALV:", waveform_length.dtype)
        print("GALV:", waveform.dtype)
        # Try audio_preprocessor.forward
        torch.onnx.export(audio_preprocessor,
                          (waveform, waveform_length),
                          'audio_preprocessor.onnx',
                          # do_constant_folding=True,
                          verbose=True,
                          input_names=['waveform', 'waveform_length'],
                          example_outputs=(feature, feature_length))
    

def test_export_audio_processor():
    pass

def test_export_prediction():
    pass

def test_export_joint():
    pass

@pytest.mark.xfail
def text_export_decoder():
    pass
