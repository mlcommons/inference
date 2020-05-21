# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import argparse
from tqdm import tqdm
import toml
from dataset import AudioToTextDataLayer
from helpers import process_evaluation_batch, process_evaluation_epoch, add_blank_label, print_dict
from decoders import RNNTGreedyDecoder, ScriptGreedyDecoder
from model_separable_rnnt import RNNT
from preprocessing import AudioPreprocessing
import torch
import random
import numpy as np
import pickle

from copy import deepcopy

import torchvision


def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--batch_size", default=16,
                        type=int, help='data batch size')
    parser.add_argument("--steps", default=None,
                        help='if not specified do evaluation on full dataset. otherwise only evaluates the specified number of iterations for each worker', type=int)
    parser.add_argument("--model_toml", type=str,
                        help='relative model configuration path given dataset folder')
    parser.add_argument("--dataset_dir", type=str,
                        help='absolute path to dataset folder')
    parser.add_argument("--val_manifest", type=str,
                        help='relative path to evaluation dataset manifest file')
    parser.add_argument("--ckpt", default=None, type=str,
                        required=True, help='path to model checkpoint')
    parser.add_argument("--pad_to", default=None, type=int,
                        help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    parser.add_argument("--cudnn_benchmark",
                        action='store_true', help="enable cudnn benchmark")
    parser.add_argument("--save_prediction", type=str, default=None,
                        help="if specified saves predictions in text form at this location")
    parser.add_argument("--logits_save_to", default=None,
                        type=str, help="if specified will save logits to path")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--cuda",
                        action='store_true', help="use cuda", default=False)
    parser.add_argument("--runtime", choices=["onnxruntime", "pytorch", "torchscript"])
    return parser.parse_args()


def eval(
        data_layer,
        audio_processor,
        encoderdecoder,
        greedy_decoder,
        labels,
        args):
    """performs inference / evaluation
    Args:
        data_layer: data layer object that holds data loader
        audio_processor: data processing module
        encoderdecoder: acoustic model
        greedy_decoder: greedy decoder
        labels: list of labels as output vocabulary
        args: script input arguments
    """
    # Register as Torchscript class first
    # audio_processor = torch.jit.script(audio_processor)
    logits_save_to = args.logits_save_to
    with torch.no_grad():
        _global_var_dict = {
            'predictions': [],
            'transcripts': [],
            'logits': [],
        }

        for it, data in enumerate(tqdm(data_layer.data_iterator)):
            # if it == 0:
            #     torch.onnx.export()

            # TODO: Make this part of this scripted model
            (t_audio_signal_e, t_a_sig_length_e,
             transcript_list, t_transcript_e,
             t_transcript_len_e) = audio_processor(data)

            # with open("predict.txt", "w") as fh:
            #     fh.write(str(greedy_decoder._model.prediction.forward.inlined_graph))
            # with open("joint.txt", "w") as fh:
            #     fh.write(str(greedy_decoder._model.joint.forward.inlined_graph))
            # with open("encode.txt", "w") as fh:
            #     fh.write(str(greedy_decoder._model.forward.inlined_graph))
            # torch._C._jit_pass_dce(greedy_decoder._model.forward.graph)
            # with open("encode_dce.txt", "w") as fh:
            #     fh.write(str(greedy_decoder._model.forward.inlined_graph))

            # print(str(greedy_decoder._model))

            # print(type(greedy_decoder._model))

            # Now why can't _c have a setter which automatically calls wrap_cpp_module()?

            # When I assign to decoder._model, it complains that the
            # type is different. I suppose that may make sense. I have
            # changed the type, since some non-constant values are now
            # constant!
            model = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(greedy_decoder._model._c))

            # TODO: Why does this silently do nothing? Try using
            # tracing in a separate unit-test.
            # greedy_decoder._model._c = \
            #     torch._C._freeze_module(greedy_decoder._model._c)

            # with open("encode_frozen.txt", "w") as fh:
            #     fh.write(str(model.forward.inlined_graph))
            
            # import sys; sys.exit(0)

            logits, logits_lens, t_predictions_e = greedy_decoder(t_audio_signal_e, t_a_sig_length_e)

            # torch.onnx.export(model,
            #                   (t_audio_signal_e, t_a_sig_length_e),
            #                   'greedy_decoder.onnx',
            #                   verbose=True,
            #                   input_names=['input', 'input_length'],
            #                   example_outputs=(logits, logits_lens, t_predictions_e)
            # )


            values_dict = dict(
                predictions=[t_predictions_e],
                transcript=transcript_list,
                transcript_length=t_transcript_len_e,
            )
            process_evaluation_batch(
                values_dict, _global_var_dict, labels=labels)

            if args.steps is not None and it + 1 >= args.steps:
                break
        wer = process_evaluation_epoch(_global_var_dict)
        print("==========>>>>>>Evaluation WER: {0}\n".format(wer))
        if args.save_prediction is not None:
            with open(args.save_prediction, 'w') as fp:
                fp.write('\n'.join(_global_var_dict['predictions']))
        if logits_save_to is not None:
            logits = []
            with open(logits_save_to, 'wb') as f:
                pickle.dump(logits, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    print("CUDNN BENCHMARK ", args.cudnn_benchmark)
    if args.cuda:
        assert(torch.cuda.is_available())

    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    rnnt_vocab = add_blank_label(dataset_vocab)

    val_manifest = args.val_manifest
    featurizer_config = model_definition['input_eval']

    if args.pad_to is not None:
        featurizer_config['pad_to'] = args.pad_to if args.pad_to >= 0 else "max"

    print('model_config')
    print_dict(model_definition)
    print('feature_config')
    print_dict(featurizer_config)

    data_layer = AudioToTextDataLayer(
        dataset_dir=args.dataset_dir,
        featurizer_config=featurizer_config,
        manifest_filepath=val_manifest,
        labels=dataset_vocab,
        batch_size=args.batch_size,
        pad_to_max=featurizer_config['pad_to'] == "max",
        shuffle=False)

    audio_preprocessor = AudioPreprocessing(**featurizer_config)

    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(rnnt_vocab)
    )

    if args.ckpt is not None:
        print("loading model from ", args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        migrated_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            # key = key.replace("encoder.", "encoder_").replace("prediction.", "prediction_")
            key = key.replace("joint_net", "joint.net")
            migrated_state_dict[key] = value
        del migrated_state_dict["audio_preprocessor.featurizer.fb"]
        del migrated_state_dict["audio_preprocessor.featurizer.window"]
        model.load_state_dict(migrated_state_dict, strict=True)

    audio_preprocessor.featurizer.normalize = "per_feature"

    if args.cuda:
        audio_preprocessor.cuda()
    audio_preprocessor.eval()

    eval_transforms = []
    if args.cuda:
        eval_transforms.append(lambda xs: [x.cuda() for x in xs])
    eval_transforms.append(lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]])
    # These are just some very confusing transposes, that's all.
    # BxFxT -> TxBxF
    eval_transforms.append(lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]])
    eval_transforms = torchvision.transforms.Compose(eval_transforms)

    if args.cuda:
        model.cuda()

    # greedy_decoder = RNNTGreedyDecoder(len(rnnt_vocab) - 1, model)
    model.eval()
    # torch.nn.LSTM.__constants__.append('training')
    greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, torch.jit.script(model))
    greedy_decoder = torch.jit.script(greedy_decoder)
    # torch.nn.LSTM.__constants__.pop()
    eval(
        data_layer=data_layer,
        audio_processor=eval_transforms,
        encoderdecoder=model,
        greedy_decoder=greedy_decoder,
        labels=rnnt_vocab,
        args=args)


if __name__ == "__main__":
    args = parse_args()

    print_dict(vars(args))

    main(args)
