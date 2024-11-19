#  The MIT License (MIT)
#
#  Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the 'Software'), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from argparse import ArgumentParser
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer
from PIL import Image

import migraphx as mgx
from functools import wraps
from tqdm import tqdm
from hip import hip
from collections import namedtuple

import os
import sys
import torch
import time
import logging
import coco
import dataset

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger("mgx-base")

formatter = logging.Formatter("{levelname} - {message}", style="{")
file_handler = logging.FileHandler("mgx.log", mode="a", encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)



HipEventPair = namedtuple('HipEventPair', ['start', 'end'])


# measurement helper
def measure(fn):
    @wraps(fn)
    def measure_ms(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter_ns()
        print(
            f"Elapsed time for {fn.__name__}: {(end_time - start_time) * 1e-6:.4f} ms\n"
        )
        return result

    return measure_ms


def get_args():
    parser = ArgumentParser()
    # Model compile
    parser.add_argument(
        "--pipeline-type",
        type=str,
        choices=["sdxl", "sdxl-opt", "sdxl-turbo"],
        required=True,
        help="Specify pipeline type. Options: `sdxl`, `sdxl-opt`, `sdxl-turbo`",
    )

    parser.add_argument(
        "--onnx-model-path",
        type=str,
        default=None,
        help=
        "Path to onnx model files. Use it to override the default models/<sdxl*> path",
    )

    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default=None,
        help=
        "Path to compiled mxr model files. If not set, it will be saved next to the onnx model.",
    )

    parser.add_argument(
        "--use-refiner",
        action="store_true",
        default=False,
        help="Use the refiner model",
    )

    parser.add_argument(
        "--refiner-onnx-model-path",
        type=str,
        default=None,
        help=
        "Path to onnx model files. Use it to override the default models/<sdxl*> path",
    )

    parser.add_argument(
        "--refiner-compiled-model-path",
        type=str,
        default=None,
        help=
        "Path to compiled mxr model files. If not set, it will be saved next to the refiner onnx model.",
    )

    parser.add_argument(
        "--fp16",
        choices=[
            "all", "vae", "clip", "clip2", "unetxl", "refiner_clip2",
            "refiner_unetxl"
        ],
        nargs="+",
        help="Quantize models with fp16 precision.",
    )

    parser.add_argument(
        "--force-compile",
        action="store_true",
        default=False,
        help="Ignore existing .mxr files and override them",
    )

    parser.add_argument(
        "--exhaustive-tune",
        action="store_true",
        default=False,
        help="Perform exhaustive tuning when compiling onnx models",
    )

    # Runtime
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "-t",
        "--steps",
        type=int,
        default=20,
        help="Number of steps",
    )

    parser.add_argument(
        "--refiner-steps",
        type=int,
        default=20,
        help="Number of refiner steps",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        # required=True,
        help="Prompt",
    )

    parser.add_argument(
        "-n",
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="Guidance scale",
    )

    parser.add_argument(
        "--refiner-aesthetic-score",
        type=float,
        default=6.0,
        help="aesthetic score for refiner",
    )

    parser.add_argument(
        "--refiner-negative-aesthetic-score",
        type=float,
        default=2.5,
        help="negative aesthetic score for refiner",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output name",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Log during run",
    )
    return parser.parse_args()


model_shapes = {
    "clip": {
        "input_ids": [2, 77]
    },
    "clip2": {
        "input_ids": [2, 77]
    },
    "unetxl": {
        "sample": [2, 4, 128, 128],
        "encoder_hidden_states": [2, 77, 2048],
        "text_embeds": [2, 1280],
        "time_ids": [2, 6],
        "timestep": [1],
    },
    "refiner_unetxl": {
        "sample": [2, 4, 128, 128],
        "encoder_hidden_states": [2, 77, 1280],
        "text_embeds": [2, 1280],
        "time_ids": [2, 5],
        "timestep": [1],
    },
    "vae": {
        "latent_sample": [1, 4, 128, 128]
    },
}

model_names = {
    "sdxl": {
        "clip": "text_encoder",
        "clip2": "text_encoder_2",
        "unetxl": "unet",
        "vae": "vae_decoder",
    },
    "sdxl-opt": {
        "clip": "clip.opt.mod",
        "clip2": "clip2.opt.mod",
        "unetxl": "unetxl.opt",
        "vae": "vae_decoder",
    },
    "sdxl-turbo": {
        "clip": "text_encoder",
        "clip2": "text_encoder_2",
        "unetxl": "unet",
        "vae": "vae_decoder",
    },
    "refiner": {
        "clip2": "clip2.opt.mod",
        "unetxl": "unetxl.opt",
    },
}

default_model_paths = {
    "sdxl": "models/sdxl-1.0-base",
    "sdxl-opt": "models/sdxl-1.0-base",
    "sdxl-turbo": "models/sdxl-turbo",
    "refiner": "models/sdxl-1.0-refiner",
}

mgx_to_torch_dtype_dict = {
    "bool_type": torch.bool,
    "uint8_type": torch.uint8,
    "int8_type": torch.int8,
    "int16_type": torch.int16,
    "int32_type": torch.int32,
    "int64_type": torch.int64,
    "float_type": torch.float32,
    "double_type": torch.float64,
    "half_type": torch.float16,
}

torch_to_mgx_dtype_dict = {
    value: key
    for (key, value) in mgx_to_torch_dtype_dict.items()
}


def tensor_to_arg(tensor):
    return mgx.argument_from_pointer(
        mgx.shape(
            **{
                "type": torch_to_mgx_dtype_dict[tensor.dtype],
                "lens": list(tensor.size()),
                "strides": list(tensor.stride())
            }), tensor.data_ptr())


def tensors_to_args(tensors):
    return {name: tensor_to_arg(tensor) for name, tensor in tensors.items()}


def get_output_name(idx):
    return f"main:#output_{idx}"


def copy_tensor_sync(tensor, data):
    tensor.copy_(data.to(tensor.dtype))
    torch.cuda.synchronize()


def copy_tensor(tensor, data):
    tensor.copy_(data.to(tensor.dtype))


def run_model_sync(model, args):
    model.run(args)
    mgx.gpu_sync()


def run_model_async(model, args, stream):
    model.run_async(args, stream, "ihipStream_t")


def allocate_torch_tensors(model):
    input_shapes = model.get_parameter_shapes()
    data_mapping = {
        name: torch.zeros(shape.lens()).to(
            mgx_to_torch_dtype_dict[shape.type_string()]).to(device="cuda")
        for name, shape in input_shapes.items()
    }
    return data_mapping


class StableDiffusionMGX():
    def __init__(self, pipeline_type, onnx_model_path, compiled_model_path,
                 use_refiner, refiner_onnx_model_path,
                 refiner_compiled_model_path, fp16, force_compile,
                 exhaustive_tune, tokenizers=None, scheduler=None):
        if not (onnx_model_path or compiled_model_path):
            onnx_model_path = default_model_paths[pipeline_type]

        self.use_refiner = use_refiner
        if not self.use_refiner and (refiner_onnx_model_path
                                     or refiner_compiled_model_path):
            print(
                "WARN: Refiner model is provided, but was *not* enabled. Use --use-refiner to enable it."
            )
        if self.use_refiner and not (refiner_onnx_model_path
                                     or refiner_compiled_model_path):
            refiner_onnx_model_path = default_model_paths["refiner"]

        is_turbo = "turbo" in pipeline_type
        model_id = "stabilityai/sdxl-turbo" if is_turbo else "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"Using {model_id}")

        if scheduler is None:
            print("Creating EulerDiscreteScheduler scheduler")
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler")
        else:
            self.scheduler = scheduler

        print("Creating CLIPTokenizer tokenizers...")
        if tokenizers is None:
            tknz_path1 = os.path.join(onnx_model_path, "tokenizer")
            tknz_path2 = os.path.join(onnx_model_path, "tokenizer_2")
            self.tokenizers = {
                "clip":
                CLIPTokenizer.from_pretrained(tknz_path1),
                "clip2":
                CLIPTokenizer.from_pretrained(tknz_path2)
            }
        else:
            self.tokenizers = tokenizers

        if fp16 is None:
            fp16 = []
        elif "all" in fp16:
            fp16 = [
                "vae", "clip", "clip2", "unetxl", "refiner_clip2",
                "refiner_unetxl"
            ]

        if "vae" in fp16:
            model_names[pipeline_type]["vae"] = "vae_decoder_fp16_fix"

        log.info("Load models...")
        self.models = {
            "vae":
            StableDiffusionMGX.load_mgx_model(
                model_names[pipeline_type]["vae"],
                model_shapes["vae"],
                os.path.join (onnx_model_path, 'vae_decoder_fp16_fix'),
                compiled_model_path=compiled_model_path,
                use_fp16="vae" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "clip":
            StableDiffusionMGX.load_mgx_model(
                model_names[pipeline_type]["clip"],
                model_shapes["clip"],
                os.path.join (onnx_model_path, 'text_encoder'),
                compiled_model_path=compiled_model_path,
                use_fp16="clip" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "clip2":
            StableDiffusionMGX.load_mgx_model(
                model_names[pipeline_type]["clip2"],
                model_shapes["clip2"],
                os.path.join (onnx_model_path, 'text_encoder_2'),
                compiled_model_path=compiled_model_path,
                use_fp16="clip2" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "unetxl":
            StableDiffusionMGX.load_mgx_model(
                model_names[pipeline_type]["unetxl"],
                model_shapes["unetxl"],
                os.path.join (onnx_model_path, 'unet'),
                compiled_model_path=compiled_model_path,
                use_fp16="unetxl" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False)
        }
        
        log.info(f"init: loaded models")

        self.tensors = {
            "clip": allocate_torch_tensors(self.models["clip"]),
            "clip2": allocate_torch_tensors(self.models["clip2"]),
            "unetxl": allocate_torch_tensors(self.models["unetxl"]),
            "vae": allocate_torch_tensors(self.models["vae"]),
        }
        
        # log.info(f"init: tensors: {self.tensors}")

        self.model_args = {
            "clip": tensors_to_args(self.tensors["clip"]),
            "clip2": tensors_to_args(self.tensors["clip2"]),
            "unetxl": tensors_to_args(self.tensors["unetxl"]),
            "vae": tensors_to_args(self.tensors["vae"]),
        }
        
        # log.info(f"init: self.model_args: {self.model_args}")

        if self.use_refiner:
            log.info(f"init: self.use_refiner: {self.use_refiner}")
            
            # Note: there is no clip for refiner, only clip2
            self.models["refiner_clip2"] = StableDiffusionMGX.load_mgx_model(
                model_names["refiner"]["clip2"],
                model_shapes["clip2"],
                refiner_onnx_model_path,
                compiled_model_path=refiner_compiled_model_path,
                use_fp16="refiner_clip2" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False)
            
            log.info(f"init: load refiner clip2")
            
            self.models["refiner_unetxl"] = StableDiffusionMGX.load_mgx_model(
                model_names["refiner"]["unetxl"],
                model_shapes[
                    "refiner_unetxl"],  # this differ from the original unetxl
                refiner_onnx_model_path,
                compiled_model_path=refiner_compiled_model_path,
                use_fp16="refiner_unetxl" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False)
            
            log.info(f"init: load refiner unet")

            self.tensors["refiner_clip2"] = allocate_torch_tensors(
                self.models["refiner_clip2"])
            self.tensors["refiner_unetxl"] = allocate_torch_tensors(
                self.models["refiner_unetxl"])
            self.model_args["refiner_clip2"] = tensors_to_args(
                self.tensors["refiner_clip2"])
            self.model_args["refiner_unetxl"] = tensors_to_args(
                self.tensors["refiner_unetxl"])
        # hipEventCreate return a tuple(error, event)
        
        log.info(f"init: creating hip events")
        
        self.events = {
            "warmup":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
            "run":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
            "clip":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
            "denoise":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
            "decode":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
        }
        
        # log.info(f"init: self.events: {self.events}")

        self.stream = hip.hipStreamCreate()[1]
        
        # log.info(f"init: self.stream: {self.stream}")

    def cleanup(self):
        for event in self.events.values():
            hip.hipEventDestroy(event.start)
            hip.hipEventDestroy(event.end)
        hip.hipStreamDestroy(self.stream)

    def profile_start(self, name):
        if name in self.events:
            hip.hipEventRecord(self.events[name].start, None)

    def profile_end(self, name):
        if name in self.events:
            hip.hipEventRecord(self.events[name].end, None)

    # @measure
    @torch.no_grad()
    def run(self,
            prompt,
            steps=20,
            negative_prompt="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
            seed=42,
            scale=5.0,
            refiner_steps=20,
            refiner_aesthetic_score=6.0,
            refiner_negative_aesthetic_score=2.5,
            verbose=False,
            prompt_tokens=None,
            latents_in=None,
            device="cuda"):
        torch.cuda.synchronize()
        self.profile_start("run")
        # need to set this for each run
        self.scheduler.set_timesteps(steps, device=device)

        if verbose:
            print("Tokenizing prompts...")
            
        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens
        else:
            # log.info(f"[mgx] input prompt: {prompt}")
            prompt_tokens = self.tokenize(prompt, negative_prompt)
            # log.info(f"[mgx] clip token: {prompt_tokens[0]['input_ids']}")
            # log.info(f"[mgx] clip2 token: {prompt_tokens[1]['input_ids']}")
            
            # raise SystemExit("Checking if tokens match")

        if verbose:
            print("Creating text embeddings...")
        self.profile_start("clip")
        hidden_states, text_embeddings = self.get_embeddings(prompt_tokens)        
        # log.info(f"[mgx] hidden_states (shape {hidden_states.shape}): {hidden_states}")
        # log.info(f"[mgx] text_embeddings (shape {text_embeddings.shape}): {text_embeddings}")
        # log.info(f"------DIVIDER--------")
        self.profile_end("clip")
        sample_size = list(self.tensors["vae"]["latent_sample"].size())
        if verbose:
            print(
                f"Creating random input data {sample_size} (latents) with {seed = }..."
            )
        
        height, width = sample_size[2:]
        time_id = [height * 8, width * 8, 0, 0, height * 8, width * 8]
        time_ids = torch.tensor([time_id, time_id]).to(device=device)
        
        if latents_in is None:
            noise = torch.randn(
                sample_size, generator=torch.manual_seed(seed)).to(device=device)
            # input h/w crop h/w output h/w

            if verbose:
                print("Apply initial noise sigma\n")
            
            # print(f"noise.device -> {noise.device}")
            # print(f"self.scheduler.init_noise_sigma.device -> {self.scheduler.init_noise_sigma.device}")
            latents = noise * self.scheduler.init_noise_sigma
        else:
            
            if verbose:
                print("Apply initial noise sigma\n")
            
            # log.info(f"[MGX] input latents provided, no need to generate")
            latents = latents_in * self.scheduler.init_noise_sigma

        if verbose:
            print("Running denoising loop...")
        self.profile_start("denoise")
        for step, t in tqdm(enumerate(self.scheduler.timesteps), 
                    total=len(self.scheduler.timesteps), 
                    desc=f"Device {device} Denoising", 
                    ncols=100, 
                    leave=True):
            if verbose:
                print(f"#{step}/{len(self.scheduler.timesteps)} step")
            latents = self.denoise_step(text_embeddings,
                                        hidden_states,
                                        latents,
                                        t,
                                        scale,
                                        time_ids,
                                        model="unetxl",
                                        device=device)
        self.profile_end("denoise")
        if self.use_refiner and refiner_steps > 0:
            hidden_states, text_embeddings = self.get_embeddings(
                prompt_tokens, is_refiner=True)
            # input h/w crop h/w scores
            time_id_pos = time_id[:4] + [refiner_aesthetic_score]
            time_id_neg = time_id[:4] + [refiner_negative_aesthetic_score]
            time_ids = torch.tensor([time_id_pos,
                                     time_id_neg]).to(device=device)
            # need to set this for each run
            self.scheduler.set_timesteps(refiner_steps, device=device)
            # Add noise to latents using timesteps
            latents = self.scheduler.add_noise(latents, noise,
                                               self.scheduler.timesteps[:1])
            if verbose:
                print("Running refiner denoising loop...")
            for step, t in enumerate(self.scheduler.timesteps):
                if verbose:
                    print(f"#{step}/{len(self.scheduler.timesteps)} step")
                latents = self.denoise_step(text_embeddings,
                                            hidden_states,
                                            latents,
                                            t,
                                            scale,
                                            time_ids,
                                            model="refiner_unetxl",
                                            device=device)
        if verbose:
            print("Scale denoised result...")
        latents = 1 / 0.18215 * latents

        self.profile_start("decode")
        if verbose:
            print("Decode denoised result...")
        image = self.decode(latents)
        self.profile_end("decode")

        torch.cuda.synchronize()
        self.profile_end("run")
        # assert image.shape == (1, 3, 1024, 1024), f"Actual shape of image is: {image.shape}"
        return image

    def print_summary(self, denoise_steps):
        print('WARMUP\t{:>9.2f} ms'.format(
            hip.hipEventElapsedTime(self.events['warmup'].start,
                                    self.events['warmup'].end)[1]))
        print('CLIP\t{:>9.2f} ms'.format(
            hip.hipEventElapsedTime(self.events['clip'].start,
                                    self.events['clip'].end)[1]))
        print('UNetx{}\t{:>9.2f} ms'.format(
            str(denoise_steps),
            hip.hipEventElapsedTime(self.events['denoise'].start,
                                    self.events['denoise'].end)[1]))
        print('VAE-Dec\t{:>9.2f} ms'.format(
            hip.hipEventElapsedTime(self.events['decode'].start,
                                    self.events['decode'].end)[1]))
        print('RUN\t{:>9.2f} ms'.format(
            hip.hipEventElapsedTime(self.events['run'].start,
                                    self.events['run'].end)[1]))

    # @measure
    @staticmethod
    def load_mgx_model(name,
                       shapes,
                       onnx_model_path,
                       compiled_model_path=None,
                       use_fp16=False,
                       force_compile=False,
                       exhaustive_tune=False,
                       offload_copy=True):
        
        log.info(f"Zixian: inside load_mgx_model")
        print(f"Loading {name} model...")
        
        if compiled_model_path is None:
            compiled_model_path = onnx_model_path
        onnx_file = f"{onnx_model_path}/{name}/model.onnx"
        mxr_file = f"{compiled_model_path}/{name}/model_{'fp16' if use_fp16 else 'fp32'}_{'gpu' if not offload_copy else 'oc'}.mxr"
        log.info(f"Zixian: mxr_file: {mxr_file}")
        
        if not force_compile and os.path.isfile(mxr_file):
            print(f"Found mxr, loading it from {mxr_file}")
            model = mgx.load(mxr_file, format="msgpack")
        elif os.path.isfile(onnx_file):
            print(f"No mxr found at {mxr_file}")
            print(f"[IMPORTANT] Parsing from {onnx_file}")
            model = mgx.parse_onnx(onnx_file, map_input_dims=shapes)
            if use_fp16:
                print(f"[IMPORTANT] Unet quantizing to FP16...")
                mgx.quantize_fp16(model)
                
            
            model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=exhaustive_tune,
                          offload_copy=offload_copy)
            print(f"Saving {name} model to {mxr_file}")
            os.makedirs(os.path.dirname(mxr_file), exist_ok=True)
            mgx.save(model, mxr_file, format="msgpack")
        else:
            log.info(f"Zixian: no model found")
            print(
                f"No {name} model found at {onnx_file} or {mxr_file}. Please download it and re-try."
            )
            sys.exit(1)
        return model

    # @measure
    def tokenize(self, prompt, negative_prompt):
        def _tokenize(tokenizer):
            return self.tokenizers[tokenizer](
                [prompt, negative_prompt],
                padding="max_length",
                max_length=self.tokenizers[tokenizer].model_max_length,
                truncation=True,
                return_tensors="pt")

        tokens = _tokenize("clip")
        tokens2 = _tokenize("clip2")
        return (tokens, tokens2)

    # @measure
    def get_embeddings(self, prompt_tokens, is_refiner=False):
        def _create_embedding(model, input):
            copy_tensor(self.tensors[model]["input_ids"], input.input_ids)
            run_model_async(self.models[model], self.model_args[model],
                            self.stream)

        clip_input, clip2_input = prompt_tokens
        clip, clip2 = "clip", ("refiner_" if is_refiner else "") + "clip2"
        if not is_refiner:
            _create_embedding(clip, clip_input)
        _create_embedding(clip2, clip2_input)

        hidden_states = torch.concatenate(
            (self.tensors[clip][get_output_name(0)],
             self.tensors[clip2][get_output_name(1)]),
            axis=2) if not is_refiner else self.tensors[clip2][get_output_name(
                1)]
        text_embeds = self.tensors[clip2][get_output_name(0)]
        return (hidden_states, text_embeds)

    @staticmethod
    def convert_to_rgb_image(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return Image.fromarray(images[0])

    @staticmethod
    def save_image(pil_image, filename="output.png"):
        pil_image.save(filename)

    # @measure
    def denoise_step(self, text_embeddings, hidden_states, latents, t, scale,
                     time_ids, model, device):
        latents_model_input = torch.cat([latents] * 2)
        latents_model_input = self.scheduler.scale_model_input(
            latents_model_input, t).to(device=device)
        timestep = torch.atleast_1d(t.to(device=device))  # convert 0D -> 1D

        copy_tensor(self.tensors[model]["sample"], latents_model_input)
        copy_tensor(self.tensors[model]["encoder_hidden_states"],
                    hidden_states)
        copy_tensor(self.tensors[model]["text_embeds"], text_embeddings)
        copy_tensor(self.tensors[model]["timestep"], timestep)
        copy_tensor(self.tensors[model]["time_ids"], time_ids)
        run_model_async(self.models[model], self.model_args[model],
                        self.stream)

        noise_pred_text, noise_pred_uncond = torch.tensor_split(
            self.tensors[model][get_output_name(0)], 2)

        # perform guidance
        noise_pred = noise_pred_uncond + scale * (noise_pred_text -
                                                  noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        return self.scheduler.step(noise_pred, t, latents).prev_sample

    # @measure
    def decode(self, latents):
        copy_tensor(self.tensors["vae"]["latent_sample"], latents)
        run_model_async(self.models["vae"], self.model_args["vae"],
                        self.stream)
        return self.tensors["vae"][get_output_name(0)]

    # @measure
    def warmup(self, num_runs):
        self.profile_start("warmup")
        init_fn = lambda x: torch.ones if "clip" in x else torch.randn
        for model in self.models.keys():
            for tensor in self.tensors[model].values():
                copy_tensor(tensor, init_fn(model)(tensor.size()))

        for _ in range(num_runs):
            for model in self.models.keys():
                run_model_async(self.models[model], self.model_args[model],
                                self.stream)
        self.profile_end("warmup")


if __name__ == "__main__":
    args = get_args()

    # sd = StableDiffusionMGX(args.pipeline_type, args.onnx_model_path,
    #                         args.compiled_model_path, args.use_refiner,
    #                         args.refiner_onnx_model_path,
    #                         args.refiner_compiled_model_path, args.fp16,
    #                         args.force_compile, args.exhaustive_tune)
    
    sd = StableDiffusionMGX("sdxl", onnx_model_path=args.onnx_model_path,
                            compiled_model_path=None, use_refiner=False,
                            refiner_onnx_model_path=None,
                            refiner_compiled_model_path=None, fp16=args.fp16,
                            force_compile=False, exhaustive_tune=True)
    print("Warmup")
    sd.warmup(5)
    print("Run")

    prompt_list = []
    prompt_list.append(["A young man in a white shirt is playing tennis.", "tennis.jpg"])
    # prompt_list.append(["Lorem ipsum dolor sit amet, consectetur adipiscing elit", "woman.jpg"])
    prompt_list.append(["Astronaut crashlanding in Madison Square Garden, cold color palette, muted colors, detailed, 8k", "crash_astro.jpg"])
    # prompt_list.append(["John Cena giving The Rock an Attitude Adjustment off the roof, warm color palette, vivid colors, detailed, 8k", "cena_rock.jpg"])

    for element in prompt_list:
        prompt, img_name = element[0], element[1]
        # result = sd.run(prompt, args.negative_prompt, args.steps, args.seed,
        #         args.scale, args.refiner_steps,
        #         args.refiner_aesthetic_score,
        #         args.refiner_negative_aesthetic_score, args.verbose)
        
        result = sd.run(prompt=prompt, steps=20, seed=args.seed,
                scale=5.0, refiner_steps=0,
                refiner_aesthetic_score=0.0,
                refiner_negative_aesthetic_score=0.0, verbose=False)

        print("Summary")
        sd.print_summary(args.steps)        

        print("Convert result to rgb image...")
        image = StableDiffusionMGX.convert_to_rgb_image(result)
        StableDiffusionMGX.save_image(image, img_name)
        print(f"Image saved to {img_name}")

    print("Cleanup")
    sd.cleanup()