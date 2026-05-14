# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import json
import argparse
import librosa
import soundfile as sf
import os
import numpy as np

MAX_DURATION = 30.0
PAD_DURATION = 0.5
SR = 16000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    return args


def get_source_name(fname):
    basename_list, _ = os.path.splitext(fname)
    return "-".join(basename_list.split("-")[:2])


def prepare_clip(current_entry, new_fname):
    pad_audio = np.zeros(int(PAD_DURATION * SR))
    new_audio = []
    new_transcript = ""
    for clip in current_entry:
        if len(new_audio) > 0:
            new_audio.append(pad_audio)
            new_transcript += " "
        new_audio.append(clip[0])
        new_transcript += clip[1]["transcript"]
    new_audio = np.concatenate(new_audio)
    new_json = get_sample_json(new_audio, new_transcript, new_fname)
    return new_audio, new_json


def get_sample_json(audio, transcript, fname):
    json_file = {
        "transcript": transcript,
        "files": [
            {
                "channels": 1,
                "sample_rate": float(SR),
                "bitdepth": 16,
                "bitrate": 256000.0,
                "duration": float(len(audio) / SR),
                "num_samples": int(len(audio)),
                "encoding": "Signed Integer PCM",
                "silent": False,
                "fname": fname,
                "speed": 1
            }
        ],
        "original_duration": float(len(audio) / SR),
        "original_num_samples": int(len(audio))
    }
    return json_file


def main():
    args = get_args()
    with open(args.manifest, "r") as manifest:
        json_data = json.load(manifest)

    os.makedirs(args.output_dir, exist_ok=True)

    catalog = dict()
    for data in json_data:
        original_fname = data["files"][0]["fname"]
        original_transcript = data["transcript"]
        original_audio = librosa.load(
            os.path.join(
                args.data_dir,
                original_fname),
            sr=SR)[0]
        original_json = get_sample_json(
            original_audio, original_transcript, original_fname)

        source_name = get_source_name(
            os.path.basename(
                os.path.basename(original_fname)))
        if source_name not in catalog:
            catalog[source_name] = []

        catalog[source_name].append((original_audio, original_json))

    full_json = []
    for key in catalog.keys():
        index = 0
        current_entry = []
        current_duration = 0
        for entry in catalog[key]:
            clip_duration = entry[1]["original_duration"]

            # Only considering clips <=30s.  If single clip duration > 30s,
            # ignore.
            if clip_duration > 30:
                continue
            # If new clip would extend compiled entry to >30s, flush the
            # existing entry
            if (len(current_entry) > 0) and (
                    current_duration + PAD_DURATION + clip_duration > 30):
                new_fname = os.path.join(
                    args.output_dir, key + "_" + str(index) + ".wav")
                new_audio, new_json = prepare_clip(current_entry, new_fname)
                sf.write(new_fname, new_audio, SR)
                full_json.append(new_json)
                current_entry = []
                current_duration = 0
                index += 1

            # Will append to existing or new list
            current_entry.append(entry)
            current_duration += clip_duration
            if len(current_entry) > 1:
                current_duration += PAD_DURATION

        # After all key clips are processed, if a remaining entry has content,
        # exports it.
        if len(current_entry) > 0:
            new_fname = os.path.join(
                args.output_dir, key + "_" + str(index) + ".wav")
            new_audio, new_json = prepare_clip(current_entry, new_fname)
            sf.write(new_fname, new_audio, SR)
            full_json.append(new_json)

    # Creates json manifest containing all newly-repacked clips
    with open(args.output_json, "w") as manifest:
        json.dump(full_json, manifest, indent=2)


if __name__ == "__main__":
    main()
