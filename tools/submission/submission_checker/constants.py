MODEL_CONFIG = {
    "v6.0": {
        "models": [
            "resnet",
            "retinanet",
            "bert-99",
            "bert-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
            "llama3.1-8b",
            "llama3.1-8b-edge",
            "llama2-70b-99",
            "llama2-70b-99.9",
            "stable-diffusion-xl",
            "mixtral-8x7b",
            "llama3.1-405b",
            "rgat",
            "pointpainting",
            "deepseek-r1",
            "whisper",
            "gpt-oss-120b",
            "wan-2.2-t2v-a14b",
            "qwen3-vl-235b-a22b",
            "dlrm-v3",
            "yolo-95",
            "yolo-99",
        ],
        "required-scenarios-datacenter": {
            "dlrm-v3": ["Server", "Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
            "llama3.1-8b": ["Offline"],
            "llama2-70b-99": ["Offline"],
            "llama2-70b-99.9": ["Offline"],
            "mixtral-8x7b": ["Server", "Offline"],
            "llama3.1-405b": ["Offline"],
            "rgat": ["Offline"],
            "whisper": ["Offline"],
            "deepseek-r1": ["Offline"],
            "gpt-oss-120b": ["Offline"],
            "qwen3-vl-235b-a22b": ["Server", "Offline"],
            "wan-2.2-t2v-a14b": ["Offline", "SingleStream"],
        },
        "optional-scenarios-datacenter": {
            "llama2-70b-99": ["Interactive", "Server"],
            "llama2-70b-99.9": ["Interactive", "Server"],
            "llama3.1-405b": ["Interactive", "Server"],
            "llama3.1-8b": ["Interactive", "Server"],
            "deepseek-r1": ["Interactive", "Server"],
            "gpt-oss-120b": ["Interactive", "Server"],
        },
        "required-scenarios-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
            "llama3.1-8b-edge": ["SingleStream", "Offline"],
            "stable-diffusion-xl": ["SingleStream", "Offline"],
            "pointpainting": ["SingleStream"],
            "whisper": ["Offline"],
            "yolo-95": ["SingleStream", "MultiStream", "Offline"],
            "yolo-99": ["SingleStream", "MultiStream", "Offline"],
        },
        "optional-scenarios-edge": {},
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline", "Server"],
            "retinanet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
            "llama3.1-8b": ["Offline"],
            "llama3.1-8b-edge": ["SingleStream", "Offline"],
            "llama2-70b-99": ["Offline"],
            "llama2-70b-99.9": ["Offline"],
            "stable-diffusion-xl": ["SingleStream", "Offline", "Server"],
            "mixtral-8x7b": ["Server", "Offline"],
            "llama3.1-405b": ["Offline"],
            "rgat": ["Offline"],
            "pointpainting": ["SingleStream"],
            "deepseek-r1": ["Offline"],
            "whisper": ["Offline"],
            "gpt-oss-120b": ["Offline"],
            "qwen3-vl-235b-a22b": ["Offline"],
            "dlrm-v3": ["Offline", "Server"],
            "yolo-95": ["SingleStream", "MultiStream", "Offline"],
            "yolo-99": ["SingleStream", "MultiStream", "Offline"],
        },
        "optional-scenarios-datacenter-edge": {
            "llama2-70b-99": ["Interactive", "Server"],
            "llama2-70b-99.9": ["Interactive", "Server"],
            "llama3.1-405b": ["Interactive", "Server"],
            "llama3.1-8b": ["Interactive", "Server"],
            "deepseek-r1": ["Interactive", "Server"],
            "gpt-oss-120b": ["Interactive", "Server"],
            "qwen3-vl-235b-a22b": ["Interactive", "Server"],
        },
        "accuracy-target": {
            "resnet": ("acc", 76.46 * 0.99),
            "retinanet": ("mAP", 37.55 * 0.99),
            "bert-99": ("F1", 90.874 * 0.99),
            "bert-99.9": ("F1", 90.874 * 0.999),
            "dlrm-v2-99": ("AUC", 80.31 * 0.99),
            "dlrm-v2-99.9": ("AUC", 80.31 * 0.999),
            "3d-unet-99": ("DICE", 0.86170 * 0.99),
            "3d-unet-99.9": ("DICE", 0.86170 * 0.999),

            "llama3.1-8b": (
                "ROUGE1",
                38.7792 * 0.99,
                "ROUGE2",
                15.9075 * 0.99,
                "ROUGEL",
                24.4957 * 0.99,
                "ROUGELSUM",
                35.793 * 0.99,
                "GEN_LEN",
                8167644 * 0.9,
            ),
            "llama3.1-8b-edge": (
                "ROUGE1",
                39.06 * 0.99,
                "ROUGE2",
                16.1147 * 0.99,
                "ROUGEL",
                24.6375 * 0.99,
                "ROUGELSUM",
                36.124 * 0.99,
                "GEN_LEN",
                3051113 * 0.9,
            ),
            "llama2-70b-99": (
                "ROUGE1",
                44.4312 * 0.99,
                "ROUGE2",
                22.0352 * 0.99,
                "ROUGEL",
                28.6162 * 0.99,
                "TOKENS_PER_SAMPLE",
                294.45 * 0.9,
            ),
            "llama2-70b-99.9": (
                "ROUGE1",
                44.4312 * 0.999,
                "ROUGE2",
                22.0352 * 0.999,
                "ROUGEL",
                28.6162 * 0.999,
                "TOKENS_PER_SAMPLE",
                294.45 * 0.9,
            ),
            "stable-diffusion-xl": (
                "CLIP_SCORE",
                31.68631873,
                "FID_SCORE",
                23.01085758,
            ),
            "mixtral-8x7b": (
                "ROUGE1",
                45.5989 * 0.99,
                "ROUGE2",
                23.3526 * 0.99,
                "ROUGEL",
                30.4608 * 0.99,
                "TOKENS_PER_SAMPLE",
                144.84 * 0.9,
                "gsm8k_accuracy",
                73.66 * 0.99,
                "mbxp_accuracy",
                60.16 * 0.99,
            ),
            "llama3.1-405b": (
                "ROUGEL",
                21.6666 * 0.99,
                "exact_match",
                90.1335 * 0.99,
                "TOKENS_PER_SAMPLE",
                684.68 * 0.9,
            ),
            "rgat": ("acc", 0.7286 * 0.99),
            "pointpainting": ("mAP", 0.5425 * 0.999),
            "deepseek-r1": ("exact_match", 0.99 * 81.3582, "TOKENS_PER_SAMPLE", 0.9 * 3886.2274),
            "whisper": ("ACCURACY", (100.0 - 2.0671) * 0.99),
            "gpt-oss-120b": ("exact_match", 83.13 * 0.99),
            # TODO: Placeholder for now
            "qwen3-vl-235b-a22b": ("F1_HIERARCHICAL", 0.7903 * 0.99),
            "dlrm-v3": (
                "DLRM_NE",
                0.86687 * 0.999,
                "DLRM_ACC",
                0.69651 * 0.999,
                "DLRM_AUC",
                0.78663 * 0.999,
            ),
            "yolo-95": ("mAP", 53.4 * 0.95),
            "yolo-99": ("mAP", 53.4 * 0.99),
            "wan-2.2-t2v-a14b": ("vbench_score", 70.48 * 0.99),
        },
        "accuracy-upper-limit": {
            "stable-diffusion-xl": (
                "CLIP_SCORE",
                31.81331801,
                "FID_SCORE",
                23.95007626,
            ),
            "llama2-70b-99": ("TOKENS_PER_SAMPLE", 294.45 * 1.1),
            "llama2-70b-99.9": ("TOKENS_PER_SAMPLE", 294.45 * 1.1),
            "mixtral-8x7b": ("TOKENS_PER_SAMPLE", 145.9 * 1.1),
            "llama3.1-405b": ("TOKENS_PER_SAMPLE", 684.68 * 1.1),
            "llama3.1-8b": ("GEN_LEN", 8167644 * 1.1),
            "llama3.1-8b-edge": ("GEN_LEN", 3051113 * 1.1),
            "deepseek-r1": ("TOKENS_PER_SAMPLE", 1.1 * 3886.2274),
            # TODO: Placeholder for now
            "gpt-oss-120b": ("TOKENS_PER_SAMPLE", 1.1 * 9999),
        },
        "accuracy-delta-perc": {
            "stable-diffusion-xl": {"CLIP_SCORE": 1, "FID_SCORE": 2}
        },
        "performance-sample-count": {
            "resnet": 1024,
            "retinanet": 64,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-v2-99": 204800,
            "dlrm-v2-99.9": 204800,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
            "llama3.1-8b": 13368,
            "llama3.1-8b-edge": 5000,
            "llama2-70b-99": 24576,
            "llama2-70b-99.9": 24576,
            "stable-diffusion-xl": 5000,
            "mixtral-8x7b": 15000,
            "llama3.1-405b": 8313,
            "rgat": 788379,
            "pointpainting": 1024,
            "deepseek-r1": 4388,
            "whisper": 1633,
            "gpt-oss-120b": 6396,
            "qwen3-vl-235b-a22b": 48289,
            "wan-2.2-t2v-a14b": 50,
            "dlrm-v3": 349823,
            "yolo-95": 64,
            "yolo-99": 64,
        },
        "accuracy-sample-count": {
            "gpt-oss-120b": 4395,
            "wan-2.2-t2v-a14b": 248,
        },
        "dataset-size": {
            "resnet": 50000,
            "retinanet": 24781,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-v2-99": 330067,
            "dlrm-v2-99.9": 330067,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
            "llama3.1-8b": 13368,
            "llama3.1-8b-edge": 5000,
            "llama2-70b-99": 24576,
            "llama2-70b-99.9": 24576,
            "stable-diffusion-xl": 5000,
            "mixtral-8x7b": 15000,
            "llama3.1-405b": 8313,
            "rgat": 788379,
            "pointpainting": 39987,
            "deepseek-r1": 4388,
            "whisper": 1633,
            # TODO: Need to add accuracy sample count checkers as well (4395)
            "gpt-oss-120b": 6396,
            "qwen3-vl-235b-a22b": 48289,
            "wan-2.2-t2v-a14b": 248,
            "dlrm-v3": 349823,
            "yolo-95": 1525,
            "yolo-99": 1525,
        },
        # model_mapping.json is expected in the root directory of the
        # submission folder for open submissions and so the below dictionary is
        # not really needed
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-resnet34": "retinanet",
            "mobilenet": "resnet",
            "resnet50": "resnet",
            "llama3_1-405b": "llama3.1-405b",
            "llama3_1-8b": "llama3.1-8b",
            "llama3_1-8b-edge": "llama3.1-8b-edge",
        },
        "seeds": {
            # TODO: Update random seeds
            "qsl_rng_seed": 2465351861681999779,
            "sample_index_rng_seed": 14276810075590677512,
            "schedule_rng_seed": 3936089224930324775,
        },
        "ignore_errors": [],
        "latency-constraint": {
            "resnet": {"Server": 15000000},
            "retinanet": {"Server": 100000000},
            "dlrm-v2-99": {"Server": 60000000},
            "dlrm-v2-99.9": {"Server": 60000000},
            "llama3.1-8b": {"Server": 20000000000},
            "stable-diffusion-xl": {"Server": 20000000000},
            "llama2-70b-99": {"Server": 20000000000},
            "llama2-70b-99.9": {"Server": 20000000000},
            "mixtral-8x7b": {"Server": 20000000000},
            "llama3.1-405b": {"Server": 60000000000},
            "deepseek-r1": {"Server": 60000000000},
            "gpt-oss-120b": {"Server": 60000000000},
            "qwen3-vl-235b-a22b": {"Server": 60000000000},
            "dlrm-v3": {"Server": 60000000000},
        },
        "min-queries": {
            "resnet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "retinanet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "bert-99": {"SingleStream": 1024, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Offline": 1},
            "dlrm-v2-99": {"Server": 270336, "Offline": 1},
            "dlrm-v2-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
            "llama3.1-8b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama3.1-8b-edge": {"SingleStream": 1024, "Offline": 1},
            "llama2-70b-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama2-70b-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "stable-diffusion-xl": {
                "SingleStream": 1024,
                "Server": 270336,
                "Offline": 1,
            },
            "mixtral-8x7b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama3.1-405b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "rgat": {"SingleStream": 1024, "Offline": 1},
            "pointpainting": {"SingleStream": 1024},
            "deepseek-r1": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "whisper": {"SingleStream": 1024, "Offline": 1},
            "gpt-oss-120b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "qwen3-vl-235b-a22b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-v3": {"Server": 270336, "Offline": 1},
            "wan-2.2-t2v-a14b": {"SingleStream": 50, "Offline": 1},
            "yolo-95": {"SingleStream": 1024, "MultiStream": 270336, "Offline": 1},
            "yolo-99": {"SingleStream": 1024, "MultiStream": 270336, "Offline": 1},
        },
        "models_TEST01": [
            "resnet",
            "retinanet",
            "bert-99",
            "bert-99.9",
            "dlrm-v2-99",
            "dlrm-v2-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
            "stable-diffusion-xl",
            "rgat",
            "pointpainting",
            "whisper",
            "yolo-99",
            "yolo-95",
        ],
        "models_TEST04": [
            "resnet",
            "stable-diffusion-xl",
            "pointpainting",
            "wan-2.2-t2v-a14b"
        ],
        "models_TEST06": [
            "llama2-70b-99",
            "llama2-70b-99.9",
            "llama2-70b-interactive-99",
            "llama2-70b-interactive-99.9",
            "llama3.1-405b",
            "llama3.1-8b",
            "llama3.1-8b-interactive",
            "llama3.1-405b-interactive",
            "mixtral-8x7b",
            "deepseek-r1",
        ],
        "models_TEST07": [
            "gpt-oss-120b",
        ],
        "models_TEST09": [
            "gpt-oss-120b",
        ],
        "models_TEST08": [
            "dlrm-v3",
        ]
    },
    "v5.0": {
        "models": [
            "resnet",
            "retinanet",
            "bert-99",
            "bert-99.9",
            "dlrm-v2-99",
            "dlrm-v2-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
            "gptj-99",
            "gptj-99.9",
            "llama2-70b-99",
            "llama2-70b-99.9",
            "llama2-70b-interactive-99",
            "llama2-70b-interactive-99.9",
            "stable-diffusion-xl",
            "mixtral-8x7b",
            "llama3.1-405b",
            "rgat",
            "pointpainting",
        ],
        "required-scenarios-datacenter": {
            "resnet": ["Server", "Offline"],
            "retinanet": ["Server", "Offline"],
            "dlrm-v2-99": ["Server", "Offline"],
            "dlrm-v2-99.9": ["Server", "Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
            "gptj-99": ["Server", "Offline"],
            "gptj-99.9": ["Server", "Offline"],
            "llama2-70b-99": ["Server", "Offline"],
            "llama2-70b-99.9": ["Server", "Offline"],
            "llama2-70b-interactive-99": ["Server", "Offline"],
            "llama2-70b-interactive-99.9": ["Server", "Offline"],
            "stable-diffusion-xl": ["Server", "Offline"],
            "mixtral-8x7b": ["Server", "Offline"],
            "llama3.1-405b": ["Server", "Offline"],
            "rgat": ["Offline"],
        },
        "optional-scenarios-datacenter": {},
        "required-scenarios-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline"],
            "retinanet": ["SingleStream", "MultiStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
            "gptj-99": ["SingleStream", "Offline"],
            "gptj-99.9": ["SingleStream", "Offline"],
            "stable-diffusion-xl": ["SingleStream", "Offline"],
            "pointpainting": ["SingleStream"],
        },
        "optional-scenarios-edge": {},
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "retinanet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["SingleStream", "Offline"],
            "dlrm-v2-99": ["Offline", "Server"],
            "dlrm-v2-99.9": ["Offline", "Server"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
            "gptj-99": ["SingleStream", "Offline", "Server"],
            "gptj-99.9": ["SingleStream", "Offline", "Server"],
            "llama2-70b-99": ["Server", "Offline"],
            "llama2-70b-99.9": ["Server", "Offline"],
            "llama2-70b-interactive-99": ["Server", "Offline"],
            "llama2-70b-interactive-99.9": ["Server", "Offline"],
            "stable-diffusion-xl": ["SingleStream", "Offline", "Server"],
            "mixtral-8x7b": ["Server", "Offline"],
            "llama3.1-405b": ["Server", "Offline"],
            "rgat": ["Offline"],
            "pointpainting": ["SingleStream"],
        },
        "optional-scenarios-datacenter-edge": {},
        "accuracy-target": {
            "resnet": ("acc", 76.46 * 0.99),
            "retinanet": ("mAP", 37.55 * 0.99),
            "bert-99": ("F1", 90.874 * 0.99),
            "bert-99.9": ("F1", 90.874 * 0.999),
            "dlrm-v2-99": ("AUC", 80.31 * 0.99),
            "dlrm-v2-99.9": ("AUC", 80.31 * 0.999),
            "3d-unet-99": ("DICE", 0.86170 * 0.99),
            "3d-unet-99.9": ("DICE", 0.86170 * 0.999),

            "gptj-99": (
                "ROUGE1",
                42.9865 * 0.99,
                "ROUGE2",
                20.1235 * 0.99,
                "ROUGEL",
                29.9881 * 0.99,
                "GEN_LEN",
                4016878 * 0.9,
            ),
            "gptj-99.9": (
                "ROUGE1",
                42.9865 * 0.999,
                "ROUGE2",
                20.1235 * 0.999,
                "ROUGEL",
                29.9881 * 0.999,
                "GEN_LEN",
                4016878 * 0.9,
            ),
            "llama2-70b-99": (
                "ROUGE1",
                44.4312 * 0.99,
                "ROUGE2",
                22.0352 * 0.99,
                "ROUGEL",
                28.6162 * 0.99,
                "TOKENS_PER_SAMPLE",
                294.45 * 0.9,
            ),
            "llama2-70b-99.9": (
                "ROUGE1",
                44.4312 * 0.999,
                "ROUGE2",
                22.0352 * 0.999,
                "ROUGEL",
                28.6162 * 0.999,
                "TOKENS_PER_SAMPLE",
                294.45 * 0.9,
            ),
            "llama2-70b-interactive-99": (
                "ROUGE1",
                44.4312 * 0.99,
                "ROUGE2",
                22.0352 * 0.99,
                "ROUGEL",
                28.6162 * 0.99,
                "TOKENS_PER_SAMPLE",
                294.45 * 0.9,
            ),
            "llama2-70b-interactive-99.9": (
                "ROUGE1",
                44.4312 * 0.999,
                "ROUGE2",
                22.0352 * 0.999,
                "ROUGEL",
                28.6162 * 0.999,
                "TOKENS_PER_SAMPLE",
                294.45 * 0.9,
            ),
            "stable-diffusion-xl": (
                "CLIP_SCORE",
                31.68631873,
                "FID_SCORE",
                23.01085758,
            ),
            "mixtral-8x7b": (
                "ROUGE1",
                45.5989 * 0.99,
                "ROUGE2",
                23.3526 * 0.99,
                "ROUGEL",
                30.4608 * 0.99,
                "TOKENS_PER_SAMPLE",
                144.84 * 0.9,
                "gsm8k_accuracy",
                73.66 * 0.99,
                "mbxp_accuracy",
                60.16 * 0.99,
            ),
            "llama3.1-405b": (
                "ROUGEL",
                21.6666 * 0.99,
                "exact_match",
                90.1335 * 0.99,
                "TOKENS_PER_SAMPLE",
                684.68 * 0.9,
            ),
            "rgat": ("acc", 0.7286 * 0.99),
            "pointpainting": ("mAP", 0.5425 * 0.999),
        },
        "accuracy-upper-limit": {
            "stable-diffusion-xl": (
                "CLIP_SCORE",
                31.81331801,
                "FID_SCORE",
                23.95007626,
            ),
            "llama2-70b-99": ("TOKENS_PER_SAMPLE", 294.45 * 1.1),
            "llama2-70b-99.9": ("TOKENS_PER_SAMPLE", 294.45 * 1.1),
            "llama2-70b-interactive-99": ("TOKENS_PER_SAMPLE", 294.45 * 1.1),
            "llama2-70b-interactive-99.9": ("TOKENS_PER_SAMPLE", 294.45 * 1.1),
            "mixtral-8x7b": ("TOKENS_PER_SAMPLE", 145.9 * 1.1),
            "llama3.1-405b": ("TOKENS_PER_SAMPLE", 684.68 * 1.1),
        },
        "accuracy-delta-perc": {
            "stable-diffusion-xl": {"CLIP_SCORE": 1, "FID_SCORE": 2}
        },
        "performance-sample-count": {
            "resnet": 1024,
            "retinanet": 64,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-v2-99": 204800,
            "dlrm-v2-99.9": 204800,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
            "gptj-99": 13368,
            "gptj-99.9": 13368,
            "llama2-70b-99": 24576,
            "llama2-70b-99.9": 24576,
            "llama2-70b-interactive-99": 24576,
            "llama2-70b-interactive-99.9": 24576,
            "stable-diffusion-xl": 5000,
            "mixtral-8x7b": 15000,
            "llama3.1-405b": 8313,
            "rgat": 788379,
            "pointpainting": 1024,
        },
        "accuracy-sample-count": {},
        "dataset-size": {
            "resnet": 50000,
            "retinanet": 24781,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-v2-99": 330067,
            "dlrm-v2-99.9": 330067,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
            "gptj-99": 13368,
            "gptj-99.9": 13368,
            "llama2-70b-99": 24576,
            "llama2-70b-99.9": 24576,
            "llama2-70b-interactive-99": 24576,
            "llama2-70b-interactive-99.9": 24576,
            "stable-diffusion-xl": 5000,
            "mixtral-8x7b": 15000,
            "llama3.1-405b": 8313,
            "rgat": 788379,
            "pointpainting": 39987,
        },
        # model_mapping.json is expected in the root directory of the
        # submission folder for open submissions and so the below dictionary is
        # not really needed
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-resnet34": "retinanet",
            "mobilenet": "resnet",
            "resnet50": "resnet",
            "llama3_1-405b": "llama3.1-405b",
        },
        "seeds": {
            # TODO: Update random seeds
            "qsl_rng_seed": 6023615788873153749,
            "sample_index_rng_seed": 15036839855038426416,
            "schedule_rng_seed": 9933818062894767841,
        },
        "ignore_errors": [],
        "latency-constraint": {
            "resnet": {"Server": 15000000},
            "retinanet": {"Server": 100000000},
            "dlrm-v2-99": {"Server": 60000000},
            "dlrm-v2-99.9": {"Server": 60000000},
            "gptj-99": {"Server": 20000000000},
            "gptj-99.9": {"Server": 20000000000},
            "stable-diffusion-xl": {"Server": 20000000000},
            "llama2-70b-99": {"Server": 20000000000},
            "llama2-70b-99.9": {"Server": 20000000000},
            "llama2-70b-interactive-99": {"Server": 20000000000},
            "llama2-70b-interactive-99.9": {"Server": 20000000000},
            "mixtral-8x7b": {"Server": 20000000000},
            "llama3.1-405b": {"Server": 60000000000}
        },
        "min-queries": {
            "resnet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "retinanet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "bert-99": {"SingleStream": 1024, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Offline": 1},
            "dlrm-v2-99": {"Server": 270336, "Offline": 1},
            "dlrm-v2-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
            "gptj-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "gptj-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama2-70b-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama2-70b-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama2-70b-interactive-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama2-70b-interactive-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "stable-diffusion-xl": {
                "SingleStream": 1024,
                "Server": 270336,
                "Offline": 1,
            },
            "mixtral-8x7b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama3.1-405b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "rgat": {"SingleStream": 1024, "Offline": 1},
            "pointpainting": {"SingleStream": 1024},
        },
        "models_TEST01": [
            "resnet",
            "retinanet",
            "bert-99",
            "bert-99.9",
            "dlrm-v2-99",
            "dlrm-v2-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
            "stable-diffusion-xl",
            "rgat",
            "pointpainting",
        ],
        "models_TEST04": [
            "resnet",
            "stable-diffusion-xl",
            "pointpainting",
        ],
        "models_TEST06": [
            "llama2-70b-99",
            "llama2-70b-99.9",
            "llama2-70b-interactive-99",
            "llama2-70b-interactive-99.9",
            "gptj-99",
            "gptj-99.9",
            "mixtral-8x7b",
        ],
    },
    "v5.1": {
        "models": [
            "resnet",
            "retinanet",
            "bert-99",
            "bert-99.9",
            "dlrm-v2-99",
            "dlrm-v2-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
            "llama3.1-8b",
            "llama3.1-8b-edge",
            "llama2-70b-99",
            "llama2-70b-99.9",
            "stable-diffusion-xl",
            "mixtral-8x7b",
            "llama3.1-405b",
            "rgat",
            "pointpainting",
            "deepseek-r1",
            "whisper",
        ],
        "required-scenarios-datacenter": {
            "retinanet": ["Server", "Offline"],
            "dlrm-v2-99": ["Server", "Offline"],
            "dlrm-v2-99.9": ["Server", "Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
            "llama3.1-8b": ["Offline"],
            "llama2-70b-99": ["Offline"],
            "llama2-70b-99.9": ["Offline"],
            "stable-diffusion-xl": ["Server", "Offline"],
            "mixtral-8x7b": ["Server", "Offline"],
            "llama3.1-405b": ["Offline"],
            "rgat": ["Offline"],
            "deepseek-r1": ["Server", "Offline"],
            "whisper": ["Offline"],
        },
        "optional-scenarios-datacenter": {
            "llama2-70b-99": ["Interactive", "Server"],
            "llama2-70b-99.9": ["Interactive", "Server"],
            "llama3.1-405b": ["Interactive", "Server"],
            "llama3.1-8b": ["Interactive", "Server"],
        },
        "required-scenarios-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline"],
            "retinanet": ["SingleStream", "MultiStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
            "llama3.1-8b-edge": ["SingleStream", "Offline"],
            "stable-diffusion-xl": ["SingleStream", "Offline"],
            "pointpainting": ["SingleStream"],
            "whisper": ["Offline"],
        },
        "optional-scenarios-edge": {},
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline", "Server"],
            "retinanet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["SingleStream", "Offline"],
            "dlrm-v2-99": ["Offline", "Server"],
            "dlrm-v2-99.9": ["Offline", "Server"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
            "llama3.1-8b": ["Offline"],
            "llama3.1-8b-edge": ["SingleStream", "Offline"],
            "llama2-70b-99": ["Offline"],
            "llama2-70b-99.9": ["Offline"],
            "stable-diffusion-xl": ["SingleStream", "Offline", "Server"],
            "mixtral-8x7b": ["Server", "Offline"],
            "llama3.1-405b": ["Offline"],
            "rgat": ["Offline"],
            "pointpainting": ["SingleStream"],
            "deepseek-r1": ["SingleStream", "Server", "Offline"],
            "whisper": ["Offline"],
        },
        "optional-scenarios-datacenter-edge": {
            "llama2-70b-99": ["Interactive", "Server"],
            "llama2-70b-99.9": ["Interactive", "Server"],
            "llama3.1-405b": ["Interactive", "Server"],
            "llama3.1-8b": ["Interactive", "Server"],
        },
        "accuracy-target": {
            "resnet": ("acc", 76.46 * 0.99),
            "retinanet": ("mAP", 37.55 * 0.99),
            "bert-99": ("F1", 90.874 * 0.99),
            "bert-99.9": ("F1", 90.874 * 0.999),
            "dlrm-v2-99": ("AUC", 80.31 * 0.99),
            "dlrm-v2-99.9": ("AUC", 80.31 * 0.999),
            "3d-unet-99": ("DICE", 0.86170 * 0.99),
            "3d-unet-99.9": ("DICE", 0.86170 * 0.999),

            "llama3.1-8b": (
                "ROUGE1",
                38.7792 * 0.99,
                "ROUGE2",
                15.9075 * 0.99,
                "ROUGEL",
                24.4957 * 0.99,
                "ROUGELSUM",
                35.793 * 0.99,
                "GEN_LEN",
                8167644 * 0.9,
            ),
            "llama3.1-8b-edge": (
                "ROUGE1",
                39.06 * 0.99,
                "ROUGE2",
                16.1147 * 0.99,
                "ROUGEL",
                24.6375 * 0.99,
                "ROUGELSUM",
                36.124 * 0.99,
                "GEN_LEN",
                3051113 * 0.9,
            ),
            "llama2-70b-99": (
                "ROUGE1",
                44.4312 * 0.99,
                "ROUGE2",
                22.0352 * 0.99,
                "ROUGEL",
                28.6162 * 0.99,
                "TOKENS_PER_SAMPLE",
                294.45 * 0.9,
            ),
            "llama2-70b-99.9": (
                "ROUGE1",
                44.4312 * 0.999,
                "ROUGE2",
                22.0352 * 0.999,
                "ROUGEL",
                28.6162 * 0.999,
                "TOKENS_PER_SAMPLE",
                294.45 * 0.9,
            ),
            "stable-diffusion-xl": (
                "CLIP_SCORE",
                31.68631873,
                "FID_SCORE",
                23.01085758,
            ),
            "mixtral-8x7b": (
                "ROUGE1",
                45.5989 * 0.99,
                "ROUGE2",
                23.3526 * 0.99,
                "ROUGEL",
                30.4608 * 0.99,
                "TOKENS_PER_SAMPLE",
                144.84 * 0.9,
                "gsm8k_accuracy",
                73.66 * 0.99,
                "mbxp_accuracy",
                60.16 * 0.99,
            ),
            "llama3.1-405b": (
                "ROUGEL",
                21.6666 * 0.99,
                "exact_match",
                90.1335 * 0.99,
                "TOKENS_PER_SAMPLE",
                684.68 * 0.9,
            ),
            "rgat": ("acc", 0.7286 * 0.99),
            "pointpainting": ("mAP", 0.5425 * 0.999),
            "deepseek-r1": ("exact_match", 0.99 * 81.3582, "TOKENS_PER_SAMPLE", 0.9 * 3886.2274),
            "whisper": ("ACCURACY", (100.0 - 2.0671) * 0.99),
        },
        "accuracy-upper-limit": {
            "stable-diffusion-xl": (
                "CLIP_SCORE",
                31.81331801,
                "FID_SCORE",
                23.95007626,
            ),
            "llama2-70b-99": ("TOKENS_PER_SAMPLE", 294.45 * 1.1),
            "llama2-70b-99.9": ("TOKENS_PER_SAMPLE", 294.45 * 1.1),
            "mixtral-8x7b": ("TOKENS_PER_SAMPLE", 145.9 * 1.1),
            "llama3.1-405b": ("TOKENS_PER_SAMPLE", 684.68 * 1.1),
            "llama3.1-8b": ("GEN_LEN", 8167644 * 1.1),
            "llama3.1-8b-edge": ("GEN_LEN", 3051113 * 1.1),
            "deepseek-r1": ("TOKENS_PER_SAMPLE", 1.1 * 3886.2274)
        },
        "accuracy-delta-perc": {
            "stable-diffusion-xl": {"CLIP_SCORE": 1, "FID_SCORE": 2}
        },
        "performance-sample-count": {
            "resnet": 1024,
            "retinanet": 64,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-v2-99": 204800,
            "dlrm-v2-99.9": 204800,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
            "llama3.1-8b": 13368,
            "llama3.1-8b-edge": 5000,
            "llama2-70b-99": 24576,
            "llama2-70b-99.9": 24576,
            "stable-diffusion-xl": 5000,
            "mixtral-8x7b": 15000,
            "llama3.1-405b": 8313,
            "rgat": 788379,
            "pointpainting": 1024,
            "deepseek-r1": 4388,
            "whisper": 1633,
        },
        "accuracy-sample-count": {},
        "dataset-size": {
            "resnet": 50000,
            "retinanet": 24781,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-v2-99": 330067,
            "dlrm-v2-99.9": 330067,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
            "llama3.1-8b": 13368,
            "llama3.1-8b-edge": 5000,
            "llama2-70b-99": 24576,
            "llama2-70b-99.9": 24576,
            "stable-diffusion-xl": 5000,
            "mixtral-8x7b": 15000,
            "llama3.1-405b": 8313,
            "rgat": 788379,
            "pointpainting": 39987,
            "deepseek-r1": 4388,
            "whisper": 1633,
        },
        # model_mapping.json is expected in the root directory of the
        # submission folder for open submissions and so the below dictionary is
        # not really needed
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-resnet34": "retinanet",
            "mobilenet": "resnet",
            "resnet50": "resnet",
            "llama3_1-405b": "llama3.1-405b",
            "llama3_1-8b": "llama3.1-8b",
            "llama3_1-8b-edge": "llama3.1-8b-edge",
        },
        "seeds": {
            # TODO: Update random seeds
            "qsl_rng_seed": 1780908523862526354,
            "sample_index_rng_seed": 14771362308971278857,
            "schedule_rng_seed": 18209322760996052031,
        },
        "ignore_errors": [],
        "latency-constraint": {
            "resnet": {"Server": 15000000},
            "retinanet": {"Server": 100000000},
            "dlrm-v2-99": {"Server": 60000000},
            "dlrm-v2-99.9": {"Server": 60000000},
            "llama3.1-8b": {"Server": 20000000000},
            "stable-diffusion-xl": {"Server": 20000000000},
            "llama2-70b-99": {"Server": 20000000000},
            "llama2-70b-99.9": {"Server": 20000000000},
            "mixtral-8x7b": {"Server": 20000000000},
            "llama3.1-405b": {"Server": 60000000000},
            "deepseek-r1": {"Server": 60000000000},
        },
        "min-queries": {
            "resnet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "retinanet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "bert-99": {"SingleStream": 1024, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Offline": 1},
            "dlrm-v2-99": {"Server": 270336, "Offline": 1},
            "dlrm-v2-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
            "llama3.1-8b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama3.1-8b-edge": {"SingleStream": 1024, "Offline": 1},
            "llama2-70b-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama2-70b-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "stable-diffusion-xl": {
                "SingleStream": 1024,
                "Server": 270336,
                "Offline": 1,
            },
            "mixtral-8x7b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "llama3.1-405b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "rgat": {"SingleStream": 1024, "Offline": 1},
            "pointpainting": {"SingleStream": 1024},
            "deepseek-r1": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "whisper": {"SingleStream": 1024, "Offline": 1},
        },
        "models_TEST01": [
            "resnet",
            "retinanet",
            "bert-99",
            "bert-99.9",
            "dlrm-v2-99",
            "dlrm-v2-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
            "stable-diffusion-xl",
            "rgat",
            "pointpainting",
            "whisper",
        ],
        "models_TEST04": [
            "resnet",
            "stable-diffusion-xl",
            "pointpainting",
        ],
        "models_TEST06": [
            "llama2-70b-99",
            "llama2-70b-99.9",
            "llama2-70b-interactive-99",
            "llama2-70b-interactive-99.9",
            "llama3.1-405b",
            "llama3.1-8b",
            "llama3.1-8b-interactive",
            "llama3.1-405b-interactive",
            "mixtral-8x7b",
            "deepseek-r1",
        ],
    },
}

VALID_DIVISIONS = ["open", "closed", "network"]
VALID_AVAILABILITIES = ["available", "preview", "rdi"]
REQUIRED_PERF_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
OPTIONAL_PERF_FILES = ["mlperf_log_accuracy.json"]
REQUIRED_PERF_POWER_FILES = ["spl.txt"]
REQUIRED_POWER_FILES = [
    "client.json",
    "client.log",
    "ptd_logs.txt",
    "server.json",
    "server.log",
]
REQUIRED_ACC_FILES = [
    "mlperf_log_summary.txt",
    "mlperf_log_detail.txt",
    "accuracy.txt",
    "mlperf_log_accuracy.json",
]
REQUIRED_ACC_BENCHMARK = {
    "stable-diffusion-xl": {
        "v5.0": {
            "images": [
                "4459",
                "4015",
                "2705",
                "1682",
                "4048",
                "4683",
                "3757",
                "1578",
                "3319",
                "95",
            ]
        },
        "v5.1": {
            "images": [
                "2747",
                "2235",
                "2165",
                "1515",
                "1538",
                "1367",
                "2419",
                "4629",
                "3657",
                "4532",
            ]
        },
        "v6.0": {
            "images": [
                "1311",
                "2476",
                "3644",
                "2188",
                "4114",
                "52",
                "388",
                "1195",
                "3427",
                "2289",
            ]
        },
    }
}
REQUIRED_MEASURE_FILES = ["user.conf", "README.md"]
REQUIRED_POWER_MEASURE_FILES = ["analyzer_table.*", "power_settings.*"]
MS_TO_NS = 1000 * 1000
S_TO_MS = 1000
FILE_SIZE_LIMIT_MB = 50
MB_TO_BYTES = 1024 * 1024
MAX_ACCURACY_LOG_SIZE = 10 * 1024
OFFLINE_MIN_SPQ = 24576
TEST_DURATION_MS_PRE_1_0 = 60000
TEST_DURATION_MS = 600000
REQUIRED_COMP_PER_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
REQUIRED_TEST01_ACC_FILES_1 = ["mlperf_log_accuracy.json", "accuracy.txt"]
REQUIRED_TEST01_ACC_FILES = REQUIRED_TEST01_ACC_FILES_1 + [
    "baseline_accuracy.txt",
    "compliance_accuracy.txt",
]

OFFLINE_MIN_SPQ_SINCE_V4 = {
    "resnet": 24576,
    "retinanet": 24576,
    "bert-99": 10833,
    "bert-99.9": 10833,
    "dlrm-v2-99": 24576,
    "dlrm-v2-99.9": 24576,
    "3d-unet-99": 43,
    "3d-unet-99.9": 43,
    "rnnt": 2513,
    "llama3.1-8b": 13368,
    "llama3.1-8b-edge": 5000,
    "llama2-70b-99": 24576,
    "llama2-70b-99.9": 24576,
    "llama2-70b-interactive-99": 24576,
    "llama2-70b-interactive-99.9": 24576,
    "stable-diffusion-xl": 5000,
    "mixtral-8x7b": 15000,
    "llama3.1-405b": 8313,
    "rgat": 788379,
    "deepseek-r1": 4388,
    "gpt-oss-120b": 6396,
    "whisper": 1633,
    "pointpainting": 6636,
    "yolo-99": 1525,
    "yolo-95": 1525,
    "dlrm-v3": 349823,
    "qwen3-vl-235b-a22b": 48289,
    "wan-2.2-t2v-a14b": 50,
}

SCENARIO_MAPPING = {
    "singlestream": "SingleStream",
    "multistream": "MultiStream",
    "server": "Server",
    "offline": "Offline",
    "interactive": "Interactive",
}

RESULT_FIELD = {
    "Offline": "Samples per second",
    "SingleStream": "90th percentile latency (ns)",
    "MultiStream": "Samples per query",
    "Server": "Scheduled samples per second",
}

RESULT_FIELD_NEW = {
    "v5.0": {
        "Offline": "result_samples_per_second",
        "SingleStream": "early_stopping_latency_ss",
        "MultiStream": "early_stopping_latency_ms",
        "Server": "result_completed_samples_per_sec",
    },
    "v5.1": {
        "Offline": "result_samples_per_second",
        "SingleStream": "early_stopping_latency_ss",
        "MultiStream": "early_stopping_latency_ms",
        "Server": "result_completed_samples_per_sec",
    },
    "v6.0": {
        "Offline": "result_samples_per_second",
        "SingleStream": "early_stopping_latency_ss",
        "MultiStream": "early_stopping_latency_ms",
        "Server": "result_completed_samples_per_sec",
    },
}

RESULT_FIELD_BENCHMARK_OVERWRITE = {
    "v5.0": {
        "llama2-70b-99": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama2-70b-99.9": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama2-70b-interactive-99": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama2-70b-interactive-99.9": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "gptj-99": {
            "Offline": "result_inferred_tokens_per_second",
            "Server": "result_inferred_completed_tokens_per_second",
        },
        "gptj-99.9": {
            "Offline": "result_inferred_tokens_per_second",
            "Server": "result_inferred_completed_tokens_per_second",
        },
        "mixtral-8x7b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama3.1-405b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
    },
    "v5.1": {
        "llama2-70b-99": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama2-70b-99.9": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama3.1-8b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama3.1-8b-edge": {
            "Offline": "result_tokens_per_second",
            "SingleStream": "result_90.00_percentile_latency_ns",
        },
        "mixtral-8x7b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama3.1-405b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "deepseek-r1": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "whisper": {
            "Offline": "result_tokens_per_second",
        }
    },
    "v6.0": {
        "gpt-oss-120b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama2-70b-99": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama2-70b-99.9": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama3.1-8b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama3.1-8b-edge": {
            "Offline": "result_tokens_per_second",
            "SingleStream": "result_90.00_percentile_latency_ns",
        },
        "mixtral-8x7b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "llama3.1-405b": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "deepseek-r1": {
            "Offline": "result_tokens_per_second",
            "Server": "result_completed_tokens_per_second",
        },
        "whisper": {
            "Offline": "result_tokens_per_second",
        }
    },
}

LLM_LATENCY_LIMITS = {
    "llama2-70b-99": {
        "Server": {
            "ttft": 2000 * 1000000, "tpot": 200 * 1000000
        },
        "Interactive": {
            "ttft": 450 * 1000000, "tpot": 40 * 1000000
        },
    },
    "llama2-70b-99.9": {
        "Server": {
            "ttft": 2000 * 1000000, "tpot": 200 * 1000000
        },
        "Interactive": {
            "ttft": 450 * 1000000, "tpot": 40 * 1000000
        },
    },
    "llama2-70b-interactive-99": {
        "Server": {
            "ttft": 450 * 1000000, "tpot": 40 * 1000000
        },
    },
    # for v5.0
    "llama2-70b-interactive-99.9": {
        "Server": {
            "ttft": 450 * 1000000, "tpot": 40 * 1000000
        },
    },
    "mixtral-8x7b": {
        "Server": {
            "ttft": 2000 * 1000000, "tpot": 200 * 1000000
        }
    },
    "llama3.1-405b": {
        "Server": {
            "ttft": 6000 * 1000000, "tpot": 175 * 1000000
        },
        "Interactive": {
            "ttft": 4500 * 1000000, "tpot": 80 * 1000000
        },
    },
    "llama3.1-8b": {
        "Server": {
            "ttft": 2000 * 1000000, "tpot": 100 * 1000000
        },
        "Interactive": {
            "ttft": 500 * 1000000, "tpot": 30 * 1000000
        }
    },
    "deepseek-r1": {
        "Server": {
            "ttft": 2000 * 1000000, "tpot": 80 * 1000000
        },
        "Interactive": {
            "ttft": 1500 * 1000000, "tpot": 15 * 1000000
        }
    },
    "gpt-oss-120b": {
        "Server": {
            "ttft": 3000 * 1000000, "tpot": 80 * 1000000
        },
        "Interactive": {
            "ttft": 2000 * 1000000, "tpot": 15 * 1000000
        }
    }

}

ACC_PATTERN = {
    "acc": r"^(?:\{\"accuracy|accuracy)[\": ]*=?\s*([\d\.]+).*",
    "meanAcc": r".*'mean-accuracy':\s+'?([\d.]+)'?.*",
    "AUC": r"^AUC=([\d\.]+).*",
    # dlrm-v3 patterns for parsing metric/lifetime_*/rating format
    "DLRM_NE": r".*metric/lifetime_ne/rating:\s*([\d\.]+).*",
    "DLRM_ACC": r".*metric/lifetime_accuracy/rating:\s*([\d\.]+).*",
    "DLRM_AUC": r".*metric/lifetime_gauc/rating:\s*([\d\.]+).*",
    "mAP": r".*(?:mAP=|'Total':)\s*([\d.]+)",
    "bleu": r"^BLEU\:\s*([\d\.]+).*",
    "F1": r"^{[\"\']exact_match[\"\']\:\s*[\d\.]+,\s*[\"\']f1[\"\']\:\s*([\d\.]+)}",
    "ACCURACY": r"Word Error Rate\:.*, accuracy=([0-9\.]+)%",
    "DICE": r"Accuracy\:\s*mean\s*=\s*([\d\.]+).*",
    "ROUGE1": r".*'rouge1':\s+'?([\d.]+)'?.*",
    "ROUGE2": r".*'rouge2':\s+'?([\d.]+)'?.*",
    "ROUGEL": r".*'rougeL':\s+'?([\d.]+)'?.*",
    "ROUGELSUM": r".*'rougeLsum':\s+'?([\d.]+)'?.*",
    "GEN_LEN": r".*'gen_len':\s([\d.]+).*",
    "TOKENS_PER_SAMPLE": r".*'tokens_per_sample':\s([\d.]+).*",
    "CLIP_SCORE": r".*'CLIP_SCORE':\s+'?([\d.]+).*",
    "FID_SCORE": r".*'FID_SCORE':\s+'?([\d.]+).*",
    "gsm8k_accuracy": r".*'gsm8k':\s([\d.]+).*",
    "mbxp_accuracy": r".*'mbxp':\s([\d.]+).*",
    "exact_match": r".*'exact_match':\s([\d.]+).*",
    "vbench_score": r".*'vbench_score':\s([\d.]+).*",
    "F1_HIERARCHICAL": r'\{.*"f1":\s*([\d\.]+).*\}',
}

SYSTEM_DESC_REQUIRED_FIELDS = [
    "division",
    "submitter",
    "status",
    "system_name",
    "number_of_nodes",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_storage_capacity",
    "host_storage_type",
    "accelerators_per_node",
    "accelerator_model_name",
    "accelerator_memory_capacity",
    "framework",
    "operating_system",
    "system_type",
    "other_software_stack",
    "host_processor_frequency",
    "host_processor_caches",
    "host_memory_configuration",
    "host_processor_interconnect",
    "host_networking",
    "host_networking_topology",
    "accelerator_frequency",
    "accelerator_host_interconnect",
    "accelerator_interconnect",
    "accelerator_interconnect_topology",
    "accelerator_memory_configuration",
    "accelerator_on-chip_memories",
    "cooling",
    "hw_notes",
    "sw_notes",
    "host_network_card_count",
    "system_type_detail",
    # "network_speed_mbit",
]

SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS = [
    "division",
    "submitter",
    "system_type",
    "status",
    "system_name",
    "number_of_nodes",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_memory_configuration",
    "host_storage_capacity",
    "host_storage_type",
    "host_networking",
    "host_network_card_count",
    "host_networking_topology",
    "accelerators_per_node",
    "accelerator_model_name",
    "accelerator_memory_capacity",
    "accelerator_host_interconnect",
    "accelerator_memory_configuration",
    "accelerator_interconnect",
    "cooling",
    "framework",
    "operating_system",
    "other_software_stack",
]

SYSTEM_DESC_NUMERIC_RESPONSE_REQUIRED_FIELDS = [
    # "network_speed_mbit"
]


SYSTEM_DESC_REQUIRED_FIELDS_POWER = [
    "power_management",
    "filesystem",
    "boot_firmware_version",
    "management_firmware_version",
    "other_hardware",
    "number_of_type_nics_installed",
    "nics_enabled_firmware",
    "nics_enabled_os",
    "nics_enabled_connected",
    "network_speed_mbit",
    "power_supply_quantity_and_rating_watts",
    "power_supply_details",
    "disk_drives",
    "disk_controllers",
    "system_power_only",
]

SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS_POWER = []

SYSTEM_DESC_IS_NETWORK_MODE = "is_network"
SYSTEM_DESC_REQUIRED_FIELDS_NETWORK_MODE = [
    SYSTEM_DESC_IS_NETWORK_MODE,
    "network_type",
    "network_media",
    "network_rate",
    "nic_loadgen",
    "number_nic_loadgen",
    "net_software_stack_loadgen",
    "network_protocol",
    "number_connections",
    "nic_sut",
    "number_nic_sut",
    "net_software_stack_sut",
    "network_topology",
]
NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME = "Network SUT"

SYSTEM_IMP_REQUIRED_FILES = [
    "input_data_types",
    "retraining",
    "starting_weights_filename",
    "weight_data_types",
    "weight_transformations",
]

SPECIAL_UNIT_DICT = {
    "llama3.1-8b": {
        "Offline": "Tokens/s",
        "Server": "Tokens/s",
    },
    "llama3.1-8b-edge": {
        "Offline": "Tokens/s",
    },
    "llama2-70b-99": {
        "Offline": "Tokens/s",
        "Server": "Tokens/s",
        "Interactive": "Tokens/s",
    },
    "llama2-70b-99.9": {
        "Offline": "Tokens/s",
        "Server": "Tokens/s",
        "Interactive": "Tokens/s",
    },
    "mixtral-8x7b": {
        "Offline": "Tokens/s",
        "Server": "Tokens/s",
        "Interactive": "Tokens/s",
    },
    "llama3.1-405b": {
        "Offline": "Tokens/s",
        "Server": "Tokens/s",
        "Interactive": "Tokens/s",
    },
    "deepseek-r1": {
        "Offline": "Tokens/s",
        "Server": "Tokens/s",
        "Interactive": "Tokens/s",
    },
}
UNIT_DICT = {
    "SingleStream": "Latency (ms)",
    "MultiStream": "Latency (ms)",
    "Offline": "Samples/s",
    "Server": "Queries/s",
    "Interactive": "Queries/s",

    "singlestream": "Latency (ms)",
    "multistream": "Latency (ms)",
    "offline": "Samples/s",
    "server": "Queries/s",
    "interactive": "Queries/s",
}
POWER_UNIT_DICT = {
    "SingleStream": "millijoules",
    "MultiStream": "millijoules",
    "Offline": "Watts",
    "Server": "Watts",
    "Interactive": "Watts",

    "singlestream": "millijoules",
    "multistream": "millijoules",
    "offline": "Watts",
    "server": "Watts",
    "interactive": "Watts",
}


PERFORMANCE_LOG_PATH = {
    "v5.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/mlperf_log_detail.txt",
    "v5.1": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/mlperf_log_detail.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/mlperf_log_detail.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/mlperf_log_detail.txt",
}

PERFORMANCE_SUMMARY_PATH = {
    "v5.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/mlperf_log_summary.txt",
    "v5.1": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/mlperf_log_summary.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/mlperf_log_summary.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/mlperf_log_summary.txt",
}

ACCURACY_LOG_PATH = {
    "v5.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/mlperf_log_detail.txt",
    "v5.1": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/mlperf_log_detail.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/mlperf_log_detail.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/mlperf_log_detail.txt",
}

ACCURACY_RESULT_PATH = {
    "v5.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/accuracy.txt",
    "v5.1": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/accuracy.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/accuracy.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/accuracy.txt",
}

ACCURACY_JSON_PATH = {
    "v5.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/mlperf_log_accuracy.json",
    "v5.1": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/mlperf_log_accuracy.json",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/mlperf_log_accuracy.json",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/mlperf_log_accuracy.json",
}

POWER_DIR_PATH = {
    "v5.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/power",
    "v5.1": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/power",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/power",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/power",
}

MEASUREMENTS_PATH = {
    "v5.0": "{division}/{submitter}/measurements/{system}/{benchmark}/{scenario}/{file}",
    "v5.1": "{division}/{submitter}/measurements/{system}/{benchmark}/{scenario}/{file}",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/measurements.json",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/measurements.json",
}

TEST01_PERF_PATH = {
    "v5.0": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST01/performance/run_1/mlperf_log_detail.txt",
    "v5.1": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST01/performance/run_1/mlperf_log_detail.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST01/performance/run_1/mlperf_log_detail.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST01/performance/run_1/mlperf_log_detail.txt",
}

TEST01_ACC_PATH = {
    "v5.0": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST01/verify_accuracy.txt",
    "v5.1": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST01/verify_accuracy.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST01/verify_accuracy.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST01/verify_accuracy.txt",
}

TEST04_PERF_PATH = {
    "v5.0": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST04/performance/run_1/mlperf_log_detail.txt",
    "v5.1": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST04/performance/run_1/mlperf_log_detail.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST04/performance/run_1/mlperf_log_detail.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST04/performance/run_1/mlperf_log_detail.txt",
}

TEST04_ACC_PATH = {
    "v5.0": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST04/verify_accuracy.txt",
    "v5.1": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST04/verify_accuracy.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST04/verify_accuracy.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST04/verify_accuracy.txt",
}

TEST06_ACC_PATH = {
    "v5.0": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST06/verify_accuracy.txt",
    "v5.1": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/TEST06/verify_accuracy.txt",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST06/verify_accuracy.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST06/verify_accuracy.txt",
}

TEST07_ACC_PATH = {
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST07/verify_accuracy.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST07/verify_accuracy.txt",
}

TEST09_ACC_PATH = {
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST09/verify_output_len.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST09/verify_output_len.txt",
}

TEST08_ACC_PATH = {
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST08/verify_accuracy.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST08/verify_accuracy.txt",
}
TEST07_ACC_PATH = {
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST07/verify_accuracy.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST07/verify_accuracy.txt",
}

TEST09_ACC_PATH = {
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST09/verify_output_len.txt",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/TEST09/verify_output_len.txt",
}

COMPLIANCE_PATH = {
    "v5.0": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/",
    "v5.1": "{division}/{submitter}/compliance/{system}/{benchmark}/{scenario}/",
    "v6.0": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/",
    "default": "{division}/{submitter}/results/{system}/{benchmark}/{scenario}/",
}

SYSTEM_PATH = {
    "v5.0": "{division}/{submitter}/systems/{system}.json",
    "v5.1": "{division}/{submitter}/systems/{system}.json",
    "v6.0": "{division}/{submitter}/systems/{system}.json",
    "default": "{division}/{submitter}/systems/{system}.json",
}

SRC_PATH = {
    "v5.0": "{division}/{submitter}/code",
    "v5.1": "{division}/{submitter}/code",
    "v6.0": "{division}/{submitter}/src",
    "default": "{division}/{submitter}/src",
}
