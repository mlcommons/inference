from ..constants import MODEL_CONFIG, ACC_PATTERN


class Config:
    """Select config value by mlperf version and submission type."""

    def __init__(
        self,
        version,
        extra_model_benchmark_map,
        ignore_uncommited=False,
        skip_compliance=False,
        skip_power_check=False,
        skip_meaningful_fields_emptiness_check=False,
        skip_check_power_measure_files=False,
        skip_empty_files_check=False,
        skip_extra_files_in_root_check=False,
        skip_extra_accuracy_files_check=False,
        skip_all_systems_have_results_check=False,
        skip_calibration_check=False,
        skip_dataset_size_check=False
    ):
        self.base = MODEL_CONFIG.get(version)
        self.extra_model_benchmark_map = extra_model_benchmark_map
        self.version = version
        self.ignore_uncommited = ignore_uncommited

        # Skip flags. All set to false for official submission
        self.skip_compliance = skip_compliance
        self.skip_power_check = skip_power_check
        self.skip_meaningful_fields_emptiness_check = skip_meaningful_fields_emptiness_check
        self.skip_check_power_measure_files = skip_check_power_measure_files
        self.skip_empty_files_check = skip_empty_files_check
        self.skip_extra_files_in_root_check = skip_extra_files_in_root_check
        self.skip_extra_accuracy_files_check = skip_extra_accuracy_files_check
        self.skip_all_systems_have_results_check = skip_all_systems_have_results_check
        self.skip_calibration_check = skip_calibration_check
        self.skip_dataset_size_check = skip_dataset_size_check
        self.load_config(version)

    def load_config(self, version):
        # TODO: Load values from
        self.models = self.base["models"]
        self.seeds = self.base["seeds"]
        if self.base.get("test05_seeds"):
            self.test05_seeds = self.base["test05_seeds"]
        self.accuracy_target = self.base["accuracy-target"]
        self.accuracy_delta_perc = self.base["accuracy-delta-perc"]
        self.accuracy_upper_limit = self.base.get("accuracy-upper-limit", {})
        self.performance_sample_count = self.base["performance-sample-count"]
        self.accuracy_sample_count = self.base["accuracy-sample-count"]
        self.dataset_size = self.base["dataset-size"]
        self.latency_constraint = self.base.get("latency-constraint", {})
        self.min_queries = self.base.get("min-queries", {})
        self.required = None
        self.optional = None

    def set_type(self, submission_type):
        if submission_type == "datacenter":
            self.required = self.base["required-scenarios-datacenter"]
            self.optional = self.base["optional-scenarios-datacenter"]
        elif submission_type == "edge":
            self.required = self.base["required-scenarios-edge"]
            self.optional = self.base["optional-scenarios-edge"]
        elif (
            submission_type == "datacenter,edge" or submission_type == "edge,datacenter"
        ):
            self.required = self.base["required-scenarios-datacenter-edge"]
            self.optional = self.base["optional-scenarios-datacenter-edge"]
        else:
            raise ValueError("invalid system type")

    def get_mlperf_model(self, model, extra_model_mapping=None):
        # preferred - user is already using the official name
        if model in self.models:
            return model

        # simple mapping, ie resnet50->resnet
        mlperf_model = self.base["model_mapping"].get(model)
        if mlperf_model:
            return mlperf_model

        # Custom mapping provided by the submitter
        if extra_model_mapping is not None:
            mlperf_model = extra_model_mapping.get(model)
            if mlperf_model:
                return mlperf_model

        # try to guess, keep this for backwards compatibility
        # TODO: Generalize this guess or remove it completely?

        if "mobilenet" in model or "efficientnet" in model or "resnet50" in model:
            model = "resnet"
        elif "bert-99.9" in model:
            model = "bert-99.9"
        elif "bert-99" in model:
            model = "bert-99"
        elif "llama3_1-405b" in model:
            model = "llama3.1-405b"
        # map again
        mlperf_model = self.base["model_mapping"].get(model, model)
        return mlperf_model

    def get_required(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.required:
            return None
        return set(self.required[model])

    def get_optional(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.optional:
            return set()
        return set(self.optional[model])

    def get_accuracy_target(self, model):
        if model not in self.accuracy_target:
            raise ValueError("model not known: " + model)
        return self.accuracy_target[model]

    def get_accuracy_upper_limit(self, model):
        return self.accuracy_upper_limit.get(model, None)

    def get_accuracy_values(self, model):
        patterns = []
        acc_targets = []
        acc_types = []
        acc_limits = []
        up_patterns = []
        acc_limit_check = False

        target = self.get_accuracy_target(model)
        acc_upper_limit = self.get_accuracy_upper_limit(model)
        if acc_upper_limit is not None:
            for i in range(0, len(acc_upper_limit), 2):
                acc_type, acc_target = acc_upper_limit[i: i + 2]
                acc_limits.append(acc_target)
                up_patterns.append(ACC_PATTERN[acc_type])

        for i in range(0, len(target), 2):
            acc_type, acc_target = target[i: i + 2]
            patterns.append(ACC_PATTERN[acc_type])
            acc_targets.append(acc_target)
            acc_types.append(acc_type)

        return patterns, acc_targets, acc_types, acc_limits, up_patterns, acc_upper_limit

    def get_performance_sample_count(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.performance_sample_count:
            raise ValueError("model not known: " + model)
        return self.performance_sample_count[model]

    def get_accuracy_sample_count(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.accuracy_sample_count:
            return self.get_dataset_size(model)
        return self.accuracy_sample_count[model]

    def ignore_errors(self, line):
        for error in self.base["ignore_errors"]:
            if error in line:
                return True
        if (
            self.ignore_uncommited
            and ("ERROR : Loadgen built with uncommitted " "changes!") in line
        ):
            return True
        return False

    def get_min_query_count(self, model, scenario):
        model = self.get_mlperf_model(model)
        if model not in self.min_queries:
            raise ValueError("model not known: " + model)
        return self.min_queries[model].get(scenario)

    def get_dataset_size(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.dataset_size:
            raise ValueError("model not known: " + model)
        return self.dataset_size[model]

    def get_delta_perc(self, model, metric):
        if model in self.accuracy_delta_perc:
            if metric in self.accuracy_delta_perc[model]:
                return self.accuracy_delta_perc[model][metric]

        more_accurate = model.find("99.9")
        if more_accurate == -1:
            required_delta_perc = 1
        else:
            required_delta_perc = 0.1
        return required_delta_perc

    def has_new_logging_format(self):
        return True

    def uses_early_stopping(self, scenario):
        return scenario in ["Server", "SingleStream", "MultiStream"]

    def requires_equal_issue(self, model, division):
        return (
            division in ["closed", "network"]
            and model
            in [
                "3d-unet-99",
                "3d-unet-99.9",
                "gptj-99",
                "gptj-99.9",
                "llama2-70b-99",
                "llama2-70b-99.9",
                "mixtral-8x7b",
            ]
            and self.version in ["v4.1"]
        )

    def get_llm_models(self):
        return [
            "llama2-70b-99",
            "llama2-70b-99.9",
            "llama2-70b-interactive-99",
            "llama2-70b-interactive-99.9",
            "mixtral-8x7b",
            "llama3.1-405b",
            "llama3.1-8b",
            "llama3.1-8b-edge",
            "deepseek-r1"
        ]
