from ..constants import MODEL_CONFIG


class Config:
    """Select config value by mlperf version and submission type."""

    def __init__(
        self,
        version,
        extra_model_benchmark_map,
        ignore_uncommited=False,
        skip_power_check=False,
    ):
        self.base = MODEL_CONFIG.get(version)
        self.extra_model_benchmark_map = extra_model_benchmark_map
        self.version = version
        self.ignore_uncommited = ignore_uncommited
        self.skip_power_check = skip_power_check
        self.load_config(version)
        

    def load_config(self, version):
        # TODO: Load values from 
        self.models = self.base["models"]
        self.seeds = self.base["seeds"]
        self.accuracy_target = self.base["accuracy-target"]
        self.accuracy_delta_perc = self.base["accuracy-delta-perc"]
        self.accuracy_upper_limit = self.base.get("accuracy-upper-limit", {})
        self.performance_sample_count = self.base["performance-sample-count"]
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

    def get_performance_sample_count(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.performance_sample_count:
            raise ValueError("model not known: " + model)
        return self.performance_sample_count[model]

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
