import pytest
from submission_checker.configuration.configuration import Config


@pytest.fixture()
def cfg():
    return Config(version="v6.0", extra_model_benchmark_map={})


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestConfigInit:
    def test_version_stored(self, cfg):
        assert cfg.version == "v6.0"

    def test_models_populated(self, cfg):
        assert "resnet" in cfg.models
        assert "bert-99" in cfg.models

    def test_unknown_version_raises(self):
        with pytest.raises((KeyError, TypeError)):
            Config(version="v99.99", extra_model_benchmark_map={})


# ---------------------------------------------------------------------------
# set_type
# ---------------------------------------------------------------------------

class TestSetType:
    def test_datacenter_sets_required(self, cfg):
        cfg.set_type("datacenter")
        assert cfg.required is not None

    def test_edge_sets_required(self, cfg):
        cfg.set_type("edge")
        assert cfg.required is not None

    def test_combined_accepted(self, cfg):
        cfg.set_type("datacenter,edge")
        assert cfg.required is not None

    def test_combined_reversed_accepted(self, cfg):
        cfg.set_type("edge,datacenter")
        assert cfg.required is not None

    def test_invalid_type_raises(self, cfg):
        with pytest.raises(ValueError, match="invalid system type"):
            cfg.set_type("cloud")


# ---------------------------------------------------------------------------
# get_mlperf_model
# ---------------------------------------------------------------------------

class TestGetMlperfModel:
    def test_official_name_passthrough(self, cfg):
        assert cfg.get_mlperf_model("resnet") == "resnet"

    def test_resnet50_maps_to_resnet(self, cfg):
        assert cfg.get_mlperf_model("resnet50") == "resnet"

    def test_mobilenet_maps_to_resnet(self, cfg):
        assert cfg.get_mlperf_model("mobilenet-v1") == "resnet"

    def test_bert_99_variant(self, cfg):
        assert cfg.get_mlperf_model("bert-99-large") == "bert-99"

    def test_extra_mapping_used(self, cfg):
        assert cfg.get_mlperf_model("my_resnet", {"my_resnet": "resnet"}) == "resnet"


# ---------------------------------------------------------------------------
# get_required / get_optional
# ---------------------------------------------------------------------------

class TestGetRequired:
    def test_resnet_edge_requires_three_scenarios(self, cfg):
        cfg.set_type("edge")
        req = cfg.get_required("resnet")
        assert req == {"SingleStream", "MultiStream", "Offline"}

    def test_unknown_model_returns_none(self, cfg):
        cfg.set_type("edge")
        assert cfg.get_required("nonexistent-model") is None

    def test_optional_empty_set_for_unknown(self, cfg):
        cfg.set_type("edge")
        assert cfg.get_optional("nonexistent-model") == set()


# ---------------------------------------------------------------------------
# get_accuracy_target
# ---------------------------------------------------------------------------

class TestGetAccuracyTarget:
    def test_resnet_accuracy_target(self, cfg):
        target = cfg.get_accuracy_target("resnet")
        assert target is not None
        assert target[0] == "acc"
        assert target[1] == pytest.approx(76.46 * 0.99)

    def test_unknown_model_raises(self, cfg):
        with pytest.raises(ValueError, match="model not known"):
            cfg.get_accuracy_target("not-a-model")


# ---------------------------------------------------------------------------
# get_delta_perc
# ---------------------------------------------------------------------------

class TestGetDeltaPerc:
    def test_standard_model_defaults_to_1(self, cfg):
        assert cfg.get_delta_perc("resnet", "acc") == 1

    def test_high_accuracy_model_defaults_to_0_1(self, cfg):
        assert cfg.get_delta_perc("bert-99.9", "F1") == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Boolean helpers
# ---------------------------------------------------------------------------

class TestBooleanHelpers:
    def test_uses_early_stopping_server(self, cfg):
        assert cfg.uses_early_stopping("Server") is True

    def test_uses_early_stopping_offline_false(self, cfg):
        assert cfg.uses_early_stopping("Offline") is False

    def test_has_new_logging_format(self, cfg):
        assert cfg.has_new_logging_format() is True


# ---------------------------------------------------------------------------
# get_llm_models
# ---------------------------------------------------------------------------

def test_llm_models_include_llama(cfg):
    llms = cfg.get_llm_models()
    assert any("llama" in m for m in llms)


# ---------------------------------------------------------------------------
# ignore_errors
# ---------------------------------------------------------------------------

def test_ignore_errors_matches_configured_string(cfg):
    # ignore_errors is driven by base["ignore_errors"]; we just verify it
    # doesn't crash and returns a bool
    result = cfg.ignore_errors("some random log line")
    assert isinstance(result, bool)
