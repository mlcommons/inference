"""CPU-only checks for the vLLM engine configuration used by the SUTs."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


class EngineConfigurationTest(unittest.TestCase):
    def setUp(self):
        # Model loading is the boundary under test; importing the reference SUT
        # should not require a GPU, LoadGen, or a downloaded model here.
        modules = {
            name: mock.MagicMock()
            for name in (
                "numpy", "torch", "torch.nn", "torch.nn.functional",
                "vllm", "vllm.inputs", "tqdm", "mlperf_loadgen", "dataset",
            )
        }
        path = Path(__file__).with_name("SUT_VLLM.py")
        spec = importlib.util.spec_from_file_location("sut_vllm_test", path)
        self.sut_module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.sut_module)

    def make_sut(self, cls):
        sut = cls.__new__(cls)
        sut.model_path = "test-model"
        sut.dtype = "bfloat16"
        sut.tensor_parallel_size = 2
        return sut

    def test_offline_disables_cross_query_prefix_caching(self):
        sut = self.make_sut(self.sut_module.SUT)
        sut.load_model()
        self.sut_module.LLM.assert_called_once_with(
            "test-model",
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_prefix_caching=False,
        )
        self.assertIs(sut.model, self.sut_module.LLM.return_value)

    def test_server_disables_cross_query_prefix_caching(self):
        sut = self.make_sut(self.sut_module.SUTServer)
        sut.load_model()
        self.sut_module.AsyncEngineArgs.assert_called_once_with(
            "test-model",
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_prefix_caching=False,
        )
        engine_factory = self.sut_module.AsyncLLMEngine.from_engine_args
        engine_factory.assert_called_once_with(
            self.sut_module.AsyncEngineArgs.return_value)
        self.assertIs(sut.model, engine_factory.return_value)


if __name__ == "__main__":
    unittest.main()
