#
# NTUHPC
# MLPerf
# SUT_API unit tests
#

from typing import Iterable
import pytest
from SUT_API import SUT

from unittest.mock import patch


@pytest.fixture
def sut() -> Iterable[SUT]:
    # Create a SUT instance with dummy parameters for testing
    with (
        patch("SUT_API.Dataset") as dataset,
        patch("SUT_API.AutoTokenizer") as tokenizer,
    ):
        yield SUT(
            model_path="model_path",
            dtype="float16",
            batch_size=1,
            dataset_path="dataset_path",
            total_sample_count=10,
            device="cpu",
            api_server=["http://server1", "http://server2"],
            api_model_name="dummy_model_name",
            workers=1,
        )


@pytest.mark.asyncio
async def test_query_batch(sut: SUT):
    for api_server in sut.api_servers:
        # mock the HTTP post method to return a dummy response from LLM API server
        with patch.object(sut._http, "post") as mock_post:
            mock_post.return_value.text = (
                '{"choices": [{"text": "Output 1"}, {"text": "Output 2"}]}'
            )

            prompt = [[1, 2]]
            outputs = await sut.query_batch(prompt, api_server)

            # only the first output is returned
            assert outputs == ["Output 1", "Output 2"]
            mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_query_api_servers(sut: SUT):
    test_inputs = [[1], [2], [3], [4]]

    with patch.object(sut, "query_batch") as mock_query_api_vllm:
        # compute expected batch size Distributing data over api_servers
        n_batch = len(test_inputs) // len(sut.api_servers)

        mock_query_api_vllm.return_value = ["Outputs"] * n_batch

        outputs = await sut.query_servers(test_inputs)

        assert outputs == ["Outputs"] * len(test_inputs)
        # expect 2 calls, one batch for each api_server
        assert mock_query_api_vllm.call_count == 2
