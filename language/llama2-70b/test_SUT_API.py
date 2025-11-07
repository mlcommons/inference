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
async def test_query(sut: SUT):

    for api_server in sut.api_servers:
        # mock the HTTP post method to return a dummy response from LLM API server
        with patch.object(sut._http, "post") as mock_post:
            mock_post.return_value.text = (
                '{"choices": [{"text": "Output 1"}, {"text": "Output 2"}]}'
            )

            prompt = [1, 2]
            output = await sut.query(prompt, api_server)

            # only the first output is returned
            assert output == "Output 1"
            mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_query_api_servers(sut: SUT):
    test_inputs = [[1], [2], [3], [4]]

    with patch.object(sut, "query") as mock_query_api_vllm:
        mock_query_api_vllm.return_value = "Output"

        outputs = await sut.query_servers(test_inputs)

        assert outputs == ["Output"] * len(test_inputs)
        assert mock_query_api_vllm.call_count == len(test_inputs)
