

import pytest
import torch
from unittest.mock import Mock

# Common prompt lists
BASIC_PROMPTS = ["def func1():", "def func2():"]
LARGE_PROMPTS = ["def function():"] * 50


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 1
    tokenizer.chat_template = None

    def mock_call(text, **kwargs):
        mock_result = Mock()
        if isinstance(text, list):
            mock_result.input_ids = torch.tensor(
                [[1, 2, 3, 4]] * len(text)
            )
            mock_result.attention_mask = torch.tensor(
                [[1, 1, 1, 1]] * len(text)
            )
        else:
            mock_result.input_ids = torch.tensor([[1, 2, 3, 4]])
            mock_result.attention_mask = torch.tensor([[1, 1, 1, 1]])
        mock_result.to = Mock(return_value=mock_result)
        mock_result.__getitem__ = lambda self, key: getattr(self, key)
        return mock_result

    tokenizer.side_effect = mock_call
    tokenizer.decode = Mock(return_value="mocked output")
    return tokenizer


@pytest.fixture
def mock_model():
    model = Mock()
    model.generate = Mock()

    def mock_generate(*args, **kwargs):
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        batch_size = args[0].shape[0] if args else 1
        total_sequences = batch_size * num_return_sequences
        return torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]] * total_sequences
        )

    model.generate.side_effect = mock_generate
    model.to = Mock(return_value=model)
    mock_param = Mock()
    mock_param.is_cuda = True
    model.parameters = Mock(return_value=[mock_param])
    return model


@pytest.fixture(scope="class")
def small_model():
    from src.models.huggingface_model import HuggingFaceModel
    try:
        model = HuggingFaceModel("gpt2", max_tokens=50)
        yield model
    except Exception:
        pytest.skip("Model loading failed")


@pytest.fixture(scope="class")
def gpu_model():
    from src.models.huggingface_model import HuggingFaceModel
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    try:
        model = HuggingFaceModel("gpt2", max_tokens=30, device="cuda")
        yield model
    except Exception:
        pytest.skip("GPU model loading failed")


@pytest.fixture(scope="class")
def small_vllm_model():
    from src.models.vllm_model import VLLMModel
    if not torch.cuda.is_available():
        pytest.skip("VLLM requires GPU")
    try:
        model = VLLMModel("microsoft/DialoGPT-small", max_tokens=50)
        yield model
    except Exception:
        pytest.skip("VLLM model loading failed")


@pytest.fixture(scope="class")
def gpu_vllm_model():
    from src.models.vllm_model import VLLMModel
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    try:
        model = VLLMModel("microsoft/DialoGPT-small", max_tokens=30)
        yield model
    except Exception:
        pytest.skip("GPU VLLM model loading failed")

