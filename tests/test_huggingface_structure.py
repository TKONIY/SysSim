"""Structural tests for Hugging Face integration (no model loading required)."""

import pytest
from syssim import HardwareInfo, SimulatorConfig


def test_integration_module_imports():
    """Test that integration module can be imported."""
    from syssim.integrations import huggingface
    assert huggingface is not None


def test_function_exports():
    """Test that HF functions are exported from main package."""
    from syssim import trace_hf_model_for_training, trace_hf_training_step
    assert callable(trace_hf_model_for_training)
    assert callable(trace_hf_training_step)


def test_hf_available_flag():
    """Test HF_AVAILABLE flag reflects transformers installation."""
    from syssim.integrations.huggingface import HF_AVAILABLE
    # Just check it's a boolean
    assert isinstance(HF_AVAILABLE, bool)


def test_config_creation():
    """Test that config can be created for HF integration."""
    hw = HardwareInfo(989.0, 989.0, 3350.0)
    config = SimulatorConfig(hw_info=hw)
    assert config is not None
    assert config.hw_info.peak_tflops_mm == 989.0


def test_helper_functions_exist():
    """Test that helper functions are defined."""
    from syssim.integrations.huggingface import (
        _create_lm_loss_fn,
        _ensure_cuda_inputs,
    )
    assert callable(_create_lm_loss_fn)
    assert callable(_ensure_cuda_inputs)


def test_ensure_cuda_inputs_with_dict():
    """Test _ensure_cuda_inputs with dict input (no actual CUDA required)."""
    from syssim.integrations.huggingface import _ensure_cuda_inputs
    import torch

    # Test with CPU tensors (won't actually move to CUDA if not available)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "some_number": 42,
        "some_string": "hello",
    }

    # This should not crash, just return the inputs
    # (actual CUDA migration only happens if CUDA is available)
    try:
        result = _ensure_cuda_inputs(inputs)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["some_number"] == 42
        assert result["some_string"] == "hello"
    except RuntimeError:
        # If CUDA not available, that's expected
        pytest.skip("CUDA not available")


def test_imports_dont_fail():
    """Test that imports don't crash (even if transformers unavailable)."""
    try:
        from syssim.integrations.huggingface import (
            trace_hf_model_for_training,
            trace_hf_training_step,
        )
        # If transformers unavailable, functions should still be defined
        assert callable(trace_hf_model_for_training)
        assert callable(trace_hf_training_step)
    except Exception as e:
        pytest.fail(f"Imports should not fail: {e}")
