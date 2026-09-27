# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import SimpleNamespace

import pytest

from vllm_ascend.models.qwen4_exp import w4_backend_policy as policy
from vllm_ascend.models.qwen4_exp.w4_backend_policy import (
    NATIVE_OPERATORS,
    configure_w4_backend,
    resolve_w4_backend,
)
from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND


def select(backend="auto", activation="float16", **overrides):
    arguments = dict(
        hidden_size=2560,
        intermediate_size=640,
        device_name="Ascend310P3",
        available_ops=NATIVE_OPERATORS | {"npu_qwen_w4_grouped_matmul_310"},
    )
    arguments.update(overrides)
    return resolve_w4_backend(
        dict(bits=4, backend=backend, group_size=128, activation_quantization=activation), **arguments
    )


def test_w8_is_unchanged_and_does_not_query_hardware():
    cfg = SimpleNamespace(quantization_config={"quant_method": "ascend", "bits": 8})
    before = vars(cfg).copy()
    assert configure_w4_backend(cfg) is None
    assert vars(cfg) == before


def test_fp16_policy_never_selects_activation_quantization():
    assert select(native_qualified=True).backend == "cube_310_grouped"
    with pytest.raises(ValueError, match="explicit activation_quantization"):
        select(NATIVE_INT4_BACKEND)


def test_auto_requires_qualification_and_permission():
    assert select(activation="int8_per_group").backend == "cube_310_grouped"
    chosen = select(activation="int8_per_group", native_qualified=True)
    assert chosen.backend == NATIVE_INT4_BACKEND
    assert chosen.activation_quantization == "int8_per_group"


@pytest.mark.parametrize(
    "overrides",
    [
        dict(device_name="Ascend910B"),
        dict(hidden_size=4096),
        dict(intermediate_size=7680),
        dict(available_ops=frozenset()),
    ],
)
def test_explicit_native_rejects_unsupported_capability(overrides):
    with pytest.raises(ValueError, match="requires Ascend 310P"):
        select(NATIVE_INT4_BACKEND, "int8_per_group", **overrides)
    assert select(activation="int8_per_group", native_qualified=True, **overrides).activation_quantization == "float16"


def test_explicit_native_is_separate_experimental_override():
    assert select(NATIVE_INT4_BACKEND, "int8_per_group").backend == NATIVE_INT4_BACKEND


def test_invalid_activation_policy_rejected():
    with pytest.raises(ValueError, match="activation_quantization must be"):
        select(activation="int4")


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_resolves_lazy_operator_registration_before_layout(monkeypatch, enabled):
    operators = SimpleNamespace()
    registrations = []

    def register():
        registrations.append(True)
        for name in NATIVE_OPERATORS:
            setattr(operators, name, object())
        return enabled

    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", SimpleNamespace(enable_custom_op=register))
    monkeypatch.setattr(
        policy,
        "torch",
        SimpleNamespace(
            npu=SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "Ascend310P3"),
            ops=SimpleNamespace(_C_ascend=operators),
        ),
    )
    metadata = dict(bits=4, group_size=128, backend=NATIVE_INT4_BACKEND, activation_quantization="int8_per_group")
    config = SimpleNamespace(hidden_size=2560, moe_intermediate_size=640, ascend_expert_quantization=metadata)
    if enabled:
        assert policy.configure_w4_backend(config).backend == NATIVE_INT4_BACKEND
    else:
        with pytest.raises(ValueError, match="both rebuilt operators"):
            policy.configure_w4_backend(config)
    assert registrations == [True]
