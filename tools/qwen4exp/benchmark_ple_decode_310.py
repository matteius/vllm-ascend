# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the fused 310P Qwen4Exp PLE decode path at model geometry.

This is an operator benchmark, not whole-model throughput. It compares the
breakable-graph eager callback that the fused operator replaces, including its
final copy into the stable graph buffer, against a fused write to that buffer.
Run it with the inference server stopped and save every result to a new path.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from vllm_ascend.models.qwen4_exp.ops.ple import ple_decode_310, ple_gate
from vllm_ascend.utils import enable_custom_op


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label", default="unlabeled")
    args = parser.parse_args()
    if args.output.exists() or min(*args.tokens, args.iterations, args.repeats) < 1:
        parser.error("use a new output path and positive token/iteration/repeat counts")
    if not torch_npu.npu.is_available() or "310P" not in torch_npu.npu.get_device_name(0).upper():
        raise RuntimeError("requires Ascend 310P")

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    generator = torch.Generator().manual_seed(31038)
    group_size = 2560
    hc_groups = 4
    hidden_size = group_size * hc_groups
    eps = 1e-6
    norm_key = (torch.randn(hidden_size, generator=generator) * 0.05).half().npu()
    norm_query = (torch.randn(hidden_size, generator=generator) * 0.05).half().npu()
    norm_conv = (torch.randn(hidden_size, generator=generator) * 0.05).half().npu()
    current_weight = (torch.randn(hidden_size, generator=generator) * 0.05).float().npu()

    with args.output.open("x") as output_file:
        for num_tokens in args.tokens:
            projected = (torch.randn(num_tokens, hidden_size + group_size, generator=generator) * 0.1).half().npu()
            hidden = (torch.randn(num_tokens, hidden_size, generator=generator) * 0.1).half().npu()
            eager_output = torch.empty_like(hidden)
            fused_output = torch.empty_like(hidden)

            def eager_into_buffer(
                projected: torch.Tensor = projected,
                hidden: torch.Tensor = hidden,
                eager_output: torch.Tensor = eager_output,
            ) -> torch.Tensor:
                key, value = projected.split((hidden_size, group_size), dim=-1)
                gated, conv_input = ple_gate(
                    key,
                    value,
                    hidden,
                    norm_key,
                    norm_query,
                    norm_conv,
                    eps,
                    accum_dtype=torch.float32,
                )
                conv = conv_input * current_weight
                eager_output.copy_((hidden.float() + gated + F.silu(conv)).half())
                return eager_output

            def fused_into_buffer(
                projected: torch.Tensor = projected,
                hidden: torch.Tensor = hidden,
                fused_output: torch.Tensor = fused_output,
            ) -> torch.Tensor:
                return ple_decode_310(
                    projected,
                    hidden,
                    norm_key,
                    norm_query,
                    norm_conv,
                    current_weight,
                    eps,
                    output=fused_output,
                )

            expected = eager_into_buffer().cpu()
            actual = fused_into_buffer().cpu()
            torch.testing.assert_close(actual, expected, rtol=7e-3, atol=4e-3)
            eager_timing = timed_ms(eager_into_buffer, args.iterations, args.repeats)
            fused_timing = timed_ms(fused_into_buffer, args.iterations, args.repeats)
            record = {
                "scope": "qwen4exp_ple_decode_callback_not_whole_model",
                "label": args.label,
                "tokens": num_tokens,
                "hidden_size": hidden_size,
                "group_size": group_size,
                "hc_groups": hc_groups,
                "max_abs_difference": (actual.float() - expected.float()).abs().max().item(),
                "eager_callback": eager_timing,
                "fused_callback": fused_timing,
                "speedup": eager_timing["median_ms"] / fused_timing["median_ms"],
                "iterations": args.iterations,
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
            }
            line = json.dumps(record)
            print(line, flush=True)
            output_file.write(line + "\n")
            output_file.flush()


if __name__ == "__main__":
    main()
