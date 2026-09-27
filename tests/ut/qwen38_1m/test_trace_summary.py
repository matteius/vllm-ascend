# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for profiling evidence accounting."""

import pytest

from tools.qwen4exp.summarize_trace import category, optional_metric, summarize_file, to_ns, union_ns


def test_epoch_timestamp_preserves_submicrosecond_difference():
    assert to_ns("1790475934962525.857\t") - to_ns("1790475934962525.856") == 1


def test_union_distinguishes_overlap_from_summed_time():
    assert union_ns([]) == 0
    assert union_ns([(0, 10), (5, 15), (3, 7), (20, 25)]) == 20


def test_trace_summary_keeps_work_separate_from_latency(tmp_path):
    trace = tmp_path / "kernel_details.csv"
    trace.write_text(
        "Device_id,Name,Start Time(us),Duration(us)\n"
        "0,QwenW4RoutedMatmulV310,1000000000000.000,10\n"
        "0,hccl_AllReduce,1000000000005.000,10\n"
        "0,Cast,1000000000020.000,5\n"
    )
    result = summarize_file(trace)
    assert result["task_count"] == 3
    assert result["task_span_ms"] == 0.025
    assert result["task_union_ms"] == 0.020
    assert result["summed_task_ms"] == 0.025
    categories = {entry["name"]: entry for entry in result["categories"]}
    assert categories["w4_routed_projection"]["percent_summed_task_time"] == 40
    assert category("qwen_w4_routed_matmul_v310_0") == "w4_routed_projection"
    assert category("QuantBatchMatmulV3") == "other_matrix_multiply"


def test_mixed_device_timeline_is_rejected(tmp_path):
    trace = tmp_path / "kernel_details.csv"
    trace.write_text("Device_id,Name,Start Time(us),Duration(us)\n0,Cast,0,10\n1,Cast,0,10\n")
    with pytest.raises(ValueError, match="multiple devices"):
        summarize_file(trace)


@pytest.mark.parametrize("value", [None, "", "N/A", "NaN", "Inf", "-1"])
def test_missing_hardware_counter_is_not_reported_as_zero(value):
    assert optional_metric(value) is None
    assert optional_metric("0") == 0.0


def test_projection_shapes_keep_hardware_metrics_separate(tmp_path):
    trace = tmp_path / "kernel_details.csv"
    trace.write_text(
        "Device_id,Name,Start Time(us),Duration(us),Input Shapes,Block Num,scalar_time(us),mac_time(us)\n"
        '0,QwenW4RoutedMatmulV310,0,10,"30,2560",20,8,1\n'
        '0,QwenW4RoutedMatmulV310,20,30,"30,2560",20,10,N/A\n'
        '0,QwenW4RoutedMatmulV310,60,50,"30,640",80,40,0\n'
    )
    shapes = {entry["input_shapes"]: entry for entry in summarize_file(trace)["projection_shapes"]}
    gate = shapes["30,2560"]
    assert gate["count"] == 2
    assert gate["block_counts"] == [20]
    assert gate["median_task_us"] == 20
    assert gate["pipeline_counters"] == {
        "scalar_time(us)": {"samples": 2, "median_us": 9},
        "mac_time(us)": {"samples": 1, "median_us": 1},
    }
    assert shapes["30,640"]["block_counts"] == [80]
    assert shapes["30,640"]["pipeline_counters"]["mac_time(us)"]["median_us"] == 0
