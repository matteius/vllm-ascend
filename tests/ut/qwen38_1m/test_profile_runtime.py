# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import cProfile
import json
import pstats
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.qwen4exp.profile_runtime import (
    analyse_npu,
    call_graph,
    capture,
    export_cprofile,
    npu_capture,
    profiler_config,
    pyspy_command,
)
from tools.qwen4exp.summarize_trace import category, summarize_file


def test_cold_profile_does_not_skip_prefill():
    assert profiler_config("/tmp/trace", "cold-prefill")["delay_iterations"] == 0
    assert profiler_config("/tmp/trace", "decode")["delay_iterations"] == 4
    assert profiler_config("/tmp/trace", "decode", delay=0)["delay_iterations"] == 0
    with pytest.raises(ValueError):
        profiler_config("/tmp/trace", "cold-prefill", steps=0)


def test_recorder_stopped_on_client_exception():
    with patch("tools.qwen4exp.profile_runtime.profile_rpc") as rpc:
        with pytest.raises(RuntimeError, match="client"), npu_capture("http://127.0.0.1:8001"):
            raise RuntimeError("client failure")
        assert [call.args[1] for call in rpc.call_args_list] == ["start_profile", "stop_profile"]


def test_failed_start_does_not_send_stop():
    with patch("tools.qwen4exp.profile_runtime.profile_rpc", side_effect=RuntimeError("disabled")) as rpc:
        with pytest.raises(RuntimeError, match="disabled"), npu_capture("http://127.0.0.1:8001"):
            pass
        assert rpc.call_count == 1


def test_capture_dry_run_never_touches_server_or_output(tmp_path):
    args = SimpleNamespace(
        command=["--", "false"],
        trace_dir="/tmp/npu",
        phase="cold-prefill",
        steps=4,
        delay=None,
        execute=False,
        output=tmp_path / "evidence",
    )
    with patch("tools.qwen4exp.profile_runtime.profile_rpc", side_effect=AssertionError("live call")):
        capture(args)
    assert not args.output.exists()


def test_speedscope_includes_workers_and_threads(tmp_path):
    command = pyspy_command(123, 60, tmp_path / "cpu.json")
    assert "--subprocesses" in command and "--threads" in command and "speedscope" in command
    with pytest.raises(ValueError):
        pyspy_command(0, 60, tmp_path / "cpu.json")


def test_cprofile_graph_export_preserves_evidence(tmp_path):
    profile = cProfile.Profile()
    profile.runcall(lambda: json.dumps({"key": "value"}))
    source, output = tmp_path / "python.pstats", tmp_path / "python.dot"
    profile.dump_stats(source)
    export_cprofile(source, output)
    dot = output.read_text()
    assert "digraph" in dot and " -> " in dot and "cumulative=" in dot
    assert "digraph" in call_graph(pstats.Stats(profile), 1)
    with pytest.raises(FileExistsError):
        export_cprofile(source, output)


def test_native_and_grouped_have_distinct_trace_categories():
    assert category("QwenW4A8Int4MatmulV310") == "w4a8_native_int4_projection"
    assert category("QwenW4GroupedMatmulV310") == "w4_device_grouped_projection"
    assert category("QwenW4RoutedMatmulV310") == "w4_routed_projection"


def test_new_projection_categories_keep_shape_diagnostics(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text(
        "Device_id,Name,Start Time(us),Duration(us),Input Shapes,Block Num\n"
        '0,QwenW4GroupedMatmulV310,0,10,"512,2560",8\n'
        '0,QwenW4A8Int4MatmulV310,20,30,"512,1280",8\n'
    )
    report = summarize_file(path)
    assert len(report["projection_shapes"]) == 2
    assert {item["name"] for item in report["categories"]} == {
        "w4_device_grouped_projection",
        "w4a8_native_int4_projection",
    }


def test_offline_npu_analysis_validates_and_preserves_output(tmp_path):
    with pytest.raises(ValueError, match="captures"):
        analyse_npu(tmp_path)
    raw = tmp_path / "rank0_ascend_pt"
    (raw / "FRAMEWORK").mkdir(parents=True)
    output = raw / "ASCEND_PROFILER_OUTPUT"

    def parse(root, *, max_process_number):
        assert root == str(tmp_path) and max_process_number == 1
        output.mkdir()
        (output / "kernel_details.csv").write_text("Name,Duration(us)\n")

    with patch("tools.qwen4exp.profile_runtime.importlib.import_module") as load:
        load.return_value.analyse.side_effect = parse
        assert analyse_npu(tmp_path) == [output / "kernel_details.csv"]
        with pytest.raises(FileExistsError):
            analyse_npu(tmp_path)
        load.assert_called_once_with("torch_npu.profiler.profiler")


def test_offline_npu_analysis_rejects_incomplete_export(tmp_path):
    (tmp_path / "rank0_ascend_pt" / "FRAMEWORK").mkdir(parents=True)
    with (
        patch("tools.qwen4exp.profile_runtime.importlib.import_module"),
        pytest.raises(RuntimeError, match="kernel_details"),
    ):
        analyse_npu(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        analyse_npu(tmp_path, 0)
