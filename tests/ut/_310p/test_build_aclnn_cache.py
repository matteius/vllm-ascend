# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for incremental 310P custom-op builds."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_kernel_changes_invalidate_copy_and_compile_stamps():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    assert 'find "${op_path}/op_kernel" -type f -newer "${source_stamp}"' in build_script
    assert '"${op_path}/op_host/CMakeLists.txt" -nt "${source_stamp}"' in build_script
    assert 'rm -f -- "${source_stamp}"' in build_script
    assert '"${binary_root}/src/${op_name}" -maxdepth 1 -type f' in build_script
    assert "-name '*.py' -delete" in build_script
    assert 'cmp -s "${generated_launcher}" "${copied_launcher}"' in build_script
    assert '-name "${op_name}_${SOC_ARG}_*.done"' in build_script


def test_w2_shared_kernel_changes_invalidate_compile_stamps():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    assert '"${op_name}" == "w2_blocked_dequant_matmul_v310"' in build_script
    assert 'find "${ROOT_DIR}/csrc/moe/common/kernel_utils" -type f' in build_script


def test_host_changes_invalidate_generated_operator_metadata():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    assert 'find "${op_path}/op_host" -type f -newer "${generated_proto}"' in build_script
    assert 'rm -f -- "${generated_proto}"' in build_script
    assert "hash_op_host_sources()" in build_script
    assert '[[ ! -f "${host_stamp}" ]]' in build_script
    assert '[[ "$(< "${host_stamp}")" != "${host_hash}" ]]' in build_script
    assert '-name "${op_type}_*_param.json"' in build_script
    assert '-name "kernel_meta_${op_type}_*"' in build_script
    assert 'rm -rf -- "${binary_root}/bin/${op_name}"' in build_script
    assert 'rm -f -- "${binary_root}/bin/${op_name}.json"' in build_script


def test_host_fingerprint_is_saved_only_after_successful_build():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    build_call = 'bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"'
    assert build_script.index("update_host_cache_fingerprints()") < build_script.index(build_call)
    assert build_script.index(build_call) < build_script.index("  update_host_cache_fingerprints")


def test_dead_compiler_locks_are_removed_before_build():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    assert "remove_stale_kernel_locks()" in build_script
    assert "lock_pid=$(tr -cd '0-9'" in build_script
    assert '! kill -0 "${lock_pid}"' in build_script
    assert 'rm -f -- "${lock_file}"' in build_script


def test_gcc15_protobuf_build_includes_cstdint():
    protobuf_cmake = (REPO_ROOT / "csrc" / "cmake" / "third_party" / "ascend_protobuf.cmake").read_text()

    assert "CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15" in protobuf_cmake
    assert 'string(APPEND protobuf_CXXFLAGS " -include cstdint")' in protobuf_cmake


def test_chunk_kda_is_registered_and_built_for_310p():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()
    op_definition = (
        REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "chunk_kda_fwd_def.cpp"
    ).read_text()

    assert '"chunk_kda_fwd"' in build_script
    assert 'AddConfig("ascend310p", config)' in op_definition


def test_chunk_kda_uses_unified_default_task_type_on_310p():
    kernel = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd.cpp").read_text()
    cmake = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "CMakeLists.txt").read_text()
    entry_signature = 'extern "C" __global__ __aicore__ void chunk_kda_fwd('
    assert kernel.count(entry_signature) == 1

    arch20_start = kernel.rindex("#if defined(KDA_310P_DEFAULT_TASK)")
    arch20_end = kernel.index("#else", arch20_start)
    task_selection_end = kernel.index("#endif", arch20_end)
    arch20_entry = kernel[arch20_start:arch20_end]

    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AICORE);" in arch20_entry
    assert "KERNEL_TASK_TYPE(1" not in arch20_entry
    assert "KERNEL_TASK_TYPE(2" not in arch20_entry
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in kernel[arch20_end:task_selection_end]
    assert "KdaForward::RunKernel(" in kernel[task_selection_end:]
    assert "KERNEL_TASK_TYPE(1" not in kernel
    assert "KERNEL_TASK_TYPE(2" not in kernel
    assert 'if("ascend310p" IN_LIST ASCEND_COMPUTE_UNIT)' in cmake
    assert "OPTIONS -DKDA_310P_DEFAULT_TASK=1" in cmake


def test_chunk_kda_dispatches_without_keyed_runtime_selection_on_310p():
    kernel = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd.cpp").read_text()

    arch20_dispatch = kernel[
        kernel.index("#if defined(KDA_310P_DEFAULT_TASK)", kernel.index("GET_TILING_DATA_WITH_STRUCT")) : kernel.index(
            "if (TILING_KEY_IS(1))"
        )
    ]
    assert "tilingData.chunkSize == 64" in arch20_dispatch
    assert "tilingData.kHeadDim == 128" in arch20_dispatch
    assert "tilingData.vHeadDim == 128" in arch20_dispatch
    assert "DispatchGenericSafeGate" in arch20_dispatch
    assert arch20_dispatch.rstrip().endswith("#else")


def test_chunk_kda_uses_unified_default_tiling_key_on_310p():
    tiling = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "chunk_kda_fwd_tiling.cpp").read_text()

    assert "isAscend310P ? 0 : (useChunk64K128V128Template ? 2 : 1)" in tiling
    assert "SocVersion::ASCEND310P" in tiling


def test_chunk_kda_normalizes_mixed_vector_core_indices_on_310p():
    kernel_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel"
    common = (kernel_dir / "chunk_kda_fwd_common.h").read_text()
    gate = (REPO_ROOT / "csrc" / "attention" / "kda_gate_cumsum" / "op_kernel" / "kda_gate_cumsum_kernel.h").read_text()
    tiling = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "chunk_kda_fwd_tiling.cpp").read_text()

    assert "return static_cast<uint64_t>(block_idx);" in common
    assert "static_cast<uint64_t>(block_idx) % usedCoreNum_" in gate
    assert "(isAscend310P ? 1 : 2)" in tiling

    for filename in (
        "chunk_kda_fwd_prepare.h",
        "chunk_kda_fwd_post_wu.h",
        "chunk_kda_fwd_finalize.h",
    ):
        source = (kernel_dir / filename).read_text()
        assert "KdaForward::GetPhysicalBlockIdx()" in source
        assert "GetBlockIdx()" not in source


def test_chunk_kda_runs_vector_stages_in_the_launched_310p_image():
    kernel_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel"
    arch20_guard = "#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)"

    common = (kernel_dir / "chunk_kda_fwd_common.h").read_text()
    vector_capability = common[
        common.index("constexpr bool CompilesVectorPipeline") : common.index("constexpr bool CompilesCubePipeline")
    ]
    assert "defined(KDA_310P_DEFAULT_TASK)" in vector_capability
    assert "defined(__DAV_VEC__)" in common
    assert "defined(__DAV_M200_VEC__)" in vector_capability
    assert "defined(__DAV_CUBE__)" in common
    assert "__ENABLE_VECTOR_CORE__" not in common
    gate_dispatch = common[common.index("void RunGateCumsum") : common.index("void RunFrontEnd")]
    assert arch20_guard in gate_dispatch
    assert "CompilesVectorPipeline()" not in gate_dispatch
    assert "vector stage directly in that image" in gate_dispatch
    assert "DispatchKdaGateCumsum" in gate_dispatch
    arch20_fp32_gate = gate_dispatch[
        gate_dispatch.index("The 310P registration accepts FP32 gates only") : gate_dispatch.index(
            "#else", gate_dispatch.index("The 310P registration accepts FP32 gates only")
        )
    ]
    assert "DispatchKdaGateCumsum<float>" in arch20_fp32_gate
    assert "DispatchKdaGateCumsum<bfloat16_t>" not in arch20_fp32_gate

    expected_pipelines = {
        "chunk_kda_fwd_prepare.h": ("op.ProcessAic();", "op.ProcessAiv();"),
        "chunk_kda_fwd_post_wu.h": (
            "op.template ProcessAic<SYNCHRONIZE_PIPELINES>();",
            "op.template ProcessAiv<SYNCHRONIZE_PIPELINES>();",
        ),
        "chunk_kda_fwd_finalize.h": (
            "op.template ProcessAic<SYNCHRONIZE_PIPELINES>();",
            "op.template ProcessAiv<SYNCHRONIZE_PIPELINES>();",
        ),
    }
    for filename, calls in expected_pipelines.items():
        source = (kernel_dir / filename).read_text()
        assert source.count(arch20_guard) >= 2
        assert "CompilesCubePipeline()" in source
        assert all(call in source for call in calls)

    for filename in ("chunk_kda_fwd_post_wu.h", "chunk_kda_fwd_finalize.h"):
        source = (kernel_dir / filename).read_text()
        assert "default-task 310P image is unified" in source


def test_chunk_kda_uses_fp16_score_workspace_on_310p():
    prepare = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd_prepare.h").read_text()
    score_type = prepare[prepare.index("using AKK_T") : prepare.index("template <typename TilingData>")]

    assert "__CCE_AICORE__ == 200" in score_type
    assert "using SCORE_T = T;" in score_type


def test_chunk_kda_uses_no_fixpipe_mmad_on_310p():
    kernel_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel"
    unified_policy = "MmadPingpongTlaMulti<KdaArchTag, true, false>"

    for filename in (
        "chunk_kda_fwd_prepare.h",
        "chunk_kda_fwd_post_wu.h",
        "chunk_kda_fwd_finalize.h",
    ):
        source = (kernel_dir / filename).read_text()
        policy_block = source[source.index("using KdaArchTag") : source.index("using KdaL1TileShape")]
        assert policy_block.count(unified_policy) == 2


def test_310p_fp32_cube_epilogue_orders_ub_before_gm_copy():
    block_mmad = (
        REPO_ROOT / "csrc" / "moe" / "common" / "kernel_utils" / "block" / "block_mmad_pingpong_tla_multi.hpp"
    ).read_text()
    fp32_epilogue = block_mmad[
        block_mmad.index("if constexpr (std::is_same_v<ElementC, ElementAccumulator>)") : block_mmad.index(
            "} else {", block_mmad.index("if constexpr (std::is_same_v<ElementC, ElementAccumulator>)")
        )
    ]
    assert "SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7)" in fp32_epilogue
    assert "WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7)" in fp32_epilogue


def test_chunk_kda_keeps_fp32_solve_off_310p_cube():
    prepare = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd_prepare.h").read_text()

    capability_end = prepare.index("using KdaArchTag")
    capability_start = prepare.rindex("#if defined(__CCE_AICORE__)", 0, capability_end)
    capability = prepare[capability_start:capability_end]
    assert "__CCE_AICORE__ == 200" in capability
    assert "KDA_SUPPORTS_FP32_CUBE_SOLVE = false" in capability
    assert prepare.count("if constexpr (KDA_SUPPORTS_FP32_CUBE_SOLVE)") >= 2

    vector_solve = prepare[
        prepare.index("void SolveUnitLower64OnVector") : prepare.index("void SolveDiagonalBlocksInRows")
    ]
    assert "Brcb(rowBrcb, row" in vector_solve
    assert "matrix.SetValue(i * matrixSize + i, 1.0f)" in vector_solve

    finalize = prepare[prepare.index("void FinalizeScores310P") : prepare.index("void ProcessPreAivScorePrepare310P")]
    assert finalize.index("PrepareAqkAkkSolveInputRows") < finalize.index("StoreSolveXRowsToAkk")
    assert "FinalizePrepareIntermediates" not in finalize
    assert "false, false, true" in finalize
    assert "if (solveFullOnVector)" in prepare
    assert "Muls(aqkMat, aqkMat, scale_" in prepare


def test_chunk_kda_310p_serializes_fp16_score_exports_through_row_buffer():
    prepare = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd_prepare.h").read_text()
    finalize = prepare[
        prepare.index("void FinalizePrepareIntermediates") : prepare.index(
            "bool ResolveFlatChunk", prepare.index("void FinalizePrepareIntermediates")
        )
    ]

    assert "__CCE_AICORE__ == 200" in finalize
    assert "LocalTensor<float> rowLocal = exp2Buf_.Get<float>();" in finalize
    assert "LoadAsFloatRow(aqk_" in finalize
    assert "StoreFloatRow(o_" in finalize
    assert "LoadAsFloatRow(akk_" in finalize
    assert "StoreFloatRow(u_" in finalize
    assert "LoadAsFloatRow(qg_" in finalize
    assert "StoreFloatRow(kg_" in finalize

    init = prepare[prepare.index("void Init(") : prepare.index("void ProcessAivOnly")]
    assert "KDA_FINALIZE_TILE_ROWS * (2 * BT_ + K_) * sizeof(T)" in init
    assert "finalizeWritebackBytes > writebackBytes" in init


def test_chunk_kda_310p_materializes_scaled_qg_during_gate_prepare():
    prepare = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd_prepare.h").read_text()
    gate_products = prepare[
        prepare.index("void PrepareGateProductsBulk310P") : prepare.index(
            "void PrepareGateProductsBulk(", prepare.index("void PrepareGateProductsBulk310P")
        )
    ]

    assert "Muls(qgScaledTyped, qTyped, static_cast<T>(scale_)" in gate_products
    assert "CopyVectorOut(kg_" in gate_products
    assert "Cast(" not in gate_products


def test_chunk_kda_prunes_unsupported_bf16_inputs_on_310p():
    op_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host"
    cmake = (op_dir / "CMakeLists.txt").read_text()
    op_def = (op_dir / "chunk_kda_fwd_def.cpp").read_text()
    op_def_310p = (op_dir / "310p" / "chunk_kda_fwd_def.cpp").read_text()
    api = (op_dir / "op_api" / "aclnn_chunk_kda_fwd.cpp").read_text()
    tiling = (op_dir / "chunk_kda_fwd_tiling.cpp").read_text()

    assert '"ascend310p" IN_LIST ASCEND_COMPUTE_UNIT' in cmake
    assert "310p/chunk_kda_fwd_def.cpp" in cmake
    assert "ge::DT_FLOAT16" in op_def_310p
    assert "ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16" not in op_def_310p
    assert "ge::DT_BF16" not in op_def_310p
    assert "Ascend 310P requires float32 g." in api
    assert "Ascend 310P requires float32 beta." in api
    assert "Ascend 310P supports chunkSize 64 only." in api
    assert "isAscend310P && chunkSize != 64" in tiling
    assert 'this->AICore().AddConfig("ascend310p", config)' in op_def_310p
    assert "Ascend 310P requires float16 q, k and v." in api
    signature_pattern = r'this->(?:Input|Output|Attr)\("([^"]+)"\)'
    assert re.findall(signature_pattern, op_def_310p) == re.findall(signature_pattern, op_def)


def test_chunk_kda_uses_physical_stage_boundaries_on_310p():
    op_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd"
    api = (op_dir / "op_host" / "op_api" / "aclnn_chunk_kda_fwd.cpp").read_text()
    kernel = (op_dir / "op_kernel" / "chunk_kda_fwd.cpp").read_text()
    post_wu = (op_dir / "op_kernel" / "chunk_kda_fwd_post_wu.h").read_text()

    assert 'std::strstr(socName, "Ascend310P")' in api
    assert "const bool splitStages = IsAscend310P() ||" in api
    assert "KDA_STAGE_POST_WU_FINALIZE" in api
    assert "KDA_STAGE_POST_WU_FINALIZE" in kernel
    assert "KDA_STAGE_FINALIZE_WRITEBACK" in api
    assert "KDA_STAGE_FINALIZE_WRITEBACK" in kernel
    stage_order = api[api.index("if (splitStages && IsAscend310P())") : api.index("} else if (splitStages)")]
    assert stage_order.index("KDA_STAGE_PREPARE_CUBE") < stage_order.index("KDA_STAGE_PREPARE_VECTOR_FINALIZE")
    assert stage_order.index("KDA_STAGE_PREPARE_VECTOR_FINALIZE") < stage_order.index("KDA_STAGE_POST_WU")
    assert "l0op::Slice(" in stage_order
    assert "KdaFwdCopyMaybeCastAfter(" in stage_order
    assert "aqkFp32, scoreMatricesCompute, aqkCompute" in stage_order
    assert "akkFp32, scoreMatricesCompute, akkCompute" in stage_order
    assert "RunChunkKdaPrepareScoreInputs310P" in kernel
    assert "RunChunkKdaPrepareScores310P" in kernel
    assert "RunChunkKdaPrepareScoreFinalize310P" in kernel
    assert "T, T, T, TilingData, true, false, false" in kernel
    assert "T, T, T, TilingData, false, true, false" in kernel
    assert "if constexpr (SYNCHRONIZE_PIPELINES)" in post_wu


def test_chunk_kda_materializes_prepare_scores_between_310p_stages():
    op_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd"
    aclnn = (op_dir / "op_host" / "op_api" / "aclnn_chunk_kda_fwd.cpp").read_text()
    op_def = (op_dir / "op_host" / "310p" / "chunk_kda_fwd_def.cpp").read_text()
    l0 = (op_dir / "op_host" / "op_api" / "chunk_kda_fwd.cpp").read_text()
    tiling = (op_dir / "op_host" / "chunk_kda_fwd_tiling.cpp").read_text()
    kernel = (op_dir / "op_kernel" / "chunk_kda_fwd.cpp").read_text()
    prepare = (op_dir / "op_kernel" / "chunk_kda_fwd_prepare.h").read_text()

    assert 'Output("score_scratch")' in op_def
    assert 'Output("score_matrices")' in op_def
    assert "scoreScratchOut, scoreMatricesOut, stageTokenOut)," in l0
    assert "GM_ADDR qg_scaled, GM_ADDR u_seed, GM_ADDR score_scratch," in kernel
    assert "GM_ADDR score_matrices, GM_ADDR stage_token, GM_ADDR workspace," in kernel
    assert "scoreScratchCompute" in aclnn
    assert "scoreMatricesCompute" in aclnn
    assert "externalScoreWorkspace" in prepare
    assert "GM_ADDR akkFp32 = scoreMatrices + matrixBytes;" in prepare
    assert prepare.count("task * slotsPerTask") == 2
    assert prepare.count("scoreSlotBase + block") >= 2
    assert "const uint64_t prepareAqkFp32Offset = isAscend310P" in tiling
    assert "const uint64_t prepareAkkFp32Offset = isAscend310P" in tiling
    assert "const uint64_t scoreBytes = isAscend310P" in tiling
    assert "? cursor\n        : AllocateWorkspace(cursor, matrixBytes)" in tiling
    assert "const uint64_t scoreBytes = isAscend310P\n        ? 0" in tiling


def test_chunk_kda_serializes_physical_stages_with_explicit_tokens():
    op_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd"
    aclnn = (op_dir / "op_host" / "op_api" / "aclnn_chunk_kda_fwd.cpp").read_text()
    op_def = (op_dir / "op_host" / "310p" / "chunk_kda_fwd_def.cpp").read_text()
    l0 = (op_dir / "op_host" / "op_api" / "chunk_kda_fwd.cpp").read_text()

    assert 'Input("stage_dependency")' in op_def
    assert 'Input("gk_fp16")' in op_def
    assert 'Input("beta_fp16")' in op_def
    assert 'Output("stage_token")' in op_def
    assert "stageDependencyOptional, gkFp16Optional, betaFp16Optional)," in l0
    assert "scoreMatricesOut, stageTokenOut)," in l0
    assert "stageDependency = stageResult[15];" in aclnn
    assert "stageDependency = akkCompute;" in aclnn


def test_chunk_kda_310p_uses_external_reverse_casts_and_fp16_math():
    op_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd"
    aclnn = (op_dir / "op_host" / "op_api" / "aclnn_chunk_kda_fwd.cpp").read_text()
    prepare = (op_dir / "op_kernel" / "chunk_kda_fwd_prepare.h").read_text()
    post_wu = (op_dir / "op_kernel" / "chunk_kda_fwd_post_wu.h").read_text()
    finalize = (op_dir / "op_kernel" / "chunk_kda_fwd_finalize.h").read_text()

    assert "l0op::KdaGateCumsum(" in aclnn
    assert "gkFp16Compute = l0op::Cast(" in aclnn
    assert "betaFp16Compute = l0op::Cast(" in aclnn

    score_fp16 = prepare[
        prepare.index("void PrepareScoreFactorsBulk310P") : prepare.index(
            "void PrepareScoreFactorsBulk(",
            prepare.index("void PrepareScoreFactorsBulk310P"),
        )
    ]
    beta_fp16 = prepare[
        prepare.index("void ScaleRowsByBeta310P") : prepare.index(
            "void ScaleRowsByBeta(", prepare.index("void ScaleRowsByBeta310P")
        )
    ]
    kg_fp16 = post_wu[
        post_wu.index("void FinalizeKg310P") : post_wu.index(
            "void CopyScratchWAndFinalizeKg", post_wu.index("void FinalizeKg310P")
        )
    ]
    assert "Cast(" not in score_fp16
    assert "Cast(" not in beta_fp16
    assert "Cast(" not in kg_fp16
    assert "using W_OUT_T = T;" in post_wu
    assert "using OUT_T = T;" in finalize


def test_chunk_kda_post_wu_initializes_catlass_pipeline_flags():
    post_wu = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd_post_wu.h").read_text()

    compute_post_wu = post_wu[post_wu.index("void ComputePostWuCube") : post_wu.index("void CopyScratchWAndFinalizeKg")]
    assert compute_post_wu.count(".preSetFlags();") == 3
    assert compute_post_wu.count(".finalWaitFlags();") == 3


def test_chunk_kda_finalize_initializes_catlass_pipeline_flags():
    finalize = (
        REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd_finalize.h"
    ).read_text()

    compute_output = finalize[finalize.index("void ComputeOutputCube") : finalize.index("void FinalizeOutputRows")]
    assert compute_output.count(".preSetFlags();") == 2
    assert compute_output.count(".finalWaitFlags();") == 2
    assert "if constexpr (SYNCHRONIZE_PIPELINES)" in finalize


def test_kda_varlen_boundaries_use_310p_scalar_global_reads():
    op_dir = REPO_ROOT / "csrc" / "attention" / "kda_gate_cumsum" / "op_kernel"
    for source_name in ("kda_gate_cumsum.cpp", "kda_gate_cumsum_kernel.h"):
        kernel = (op_dir / source_name).read_text()
        read_int64 = kernel[kernel.index("ReadInt64") : kernel.index("ExpScalar")]

        assert "__CCE_AICORE__ == 200" in read_int64
        assert "return tensor.GetValue(offset);" in read_int64


def test_chunk_kda_helpers_are_registered_for_310p():
    for op_name in ("kda_gate_cumsum", "kda_layout_swap12"):
        op_definition = (REPO_ROOT / "csrc" / "attention" / op_name / "op_host" / f"{op_name}_def.cpp").read_text()

        assert 'AddConfig("ascend310p", aicoreConfig)' in op_definition


def test_kda_layout_swap_avoids_keyed_task_types_on_310p():
    kernel = (
        REPO_ROOT / "csrc" / "attention" / "kda_layout_swap12" / "op_kernel" / "kda_layout_swap12.cpp"
    ).read_text()

    guard = "#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)"
    for tiling_key in range(3):
        task_type = f"KERNEL_TASK_TYPE({tiling_key}"
        assert kernel.rindex(guard, 0, kernel.index(task_type)) >= 0


def test_chunk_kda_helpers_exclude_bf16_on_310p():
    for op_name in ("kda_gate_cumsum", "kda_layout_swap12"):
        op_dir = REPO_ROOT / "csrc" / "attention" / op_name
        kernel = (op_dir / "op_kernel" / f"{op_name}.cpp").read_text()
        tiling = (op_dir / "op_host" / f"{op_name}_tiling.cpp").read_text()

        arch_guard = "#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)"
        assert kernel.index(arch_guard) < kernel.index("bfloat16_t")
        assert "SocVersion::ASCEND310P" in tiling
        assert "ge::DT_BF16" in tiling


def test_chunk_kda_loads_310p_compat_before_catlass_kernels():
    kernel_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel"
    common_header = (kernel_dir / "chunk_kda_fwd_common.h").read_text()
    compat_header = (kernel_dir / "arch20" / "compat_310p.h").read_text()

    compat_include = '#include "arch20/compat_310p.h"'
    first_catlass_kernel = '#include "../../kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h"'
    assert common_header.index(compat_include) < common_header.index(first_catlass_kernel)
    assert "#define CATLASS_UNIFIED_CORE 1" in compat_header
    assert "struct bfloat16_t" in compat_header
    assert "#define PIPE_FIX PIPE_MTE3" in compat_header
    assert "#define LoadDataWithSparse LoadDataWithSparseCal" in compat_header


def test_chunk_kda_uses_unified_gdn_state_kernel_on_310p():
    kda_common = (
        REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd_common.h"
    ).read_text()
    gdn_dir = REPO_ROOT / "csrc" / "moe" / "chunk_gated_delta_rule_fwd_h" / "op_kernel" / "arch20" / "gemm"
    gdn_kernel = (gdn_dir / "kernel" / "gdn_fwd_h_kernel.hpp").read_text()
    gdn_scheduler = (gdn_dir / "block" / "block_scheduler_gdn_fwd_h.hpp").read_text()
    gdn_vnew = (
        REPO_ROOT
        / "csrc"
        / "moe"
        / "chunk_gated_delta_rule_fwd_h"
        / "op_kernel"
        / "arch20"
        / "epilogue"
        / "block"
        / "block_epilogue_gdn_fwdh_vnew.hpp"
    ).read_text()
    gdn_update = (
        REPO_ROOT
        / "csrc"
        / "moe"
        / "chunk_gated_delta_rule_fwd_h"
        / "op_kernel"
        / "arch20"
        / "epilogue"
        / "block"
        / "block_epilogue_gdn_fwdh_update.hpp"
    ).read_text()

    assert 'arch20/gemm/kernel/gdn_fwd_h_kernel.hpp"' in kda_common
    assert "__CCE_AICORE__ == 200" in kda_common
    assert re.search(r"GDNFwdHKernel<\s*T, float, float, float>", kda_common)
    assert "void InitFromData(" in gdn_kernel
    assert "cubeBlockScheduler.InitFromData(" in gdn_kernel
    assert "void InitFromData(" in gdn_scheduler
    assert "AscendC::DataCopyPad(" in gdn_vnew
    assert "mActual * sizeof(GElementInput)" in gdn_vnew
    assert "useKdaGatedPath = true" in gdn_kernel
    assert "if (useKdaGatedPath)" in gdn_vnew
    assert "(chunkSize - 1) * kHeadDim + mOffset" in gdn_update
    assert "constexpr float LN2" in gdn_update


def test_generic_kda_state_kernel_avoids_host_debug_header_on_310p():
    kernel = (
        REPO_ROOT
        / "csrc"
        / "moe"
        / "chunk_gated_delta_rule_fwd_h"
        / "op_kernel"
        / "gemm"
        / "kernel"
        / "gdn_fwd_h_kernel.hpp"
    ).read_text()

    arch_guard = "#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)"
    assert kernel.index(arch_guard) < kernel.index('#include "catlass/debug.hpp"')
