#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "chunk_kda_fwd_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 && \
    (!defined(TILING_KEY_VAR) || TILING_KEY_VAR == 2UL)
#define KDA_COMPILE_ARCH35_FAST_PATH 1
#include "arch35/chunk_kda_fwd_impl.h"
#else
#define KDA_COMPILE_ARCH35_FAST_PATH 0
#endif

namespace KdaForward {

constexpr int64_t KDA_STAGE_FULL = -1;
constexpr int64_t KDA_STAGE_GATE_PREPARE = 0;
constexpr int64_t KDA_STAGE_POST_WU = 1;
constexpr int64_t KDA_STAGE_FWD_H = 2;
constexpr int64_t KDA_STAGE_FINALIZE = 3;
constexpr int64_t KDA_STAGE_POST_WU_FINALIZE = 4;
constexpr int64_t KDA_STAGE_FINALIZE_WRITEBACK = 5;
constexpr int64_t KDA_STAGE_PREPARE_CUBE = 6;
constexpr int64_t KDA_STAGE_PREPARE_VECTOR_FINALIZE = 7;
constexpr int64_t KDA_STAGE_PREPARE_WU_SCALE = 8;

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchStage(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR gkFp16,
    GM_ADDR betaFp16, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR qgScaled, GM_ADDR uSeed, GM_ADDR scoreScratch,
    GM_ADDR scoreMatrices, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    auto addresses = ResolveAddresses(
        finalState, gk, w, u, qg, kg, vNew, h, userWorkspace, tiling);
    addresses.qgScaled = qgScaled;
    if (tiling.stage == KDA_STAGE_GATE_PREPARE) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)
        RunGateCumsum(g, aLog, dtBias, cuSeqlens, addresses.gk, tiling);
        if (!tiling.computeGateInPrepare) {
            SyncAll<false>();
        }
#endif
        TPipe pipe;
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        KdaPrepare::RunChunkKdaPrepareScoreInputs310P<
            SAFE_GATE, T, T, BETA_T>(
            q, k, v, gkFp16, beta, initialState, cuSeqlens,
            chunkIndices, aqk, akk, addresses.qg, addresses.qgScaled,
            addresses.w, uSeed, scoreScratch, scoreMatrices, userWorkspace,
            tiling, pipe);
#else
        KdaPrepare::RunChunkKdaPrepare<SAFE_GATE, T, float, BETA_T,
            TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, addresses.gk, g, aLog, dtBias, beta, initialState,
            cuSeqlens, chunkIndices, aqk, akk, addresses.qg,
            addresses.qgScaled, addresses.w, uSeed, addresses.kg,
            userWorkspace, tiling, pipe, tiling.storeQG);
#endif
    } else if (tiling.stage == KDA_STAGE_PREPARE_CUBE) {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        TPipe pipe;
        KdaPrepare::RunChunkKdaPrepareScores310P<
            SAFE_GATE, T, T, BETA_T>(
            q, k, v, gkFp16, beta, initialState, cuSeqlens,
            chunkIndices, aqk, akk, addresses.qg, addresses.qgScaled,
            addresses.w, uSeed, scoreScratch, scoreMatrices, userWorkspace,
            tiling, pipe);
#endif
    } else if (tiling.stage == KDA_STAGE_PREPARE_VECTOR_FINALIZE) {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        TPipe pipe;
        KdaPrepare::RunChunkKdaPrepareScoreFinalize310P<
            SAFE_GATE, T, T, BETA_T>(
            q, k, v, gkFp16, beta, initialState, cuSeqlens,
            chunkIndices, aqk, akk, addresses.qg, addresses.qgScaled,
            addresses.w, uSeed, scoreScratch, scoreMatrices, userWorkspace,
            tiling, pipe);
#endif
    } else if (tiling.stage == KDA_STAGE_PREPARE_WU_SCALE) {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        TPipe pipe;
        KdaPrepare::RunChunkKdaPrepareWuScale310P<
            SAFE_GATE, T, T, BETA_T>(
            q, k, v, gkFp16, beta, initialState, cuSeqlens,
            chunkIndices, aqk, akk, addresses.qg, addresses.qgScaled,
            addresses.w, uSeed, scoreScratch, scoreMatrices, userWorkspace,
            tiling, pipe);
#endif
    } else if (tiling.stage == KDA_STAGE_POST_WU) {
        TPipe pipe;
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        // dav-m200 does not support the cross-core events needed to hand Cube
        // output to AIV in one mixed launch. Run Cube here and complete its
        // vector writeback in a separately ordered physical launch.
        KdaPostWu::RunChunkKdaPostWu<
            T, T, T, TilingData, true, false, false>(
            q, k, v, gkFp16, betaFp16, initialState, cuSeqlens,
            chunkIndices, addresses.w, akk, uSeed, addresses.w,
            addresses.u, addresses.kg, addresses.vNew, userWorkspace,
            tiling, pipe);
#else
        KdaPostWu::RunChunkKdaPostWu<T, float, BETA_T>(
            q, k, v, addresses.gk, beta, initialState, cuSeqlens,
            chunkIndices, addresses.w, akk, uSeed, addresses.w,
            addresses.u, addresses.kg, addresses.vNew, userWorkspace,
            tiling, pipe);
#endif
    } else if (tiling.stage == KDA_STAGE_POST_WU_FINALIZE) {
        TPipe pipe;
        KdaPostWu::RunChunkKdaPostWu<
            T, T, T, TilingData, false, true, false>(
            q, k, v, gkFp16, betaFp16, initialState, cuSeqlens,
            chunkIndices, addresses.w, akk, uSeed, addresses.w,
            addresses.u, addresses.kg, addresses.vNew, userWorkspace,
            tiling, pipe);
    } else if (tiling.stage == KDA_STAGE_FWD_H) {
        RunSelectedFwdH<T>(initialState, cuSeqlens, chunkIndices, addresses,
                           userWorkspace, tiling);
    } else if (tiling.stage == KDA_STAGE_FINALIZE) {
        TPipe pipe;
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        KdaFinalize::RunChunkKdaOutput<
            T, T, T, TilingData, true, false, false>(
            q, k, v, gkFp16, betaFp16, initialState, cuSeqlens,
            chunkIndices, addresses.qgScaled, aqk, addresses.vNew,
            addresses.h, attnOut, userWorkspace, tiling, pipe);
#else
        KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
            q, k, v, addresses.gk, beta, initialState, cuSeqlens,
            chunkIndices, addresses.qgScaled, aqk, addresses.vNew,
            addresses.h, attnOut, userWorkspace, tiling, pipe);
#endif
    } else if (tiling.stage == KDA_STAGE_FINALIZE_WRITEBACK) {
        TPipe pipe;
        KdaFinalize::RunChunkKdaOutput<
            T, T, T, TilingData, false, true, false>(
            q, k, v, gkFp16, betaFp16, initialState, cuSeqlens,
            chunkIndices, addresses.qgScaled, aqk, addresses.vNew,
            addresses.h, attnOut, userWorkspace, tiling, pipe);
    }
}

template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchStageSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR gkFp16,
    GM_ADDR betaFp16, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR qgScaled, GM_ADDR uSeed, GM_ADDR scoreScratch,
    GM_ADDR scoreMatrices, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchStage<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, gkFp16, betaFp16, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, qgScaled, uSeed, scoreScratch, scoreMatrices,
            userWorkspace, tiling);
    } else {
        DispatchStage<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, gkFp16, betaFp16, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, qgScaled, uSeed, scoreScratch, scoreMatrices,
            userWorkspace, tiling);
    }
}

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGeneric(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    RunGeneric<SAFE_GATE, T, BETA_T, TilingData,
        COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg, kg,
        vNew, h, userWorkspace, tiling);
}

#if KDA_COMPILE_ARCH35_FAST_PATH
template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    AscendC::TPipe pipe;
    if (tiling.safeGate) {
        arch35::Run<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling, pipe);
    } else {
        arch35::Run<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling, pipe);
    }
}
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR,
    const TilingData &)
{
}
#endif

template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGenericSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchGeneric<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    } else {
        DispatchGeneric<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    }
}

__aicore__ inline void RunKernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR stage_dependency,
    GM_ADDR gk_fp16, GM_ADDR beta_fp16,
    GM_ADDR attn_out,
    GM_ADDR final_state, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR qg_scaled, GM_ADDR u_seed, GM_ADDR score_scratch,
    GM_ADDR score_matrices, GM_ADDR stage_token, GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)stage_dependency;
    (void)stage_token;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(ChunkKdaFwdTilingData, tilingData, tiling);
#if defined(KDA_310P_DEFAULT_TASK) || \
    (defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200))
    // dav-m200 only accepts the default task-type registration, so keyed
    // TILING_KEY_IS branches are not selected at runtime. Dispatch from the
    // runtime dimensions while preserving the specialized GLM/Kimi shape.
    const bool useChunk64K128V128Template =
        tilingData.chunkSize == 64 && tilingData.kHeadDim == 128 &&
        tilingData.vHeadDim == 128;
    if (useChunk64K128V128Template) {
        if (tilingData.stage != KdaForward::KDA_STAGE_FULL) {
            KdaForward::DispatchStageSafeGate<DTYPE_Q, DTYPE_BETA,
                ChunkKdaFwdTilingData, 64, 128, 128>(
                q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
                chunk_indices, gk_fp16, beta_fp16, attn_out, final_state, gk, aqk, akk, w, u,
                qg, kg, v_new, h, qg_scaled, u_seed, score_scratch,
                score_matrices, userWorkspace, tilingData);
            return;
        }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        // The public 310P op always launches physical stages. Do not compile
        // the unsupported mixed-pipeline fallback into the dav-m200 binary.
        return;
#else
        KdaForward::DispatchGenericSafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace, tilingData);
        return;
#endif
    }
    if (tilingData.stage != KdaForward::KDA_STAGE_FULL) {
        KdaForward::DispatchStageSafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 0, 0, 0>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, gk_fp16, beta_fp16, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, qg_scaled, u_seed, score_scratch,
            score_matrices, userWorkspace, tilingData);
        return;
    }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
    return;
#else
    KdaForward::DispatchGenericSafeGate<DTYPE_Q, DTYPE_BETA,
        ChunkKdaFwdTilingData, 0, 0, 0>(
        q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
        chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg, kg,
        v_new, h, userWorkspace, tilingData);
    return;
#endif
#else
    if (TILING_KEY_IS(1)) {
        if (tilingData.stage != KdaForward::KDA_STAGE_FULL) {
            KdaForward::DispatchStageSafeGate<DTYPE_Q, DTYPE_BETA,
                ChunkKdaFwdTilingData, 0, 0, 0>(
                q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
                chunk_indices, gk_fp16, beta_fp16, attn_out, final_state, gk, aqk, akk, w, u,
                qg, kg, v_new, h, qg_scaled, u_seed, score_scratch,
                score_matrices, userWorkspace, tilingData);
            return;
        }
        KdaForward::DispatchGenericSafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 0, 0, 0>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace, tilingData);
    } else if (TILING_KEY_IS(2)) {
        if (tilingData.stage != KdaForward::KDA_STAGE_FULL) {
            KdaForward::DispatchStageSafeGate<DTYPE_Q, DTYPE_BETA,
                ChunkKdaFwdTilingData, 64, 128, 128>(
                q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
                chunk_indices, gk_fp16, beta_fp16, attn_out, final_state, gk, aqk, akk, w, u,
                qg, kg, v_new, h, qg_scaled, u_seed, score_scratch,
                score_matrices, userWorkspace, tilingData);
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaForward::DispatchArch35SafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
#else
        KdaForward::DispatchGenericSafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
#endif
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace, tilingData);
    }
#endif
}

} // namespace KdaForward

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR stage_dependency,
    GM_ADDR gk_fp16, GM_ADDR beta_fp16,
    GM_ADDR attn_out,
    GM_ADDR final_state, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR qg_scaled, GM_ADDR u_seed, GM_ADDR score_scratch,
    GM_ADDR score_matrices, GM_ADDR stage_token, GM_ADDR workspace,
    GM_ADDR tiling)
{
#if defined(KDA_310P_DEFAULT_TASK) || \
    (defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200))
    // dav-m200 executes Cube and vector instructions from one unified image.
    // The separately compiled dav-m200-vec image is not launched by the 310P
    // runtime for this task type.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AICORE);
#else
    // Newer split-core architectures use their paired AIC/AIV ABI.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    KdaForward::RunKernel(
        q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
        chunk_indices, stage_dependency, gk_fp16, beta_fp16, attn_out, final_state, gk, aqk,
        akk, w, u, qg, kg,
        v_new, h, qg_scaled, u_seed, score_scratch, score_matrices,
        stage_token, workspace, tiling);
}

#undef KDA_COMPILE_ARCH35_FAST_PATH
