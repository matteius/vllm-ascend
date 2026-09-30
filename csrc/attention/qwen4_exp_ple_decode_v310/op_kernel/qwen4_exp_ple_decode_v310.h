#ifndef QWEN4EXP_PLE_DECODE_V310_H
#define QWEN4EXP_PLE_DECODE_V310_H

#include "kernel_operator.h"
#include "qwen4_exp_ple_decode_v310_tiling_data.h"

namespace NsQwen4ExpPleDecode {

using namespace AscendC;

constexpr uint32_t REDUCE_LANES = 8;
constexpr uint32_t FP32_REDUCE_WIDTH = 64;
constexpr float GATE_MAGNITUDE_FLOOR = 1e-6f;

class Qwen4ExpPleDecodeV310 {
public:
    __aicore__ inline void Init(
        GM_ADDR projected, GM_ADDR hidden,
        GM_ADDR normKeyWeight, GM_ADDR normQueryWeight,
        GM_ADDR normConvWeight, GM_ADDR currentConvWeight,
        GM_ADDR output,
        const Qwen4ExpPleDecodeV310TilingData *tiling,
        TPipe *pipe)
    {
        numTokens_ = tiling->numTokens;
        hcGroups_ = tiling->hcGroups;
        groupSize_ = tiling->groupSize;
        tasksPerCore_ = tiling->tasksPerCore;
        reduceWidth_ = tiling->reduceWidth;
        normEps_ = tiling->normEps;
        invGroupSize_ = tiling->invGroupSize;
        invSqrtGroupSize_ = tiling->invSqrtGroupSize;
        projectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(projected));
        hiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(hidden));
        normKeyWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(normKeyWeight));
        normQueryWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(normQueryWeight));
        normConvWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(normConvWeight));
        currentConvWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(currentConvWeight));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
        pipe_ = pipe;
        pipe_->InitBuffer(rawHalfBuf_, groupSize_ * sizeof(half));
        pipe_->InitBuffer(outputHalfBuf_, groupSize_ * sizeof(half));
        pipe_->InitBuffer(keyFloatBuf_, groupSize_ * sizeof(float));
        pipe_->InitBuffer(queryFloatBuf_, groupSize_ * sizeof(float));
        pipe_->InitBuffer(workFloatBuf_, groupSize_ * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, REDUCE_LANES * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t firstTask = GetBlockIdx() * tasksPerCore_;
        const int64_t totalTasks = numTokens_ * hcGroups_;
        for (int64_t offset = 0; offset < tasksPerCore_; ++offset) {
            const int64_t task = firstTask + offset;
            if (task >= totalTasks) {
                break;
            }
            ProcessGroup(task);
        }
    }

private:
    __aicore__ inline float ReduceSum(LocalTensor<float> values)
    {
        LocalTensor<float> reduce = reduceBuf_.Get<float>();
        const int64_t tail = groupSize_ - reduceWidth_;
        if (tail > 0) {
            Add(values, values, values[reduceWidth_], tail);
            PipeBarrier<PIPE_V>();
        }
        int64_t width = reduceWidth_;
        while (width > FP32_REDUCE_WIDTH) {
            width /= 2;
            Add(values, values, values[width], width);
            PipeBarrier<PIPE_V>();
        }
        WholeReduceSum(reduce, values, static_cast<uint32_t>(width), 1, 1, 1, REDUCE_LANES);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return reduce.GetValue(0);
    }

    __aicore__ inline float InverseRoot(float value)
    {
        LocalTensor<float> reduce = reduceBuf_.Get<float>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Duplicate(reduce, value, REDUCE_LANES);
        PipeBarrier<PIPE_V>();
        Sqrt(reduce, reduce, REDUCE_LANES);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return 1.0f / reduce.GetValue(0);
    }

    __aicore__ inline void NormalizeLocal(
        LocalTensor<half> source,
        GlobalTensor<half> weight,
        int64_t weightOffset,
        LocalTensor<float> normalized)
    {
        LocalTensor<half> rawHalf = rawHalfBuf_.Get<half>();
        LocalTensor<float> work = workFloatBuf_.Get<float>();
        Cast(normalized, source, RoundMode::CAST_NONE, groupSize_);
        PipeBarrier<PIPE_V>();
        Mul(work, normalized, normalized, groupSize_);
        PipeBarrier<PIPE_V>();
        const float inverseRoot = InverseRoot(
            ReduceSum(work) * invGroupSize_ + normEps_);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(normalized, normalized, inverseRoot, groupSize_);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopy(rawHalf, weight[weightOffset], groupSize_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Cast(work, rawHalf, RoundMode::CAST_NONE, groupSize_);
        PipeBarrier<PIPE_V>();
        Adds(work, work, 1.0f, groupSize_);
        PipeBarrier<PIPE_V>();
        Mul(normalized, normalized, work, groupSize_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Normalize(
        GlobalTensor<half> source,
        GlobalTensor<half> weight,
        int64_t sourceOffset,
        int64_t weightOffset,
        LocalTensor<float> normalized)
    {
        LocalTensor<half> rawHalf = rawHalfBuf_.Get<half>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopy(rawHalf, source[sourceOffset], groupSize_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        NormalizeLocal(rawHalf, weight, weightOffset, normalized);
    }

    __aicore__ inline float Gate(LocalTensor<float> key,
                                 LocalTensor<float> query)
    {
        LocalTensor<float> work = workFloatBuf_.Get<float>();
        LocalTensor<float> reduce = reduceBuf_.Get<float>();
        Mul(work, key, query, groupSize_);
        PipeBarrier<PIPE_V>();
        const float dot = ReduceSum(work) * invSqrtGroupSize_;
        float magnitudeInput = dot < 0.0f ? -dot : dot;
        if (magnitudeInput < GATE_MAGNITUDE_FLOOR) {
            magnitudeInput = GATE_MAGNITUDE_FLOOR;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Duplicate(reduce, magnitudeInput, REDUCE_LANES);
        PipeBarrier<PIPE_V>();
        Sqrt(reduce, reduce, REDUCE_LANES);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float signedMagnitude = reduce.GetValue(0);
        if (dot < 0.0f) {
            signedMagnitude = -signedMagnitude;
        } else if (dot == 0.0f) {
            signedMagnitude = 0.0f;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Duplicate(reduce, -signedMagnitude, REDUCE_LANES);
        PipeBarrier<PIPE_V>();
        Exp(reduce, reduce, REDUCE_LANES);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return 1.0f / (1.0f + reduce.GetValue(0));
    }

    __aicore__ inline void NormalizeGated(
        LocalTensor<float> gated,
        int64_t weightOffset,
        LocalTensor<float> normalized)
    {
        LocalTensor<half> rawHalf = rawHalfBuf_.Get<half>();
        LocalTensor<float> work = workFloatBuf_.Get<float>();
        Mul(work, gated, gated, groupSize_);
        PipeBarrier<PIPE_V>();
        const float inverseRoot = InverseRoot(
            ReduceSum(work) * invGroupSize_ + normEps_);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(normalized, gated, inverseRoot, groupSize_);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopy(rawHalf, normConvWeightGm_[weightOffset], groupSize_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Cast(work, rawHalf, RoundMode::CAST_NONE, groupSize_);
        PipeBarrier<PIPE_V>();
        Adds(work, work, 1.0f, groupSize_);
        PipeBarrier<PIPE_V>();
        Mul(normalized, normalized, work, groupSize_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessGroup(int64_t task)
    {
        const int64_t token = task / hcGroups_;
        const int64_t group = task % hcGroups_;
        const int64_t activationOffset = task * groupSize_;
        const int64_t weightOffset = group * groupSize_;
        const int64_t projectedRowOffset = token * (hcGroups_ + 1) * groupSize_;
        const int64_t keyOffset = projectedRowOffset + group * groupSize_;
        const int64_t valueOffset = projectedRowOffset + hcGroups_ * groupSize_;
        LocalTensor<half> rawHalf = rawHalfBuf_.Get<half>();
        LocalTensor<half> outputHalf = outputHalfBuf_.Get<half>();
        LocalTensor<float> key = keyFloatBuf_.Get<float>();
        LocalTensor<float> query = queryFloatBuf_.Get<float>();
        LocalTensor<float> work = workFloatBuf_.Get<float>();

        Normalize(projectedGm_, normKeyWeightGm_, keyOffset, weightOffset, key);
        DataCopy(outputHalf, hiddenGm_[activationOffset], groupSize_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        NormalizeLocal(outputHalf, normQueryWeightGm_, weightOffset, query);
        const float gate = Gate(key, query);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopy(rawHalf, projectedGm_[valueOffset], groupSize_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Cast(key, rawHalf, RoundMode::CAST_NONE, groupSize_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(key, key, gate, groupSize_);
        PipeBarrier<PIPE_V>();
        NormalizeGated(key, weightOffset, query);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopy(work, currentConvWeightGm_[weightOffset], groupSize_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Mul(query, query, work, groupSize_);
        PipeBarrier<PIPE_V>();
        Silu(work, query, groupSize_);
        PipeBarrier<PIPE_V>();
        Add(query, key, work, groupSize_);
        PipeBarrier<PIPE_V>();

        Cast(work, outputHalf, RoundMode::CAST_NONE, groupSize_);
        PipeBarrier<PIPE_V>();
        Add(query, query, work, groupSize_);
        PipeBarrier<PIPE_V>();
        Cast(outputHalf, query, RoundMode::CAST_NONE, groupSize_);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm_[activationOffset], outputHalf, groupSize_);
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> rawHalfBuf_;
    TBuf<TPosition::VECCALC> outputHalfBuf_;
    TBuf<TPosition::VECCALC> keyFloatBuf_;
    TBuf<TPosition::VECCALC> queryFloatBuf_;
    TBuf<TPosition::VECCALC> workFloatBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    GlobalTensor<half> projectedGm_;
    GlobalTensor<half> hiddenGm_;
    GlobalTensor<half> normKeyWeightGm_;
    GlobalTensor<half> normQueryWeightGm_;
    GlobalTensor<half> normConvWeightGm_;
    GlobalTensor<float> currentConvWeightGm_;
    GlobalTensor<half> outputGm_;
    int64_t numTokens_ = 0;
    int64_t hcGroups_ = 0;
    int64_t groupSize_ = 0;
    int64_t tasksPerCore_ = 0;
    int64_t reduceWidth_ = 0;
    float normEps_ = 0.0f;
    float invGroupSize_ = 0.0f;
    float invSqrtGroupSize_ = 0.0f;
};

}  // namespace NsQwen4ExpPleDecode

#endif
