// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t GROUP = 128, BATCH = 8, LANES = 8, HALF_GROUP = GROUP / 2;
constexpr uint32_t ELEMENTS = GROUP * BATCH, STORAGE_BYTES = 32 * 1024;

struct SwigluPackTiling {
  int64_t rows;
  int64_t groups_per_row;
};

class SwigluPackActivation {
 public:
  __aicore__ inline void Run(GM_ADDR gate_up, GM_ADDR low, GM_ADDR high, GM_ADDR scale, GM_ADDR sum,
                             int64_t rows, int64_t groupsPerRow) {
    gateUp_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gate_up));
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    sum_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
    pipe_.InitBuffer(storage_, STORAGE_BYTES);

    // The first 4 KiB holds two FP16 inputs. Remaining offsets deliberately
    // mirror qwen_w4_a8_pack_v310 so its quantization and limb packing stay
    // byte-identical after the explicit FP32-SwiGLU -> FP16 rounding point.
    auto gateInput = storage_.Get<half>();
    auto upInput = gateInput[ELEMENTS];
    auto values = storage_.Get<float>()[ELEMENTS];
    auto temporary = storage_.Get<float>()[2 * ELEMENTS];
    auto quant = storage_.Get<float>()[3 * ELEMENTS];
    auto integers = storage_.Get<int32_t>()[4 * ELEMENTS];
    auto limb = storage_.Get<half>()[10 * ELEMENTS];
    auto packed = storage_.Get<int8_t>()[24 * ELEMENTS];
    auto reduced = storage_.Get<float>()[7 * ELEMENTS];
    auto pairs = reduced[HALF_GROUP * BATCH];
    auto maxima = pairs[2 * BATCH];
    auto scales = maxima[LANES];
    auto totals = scales[LANES];
    auto ones = totals[LANES];
    auto broadcast = ones[LANES];
    auto sumBroadcast = broadcast[LANES * BATCH];
    auto indices = sumBroadcast[LANES * BATCH].ReinterpretCast<uint32_t>();
    for (uint32_t i = 0; i < BATCH; ++i) indices.SetValue(i, i * 2 * sizeof(float));
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);

    const int64_t batchesPerRow = (groupsPerRow + BATCH - 1) / BATCH;
    const int64_t batchCount = rows * batchesPerRow;
    const int64_t activationWidth = groupsPerRow * GROUP;
    const int64_t gateUpWidth = 2 * activationWidth;
    for (int64_t batch = GetBlockIdx(); batch < batchCount; batch += GetBlockNum()) {
      const int64_t row = batch / batchesPerRow;
      const int64_t firstGroupInRow = (batch - row * batchesPerRow) * BATCH;
      const uint32_t count = groupsPerRow - firstGroupInRow < BATCH ? groupsPerRow - firstGroupInRow : BATCH;
      const int64_t first = row * groupsPerRow + firstGroupInRow;
      const int64_t gateOffset = row * gateUpWidth + firstGroupInRow * GROUP;
      const int64_t upOffset = gateOffset + activationWidth;

      Duplicate(gateInput, static_cast<half>(0), ELEMENTS);
      Duplicate(upInput, static_cast<half>(0), ELEMENTS);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      DataCopy(gateInput, gateUp_[gateOffset], count * GROUP);
      DataCopy(upInput, gateUp_[upOffset], count * GROUP);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

      // Match the current Python path: gate/up FP16 -> FP32 SwiGLU -> FP16.
      Cast(values, gateInput, RoundMode::CAST_NONE, ELEMENTS);
      Cast(temporary, upInput, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Muls(quant, values, -1.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Exp(quant, quant, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Adds(quant, quant, 1.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Div(quant, values, quant, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Mul(values, quant, temporary, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(gateInput, values, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(values, gateInput, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();

      Abs(temporary, values, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Max(reduced, temporary, temporary[HALF_GROUP], HALF_GROUP, BATCH, {1, 1, 1, 8, 16, 16});
      PipeBarrier<PIPE_V>();
      WholeReduceMax(pairs, reduced, HALF_GROUP, BATCH, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
      PipeBarrier<PIPE_V>();
      Gather(maxima, pairs, indices, static_cast<uint32_t>(0), LANES);
      Duplicate(ones, 127.0f, LANES);
      PipeBarrier<PIPE_V>();
      Div(scales, maxima, ones, LANES);
      PipeBarrier<PIPE_V>();
      constexpr float FP16_MIN_MAGNITUDE_RECIPROCAL = 16777216.0f;
      Muls(ones, maxima, FP16_MIN_MAGNITUDE_RECIPROCAL, LANES);
      PipeBarrier<PIPE_V>();
      Mins(ones, ones, 1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Muls(ones, ones, -1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Adds(ones, ones, 1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Add(scales, scales, ones, LANES);
      PipeBarrier<PIPE_V>();
      Brcb(broadcast, scales, 1, {1, 8});
      PipeBarrier<PIPE_V>();
      for (uint32_t halfGroup = 0; halfGroup < 2; ++halfGroup) {
        Div(quant[halfGroup * HALF_GROUP], values[halfGroup * HALF_GROUP], broadcast, HALF_GROUP, BATCH,
            {1, 1, 0, 16, 16, 1});
      }
      PipeBarrier<PIPE_V>();
      Cast(integers, quant, RoundMode::CAST_RINT, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(quant, integers, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Mins(quant, quant, 127.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Maxs(quant, quant, -127.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Add(reduced, quant, quant[HALF_GROUP], HALF_GROUP, BATCH, {1, 1, 1, 8, 16, 16});
      PipeBarrier<PIPE_V>();
      WholeReduceSum(totals, reduced, HALF_GROUP, BATCH, 1, 1, 8);
      Muls(temporary, quant, 1.0f / 16.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Brcb(sumBroadcast, totals, 1, {1, 8});
      Cast(integers, temporary, RoundMode::CAST_FLOOR, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(temporary, integers, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(limb, temporary, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(packed.ReinterpretCast<int4b_t>(), limb, RoundMode::CAST_NONE, ELEMENTS);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(high_[first * GROUP / 2], packed, count * GROUP / 2);
      Muls(temporary, temporary, -16.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Add(temporary, quant, temporary, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Adds(temporary, temporary, -8.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(limb, temporary, RoundMode::CAST_NONE, ELEMENTS);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      Cast(packed.ReinterpretCast<int4b_t>(), limb, RoundMode::CAST_NONE, ELEMENTS);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(low_[first * GROUP / 2], packed, count * GROUP / 2);
      DataCopy(scale_[first * LANES], broadcast, count * LANES);
      DataCopy(sum_[first * LANES], sumBroadcast, count * LANES);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<half> gateUp_;
  GlobalTensor<int8_t> low_, high_;
  GlobalTensor<float> scale_, sum_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_w4_a8_swiglu_pack_v310(
    GM_ADDR gate_up, GM_ADDR low, GM_ADDR high, GM_ADDR scale, GM_ADDR sum, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  auto td = reinterpret_cast<__gm__ SwigluPackTiling*>(tiling);
  SwigluPackActivation op;
  op.Run(gate_up, low, high, scale, sum, td->rows, td->groups_per_row);
}
