/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qwen_w4_group_matmul_v310.h
 * \brief 310P Qwen signed-W4 asymmetric group-128 Cube matmul.
 *
 * The checkpoint remains byte-packed in canonical row-major order. The tiled
 * backend losslessly re-encodes nibbles during loading into Cube NZ order,
 * avoiding runtime weight/metadata gathers. The canonical backend instead
 * gathers each decoded 16-output-channel plane into NZ. Each AI core decodes
 * only its current 32-output-channel tile into a per-output-tile workspace.
 * CATLASS therefore consumes an already-NZ B operand:
 * temporary scratch totals [N,K] FP16 for one invocation, but no persistent
 * dequantized expert bank or separate ND-to-NZ conversion is retained.
 */

#ifndef QWEN_W4_GROUP_MATMUL_V310_H
#define QWEN_W4_GROUP_MATMUL_V310_H

#define CATLASS_ARCH 2201
#define CATLASS_UNIFIED_CORE 1

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "kernel_operator.h"

namespace NsQwenW4 {

using namespace AscendC;
using namespace Catlass;
using namespace tla;

constexpr int64_t QW4_GROUP_SIZE = 128;
constexpr uint32_t QW4_FRACTAL_SIZE = 16;
constexpr uint32_t QW4_TILE_N = 32;
constexpr uint32_t QW4_TILE_K = 128;
constexpr uint32_t QW4_TILED_K_BATCH = 512;
constexpr uint32_t QW4_MAX_TILED_K_BATCH = 640;
constexpr uint32_t QW4_HALF_VECTOR_ELEMENTS = 128;
constexpr uint32_t QW4_VECTOR_BLOCKS = 8;
constexpr uint32_t QW4_K_FRACTALS_PER_TILE = QW4_TILE_K / QW4_FRACTAL_SIZE;
// For every byte b, RINT(b/16 - 15/32) == floor(b/16): the fractional
// range is [-15/32,15/32], so no rounding ties occur. All operands are
// exact in FP16. Unlike FLOOR, RINT half->int16 is supported on 310P.
constexpr float QW4_HIGH_NIBBLE_ROUND_BIAS = -15.0f / 32.0f;

template <typename T>
__aicore__ inline T CeilDivU(T a, T b) {
  return (b == 0) ? 0 : (a + b - 1) / b;
}
template <typename T>
__aicore__ inline T AlignUpU(T a, T b) {
  return (b == 0) ? 0 : (a + b - 1) / b * b;
}
template <typename T>
__aicore__ inline T MinU(T a, T b) {
  return (a < b) ? a : b;
}

class QwenW4GroupMatmulV310Cube {
 public:
  using ArchTag = Arch::AtlasA2;
  using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
  using L1TileShapeTla = tla::Shape<tla::Int<128>, tla::Int<QW4_TILE_N>, tla::Int<128>>;
  using L0TileShapeTla = L1TileShapeTla;
  using TileCopy =
      Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, half, layout::RowMajor, half, layout::zN, half, layout::RowMajor>;
  using BlockMmad =
      Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, half, half, half, void, TileCopy>;

  __aicore__ inline QwenW4GroupMatmulV310Cube() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR codes, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR user,
                              GM_ADDR tiling) {
    __gm__ QwenW4GroupMatmulTilingData* __restrict td =
        reinterpret_cast<__gm__ QwenW4GroupMatmulTilingData* __restrict>(tiling);
    InitGeometry(x, codes, scale, offset, y, user, td->numTokens, td->nDim, td->kDim, td->tiled != 0);
  }

  __aicore__ inline void InitGeometry(GM_ADDR x, GM_ADDR codes, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR user,
                                      int64_t tokens, int64_t outputs, int64_t inputs, bool tiled) {
    T_ = tokens;
    N_ = outputs;
    K_ = inputs;
    codesPerByte_ = 2;
    tiled_ = tiled;
    tileId_ = GetBlockIdx();
    tileCount_ = GetBlockNum();
    tileK_ = QW4_TILE_K;
    if (tiled_) {
      tileK_ = K_ <= QW4_MAX_TILED_K_BATCH ? K_ : QW4_TILED_K_BATCH;
      while (K_ % tileK_ != 0) {
        tileK_ /= 2;
      }
    }
    bitsPerCode_ = 8 / codesPerByte_;
    packedK_ = K_ / codesPerByte_;
    packedTileCols_ = tileK_ / codesPerByte_;
    packedTileCount_ = (tiled_ ? QW4_TILE_N : QW4_FRACTAL_SIZE) * packedTileCols_;
    decodedTileCount_ = codesPerByte_ * packedTileCount_;
    fieldMask_ = (1 << bitsPerCode_) - 1;
    signHalf_ = 1 << (bitsPerCode_ - 1);
    kbCount_ = K_ / QW4_GROUP_SIZE;

    const int64_t nzElementsPerCore = QW4_TILE_N * K_;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    codesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(codes));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scale));
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(offset));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    wdqNzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(user));

    coreNzBase_ = static_cast<int64_t>(GetBlockIdx()) * nzElementsPerCore;
  }

  __aicore__ inline void SetTileRange(uint32_t tileId, uint32_t tileCount) {
    tileId_ = tileId;
    tileCount_ = tileCount;
  }

  __aicore__ inline void Process() {
    const uint32_t coreId = tileId_;
    const uint32_t coreNum = tileCount_;
    const uint32_t nBlocks = CeilDivU<uint32_t>((uint32_t)N_, QW4_TILE_N);

    AllocBuffers();
    FillTables();

    auto aLayout = tla::MakeLayout<half, layout::RowMajor>((uint32_t)T_, (uint32_t)K_);
    auto bLayout = tla::MakeLayout<half, layout::zN>((uint32_t)K_, QW4_TILE_N);
    auto cLayout = tla::MakeLayout<half, layout::RowMajor>((uint32_t)T_, (uint32_t)N_);
    auto tensorA = tla::MakeTensor(xGm_, aLayout, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(wdqNzGm_[coreNzBase_], bLayout, Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(yGm_, cLayout, Arch::PositionGM{});

    for (uint32_t nb = coreId; nb < nBlocks; nb += coreNum) {
      const uint32_t n0 = nb * QW4_TILE_N;
      const uint32_t nActual = MinU<uint32_t>(QW4_TILE_N, (uint32_t)N_ - n0);

      DequantTileToNz(n0, nActual);
      // The dequantizer writes this core's packed-NZ workspace through
      // MTE3, while BlockMmad consumes it from GM through MTE2.  A pipe
      // barrier does not order those engines on 310P.
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);

      GemmCoord shape{(uint32_t)T_, nActual, (uint32_t)K_};
      auto tA = GetTile(tensorA, tla::MakeCoord((uint32_t)0, (uint32_t)0), tla::MakeShape((uint32_t)T_, (uint32_t)K_));
      auto tB = GetTile(tensorB, tla::MakeCoord((uint32_t)0, (uint32_t)0), tla::MakeShape((uint32_t)K_, nActual));
      auto tC = GetTile(tensorC, tla::MakeCoord((uint32_t)0, n0), tla::MakeShape((uint32_t)T_, nActual));
      BlockMmad blockMmad(resource);
      blockMmad.preSetFlags();
      blockMmad(tA, tB, tC, shape);
      blockMmad.finalWaitFlags();
      // A core can process several N tiles and reuse the same GM
      // workspace.  Finish BlockMmad's MTE2 reads before the next tile
      // overwrites that workspace through MTE3.
      SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID4);
      WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID4);
    }
  }

 private:
  __aicore__ inline void AllocBuffers() {
    uint32_t off = 0;
    scaleUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
    off = AlignUpU<uint32_t>(off + QW4_TILE_N * (uint32_t)kbCount_ * sizeof(half), 512);
    offsetUB_ = resource.ubBuf.template GetBufferByByte<int8_t>(off);
    off = AlignUpU<uint32_t>(off + QW4_TILE_N * (uint32_t)kbCount_, 512);
    offsetHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
    off = AlignUpU<uint32_t>(off + QW4_TILE_N * (uint32_t)kbCount_ * sizeof(half), 512);
    scaleVecUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(half), 512);
    offsetVecUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(half), 512);
    cU8_ = resource.ubBuf.template GetBufferByByte<uint8_t>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(uint8_t), 512);
    cH_ = resource.ubBuf.template GetBufferByByte<half>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(half), 512);
    c16_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(int16_t), 512);
    fieldHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(half), 512);
    signedHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(half), 512);
    if (tiled_) {
      return;
    }
    andTmp_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(int16_t), 512);
    fieldI16UB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(int16_t), 512);
    nzTileUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(half), 512);
    gatherOffsetsUB_ = resource.ubBuf.template GetBufferByByte<uint32_t>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(uint32_t), 512);
    gatherIndexFloatUB_ = resource.ubBuf.template GetBufferByByte<float>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(float), 512);
    threeUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(int16_t), 512);
    signHalfUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(int16_t), 512);
    masksUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
    off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(int16_t), 512);
  }

  __aicore__ inline void FillTables() {
    if (tiled_) {
      // Load-time NZ encoding and vector metadata broadcast need no
      // runtime index construction, weight transpose, or Gather.
      return;
    }
    if (!tiled_) {
      Duplicate(threeUB_, static_cast<int16_t>(fieldMask_), (int32_t)decodedTileCount_);
      Duplicate(signHalfUB_, static_cast<int16_t>(signHalf_), (int32_t)decodedTileCount_);
      for (int64_t field = 0; field < codesPerByte_; ++field) {
        Duplicate(masksUB_[field * packedTileCount_], static_cast<int16_t>(fieldMask_ << (bitsPerCode_ * field)),
                  (int32_t)packedTileCount_);
      }
    }

    // The vector unpack is field-major across the complete packed tile.
    // Gather directly into eight [K=16,N=16] NZ fragments of W^T. This
    // includes the transpose in the permutation and permits one contiguous
    // DMA per tile instead of eight Transpose/DMA/synchronization pairs.
    // Only the first K row needs scalar stores. All other rows/fractals
    // repeat that pattern with a constant byte displacement.
    for (uint32_t row = 0; row < QW4_FRACTAL_SIZE; ++row) {
      gatherIndexFloatUB_.SetValue(row, static_cast<float>(static_cast<int32_t>(row * QW4_TILE_K)));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID3);
    WaitFlag<HardEvent::S_V>(EVENT_ID3);
    for (uint32_t k = 1; k < QW4_FRACTAL_SIZE; ++k) {
      const uint32_t byteOffset = ((k % 2) * packedTileCount_ + k / 2) * sizeof(half);
      Adds(gatherIndexFloatUB_[k * QW4_FRACTAL_SIZE], gatherIndexFloatUB_,
           static_cast<float>(static_cast<int32_t>(byteOffset)), QW4_FRACTAL_SIZE);
    }
    PipeBarrier<PIPE_V>();
    constexpr uint32_t fractalElements = QW4_FRACTAL_SIZE * QW4_FRACTAL_SIZE;
    for (uint32_t fractal = 1; fractal < QW4_K_FRACTALS_PER_TILE; ++fractal) {
      Adds(gatherIndexFloatUB_[fractal * fractalElements], gatherIndexFloatUB_,
           static_cast<float>(static_cast<int32_t>(fractal * QW4_FRACTAL_SIZE)), fractalElements);
    }
    PipeBarrier<PIPE_V>();
    Cast(gatherOffsetsUB_.ReinterpretCast<int32_t>(), gatherIndexFloatUB_, RoundMode::CAST_RINT,
         (int32_t)decodedTileCount_);
    PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void DecodeTile(int64_t codeOffset, uint32_t rowBase, int64_t group) {
    // The next bulk DMA can otherwise overwrite cU8 while the vector
    // engine is still reading the previous tile. Scalar per-row DMA made
    // this race less visible, but did not provide a pipeline dependency.
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    // Copy one packed row at a time. A strided 2-D GM-to-UB DataCopy looks
    // attractive here, but dav_m200 rejects that descriptor at runtime
    // with an MTE "burst num" exception even when every row and stride is
    // 32-byte aligned. The scalar overload is hardware-proven on 310P and
    // still batches all 128 logical K values for the vector decode below.
    if (tiled_) {
      DataCopy(cU8_, codesGm_[codeOffset], static_cast<int32_t>(packedTileCount_));
    } else {
      for (uint32_t row = 0; row < QW4_FRACTAL_SIZE; ++row) {
        DataCopy(cU8_[row * packedTileCols_], codesGm_[codeOffset + static_cast<int64_t>(row) * packedK_],
                 static_cast<int32_t>(packedTileCols_));
      }
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(cH_, cU8_, RoundMode::CAST_NONE, (int32_t)packedTileCount_);
    PipeBarrier<PIPE_V>();
    if (tiled_) {
      // Load-time sign-bit toggling stores q+8 in each nibble. Extract
      // unsigned high/low with exact FP16 arithmetic, avoiding two
      // bit-mask/cast chains and runtime sign extension.
      auto high = signedHalfUB_[packedTileCount_];
      Muls(fieldHalfUB_, cH_, static_cast<half>(1.0f / 16.0f), (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();
      Adds(fieldHalfUB_, fieldHalfUB_, static_cast<half>(QW4_HIGH_NIBBLE_ROUND_BIAS), (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();
      Cast(c16_, fieldHalfUB_, RoundMode::CAST_RINT, (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();
      Cast(high, c16_, RoundMode::CAST_NONE, (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();
      Muls(fieldHalfUB_, high, static_cast<half>(16), (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();
      Sub(signedHalfUB_, cH_, fieldHalfUB_, (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();
    } else {
      Cast(c16_, cH_, RoundMode::CAST_RINT, (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();

      for (int32_t field = 0; field < codesPerByte_; ++field) {
        And(andTmp_, c16_, masksUB_[field * packedTileCount_], (int32_t)packedTileCount_);
        PipeBarrier<PIPE_V>();
        Cast(fieldHalfUB_, andTmp_, RoundMode::CAST_NONE, (int32_t)packedTileCount_);
        PipeBarrier<PIPE_V>();
        const half fieldRecip = static_cast<half>(1.0f / static_cast<float>(1 << (bitsPerCode_ * field)));
        Muls(fieldHalfUB_, fieldHalfUB_, fieldRecip, (int32_t)packedTileCount_);
        PipeBarrier<PIPE_V>();
        Cast(fieldI16UB_[field * packedTileCount_], fieldHalfUB_, RoundMode::CAST_RINT, (int32_t)packedTileCount_);
        PipeBarrier<PIPE_V>();
      }

      // Sign-extend the two's-complement W4 field in int16.
      Add(fieldI16UB_, fieldI16UB_, signHalfUB_, (int32_t)decodedTileCount_);
      PipeBarrier<PIPE_V>();
      And(fieldI16UB_, fieldI16UB_, threeUB_, (int32_t)decodedTileCount_);
      PipeBarrier<PIPE_V>();
      Sub(fieldI16UB_, fieldI16UB_, signHalfUB_, (int32_t)decodedTileCount_);
      PipeBarrier<PIPE_V>();
      Cast(signedHalfUB_, fieldI16UB_, RoundMode::CAST_NONE, (int32_t)decodedTileCount_);
      PipeBarrier<PIPE_V>();
    }

    if (tiled_) {
      constexpr uint32_t groupElements = QW4_GROUP_SIZE * QW4_FRACTAL_SIZE;
      constexpr uint8_t repeats = groupElements / QW4_HALF_VECTOR_ELEMENTS;
      const BinaryRepeatParams broadcast(1, 1, 0, QW4_VECTOR_BLOCKS, QW4_VECTOR_BLOCKS, 0);
      for (int32_t field = 0; field < codesPerByte_; ++field) {
        for (int32_t groupInTile = 0; groupInTile < tileK_ / QW4_GROUP_SIZE; ++groupInTile) {
          auto values = signedHalfUB_[field * packedTileCount_ + groupInTile * groupElements];
          const int64_t base = (group + groupInTile) * QW4_TILE_N + field * QW4_FRACTAL_SIZE;
          // A 32-byte metadata block holds all 16 output channels.
          // Zero src1 block/repeat strides broadcast it across K using
          // vsub/vmul directly, with no expanded coefficients or Gather.
          Sub(values, values, offsetHalfUB_[base], QW4_HALF_VECTOR_ELEMENTS, repeats, broadcast);
          PipeBarrier<PIPE_V>();
          Mul(values, values, scaleUB_[base], QW4_HALF_VECTOR_ELEMENTS, repeats, broadcast);
          PipeBarrier<PIPE_V>();
        }
      }
      return;
    }

    // One scale and signed offset per output row and contiguous K=128
    // group. The two unpacked nibble planes share these coefficients.
    // q-offset is exact in FP16; multiply rounds once, matching the
    // reference's FP32 dequantization followed by an FP16 weight cast.
    if (!tiled_) {
      for (uint32_t row = 0; row < QW4_FRACTAL_SIZE; ++row) {
        const int64_t index = (rowBase + row) * kbCount_ + group;
        Duplicate(scaleVecUB_[row * packedTileCols_], scaleUB_.GetValue(index), (int32_t)packedTileCols_);
        Duplicate(offsetVecUB_[row * packedTileCols_], static_cast<half>(offsetUB_.GetValue(index)),
                  (int32_t)packedTileCols_);
      }
    }
    PipeBarrier<PIPE_V>();
    for (int32_t field = 0; field < codesPerByte_; ++field) {
      auto values = signedHalfUB_[field * packedTileCount_];
      Sub(values, values, offsetVecUB_, (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();
      Mul(values, values, scaleVecUB_, (int32_t)packedTileCount_);
      PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();

    if (!tiled_) {
      Gather(nzTileUB_, signedHalfUB_, gatherOffsetsUB_, (uint32_t)0, (uint32_t)decodedTileCount_);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void DequantTileToNz(uint32_t n0, uint32_t nActual) {
    // Copy all 32 rows together: even K=640's five INT8 offsets per
    // row give aligned, exact-length DMAs with no last-row overread.
    const int32_t metadataCount = nActual * kbCount_;
    DataCopy(scaleUB_, scaleGm_[static_cast<int64_t>(n0) * kbCount_], metadataCount);
    DataCopy(offsetUB_, offsetGm_[static_cast<int64_t>(n0) * kbCount_], metadataCount);
    if (tiled_) {
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(offsetHalfUB_, offsetUB_, RoundMode::CAST_NONE, metadataCount);
      PipeBarrier<PIPE_V>();
    } else {
      SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
      WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
    }

    for (uint32_t rowBase = 0; rowBase < nActual; rowBase += tiled_ ? QW4_TILE_N : QW4_FRACTAL_SIZE) {
      for (int64_t k0 = 0; k0 < K_; k0 += tileK_) {
        const int64_t codeOffset = tiled_ ? static_cast<int64_t>(n0) * packedK_ + k0 * QW4_TILE_N / codesPerByte_
                                          : (static_cast<int64_t>(n0) + rowBase) * packedK_ + k0 / codesPerByte_;
        DecodeTile(codeOffset, rowBase, k0 / QW4_GROUP_SIZE);

        // Gather already transposed W's eight 16x16 fragments into
        // Cube NZ order. Their K-fractal positions are contiguous.
        const int64_t nFractal = rowBase / QW4_FRACTAL_SIZE;
        const int64_t nzColumnBlockStride = K_ * QW4_FRACTAL_SIZE;
        const int64_t nzBase = coreNzBase_ + nFractal * nzColumnBlockStride;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        if (tiled_) {
          DataCopy(wdqNzGm_[nzBase + k0 * QW4_FRACTAL_SIZE], signedHalfUB_, (int32_t)packedTileCount_);
          DataCopy(wdqNzGm_[nzBase + nzColumnBlockStride + k0 * QW4_FRACTAL_SIZE], signedHalfUB_[packedTileCount_],
                   (int32_t)packedTileCount_);
        } else {
          DataCopy(wdqNzGm_[nzBase + k0 * QW4_FRACTAL_SIZE], nzTileUB_, (int32_t)decodedTileCount_);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
      }
    }
  }

  Arch::Resource<ArchTag> resource;

  GlobalTensor<half> xGm_;
  GlobalTensor<uint8_t> codesGm_;
  GlobalTensor<half> scaleGm_;
  GlobalTensor<int8_t> offsetGm_;
  GlobalTensor<half> yGm_;
  GlobalTensor<half> wdqNzGm_;

  LocalTensor<half> scaleUB_;
  LocalTensor<int8_t> offsetUB_;
  LocalTensor<half> offsetHalfUB_;
  LocalTensor<half> scaleVecUB_;
  LocalTensor<half> offsetVecUB_;
  LocalTensor<uint8_t> cU8_;
  LocalTensor<half> cH_;
  LocalTensor<int16_t> c16_;
  LocalTensor<int16_t> andTmp_;
  LocalTensor<half> fieldHalfUB_;
  LocalTensor<int16_t> fieldI16UB_;
  LocalTensor<half> signedHalfUB_;
  LocalTensor<half> nzTileUB_;
  LocalTensor<uint32_t> gatherOffsetsUB_;
  LocalTensor<float> gatherIndexFloatUB_;
  LocalTensor<int16_t> threeUB_;
  LocalTensor<int16_t> signHalfUB_;
  LocalTensor<int16_t> masksUB_;

  int64_t T_;
  int64_t N_;
  int64_t K_;
  int64_t tileK_;
  int64_t codesPerByte_;
  int64_t bitsPerCode_;
  int64_t packedK_;
  int64_t packedTileCols_;
  int64_t packedTileCount_;
  int64_t decodedTileCount_;
  int64_t fieldMask_;
  int64_t signHalf_;
  int64_t kbCount_;
  int64_t coreNzBase_;
  bool tiled_;
  uint32_t tileId_;
  uint32_t tileCount_;
};

}  // namespace NsQwenW4
#endif  // QWEN_W4_GROUP_MATMUL_V310_H
