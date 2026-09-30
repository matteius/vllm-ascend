/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class HOutputType_,
    class GInputType_,
    class HInputType_,
    class HUpdateInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHUpdate,
    HOutputType_,
    GInputType_,
    HInputType_,
    HUpdateInputType_,
    FinalStateType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHUpdate;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using HElementInput = typename HInputType_::Element;
    using HUpdateElementInput = typename HUpdateInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

    static constexpr uint32_t FP32_VALUES_PER_BLOCK = 8;
    static constexpr uint32_t FP32_VALUES_PER_REPEAT = 64;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        // Bumped layout to fit kHeadDim up to 256 with subBlockNum=2 (per-subblock M up to 128).
        // Required: calc (fp32) up to 128*128*4=64KB; h (fp16) up to 128*128*2=32KB;
        // hUpdate/hOutput/finalOutput at the same offset, max needed 64KB; glast small.
        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t PING_BUF_0_OFFSET = 64 * 1024;
        constexpr uint32_t PING_BUF_1_OFFSET = 96 * 1024;
        constexpr uint32_t PING_BUF_2_OFFSET = 112 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;
        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        hUpdateUbTensor = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_1_OFFSET);
        hUbTensor = resource.ubBuf.template GetBufferByByte<HElementInput>(PING_BUF_0_OFFSET);

        hOutputUbTensor = resource.ubBuf.template GetBufferByByte<HElementOutput>(PING_BUF_1_OFFSET);
        finalOutputUbTensor = resource.ubBuf.template GetBufferByByte<FinalStateElement>(PING_BUF_1_OFFSET);

        glastUbTensor = resource.ubBuf.template GetBufferByByte<float>(PING_G_BUF_OFFSET);
        gkBroadcastUbTensor = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_2_OFFSET);

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        AscendC::GlobalTensor<float> hUpdateInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done,
        bool isFinalState,
        bool useKdaGatedPath
    )
    {
        uint32_t mActual = kHeadDim;
        uint32_t nActual = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetH = mOffset * nActual + nOffset;

        AscendC::ResetMask();

        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetH];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetH];
        AscendC::GlobalTensor<float> hUpdateInputThisSubBlock = hUpdateInput[offsetH];
        AscendC::GlobalTensor<FinalStateElement> finalStateThisSubBlock = finalState[offsetH];

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(calcUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_V>();
        
        if (useKdaGatedPath) {
            AscendC::GlobalTensor<GElementInput> gkLastInput =
                gInputThisSubBlock[(chunkSize - 1) * kHeadDim + mOffset];
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            if constexpr(std::is_same<GElementInput, float>::value) {
                AscendC::DataCopy(
                    glastUbTensor, gkLastInput, mActualThisSubBlock);
            } else {
                AscendC::DataCopy(
                    hOutputUbTensor, gkLastInput, mActualThisSubBlock);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr(!std::is_same<GElementInput, float>::value) {
                AscendC::Cast(glastUbTensor, hOutputUbTensor,
                              AscendC::RoundMode::CAST_NONE,
                              mActualThisSubBlock);
                AscendC::PipeBarrier<PIPE_V>();
            }
            constexpr float LN2 = 0.6931471805599453f;
            AscendC::Muls(glastUbTensor, glastUbTensor, LN2,
                          mActualThisSubBlock);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(glastUbTensor, glastUbTensor,
                         mActualThisSubBlock);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t gateRows =
                ((mActualThisSubBlock + FP32_VALUES_PER_BLOCK - 1) /
                 FP32_VALUES_PER_BLOCK) * FP32_VALUES_PER_BLOCK;
            AscendC::Brcb(
                gkBroadcastUbTensor, glastUbTensor,
                static_cast<uint8_t>(gateRows / FP32_VALUES_PER_BLOCK),
                {1, FP32_VALUES_PER_BLOCK});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BinaryRepeatParams gateMulParams{
                1, 1, 0, FP32_VALUES_PER_BLOCK,
                FP32_VALUES_PER_BLOCK, 0};
            uint32_t fullRepeats = nActual / FP32_VALUES_PER_REPEAT;
            uint32_t tailElements = nActual % FP32_VALUES_PER_REPEAT;
            for (uint32_t row = 0; row < mActualThisSubBlock; ++row) {
                uint32_t rowOffset = row * nActual;
                if (fullRepeats > 0) {
                    AscendC::Mul(
                        calcUbTensor[rowOffset], calcUbTensor[rowOffset],
                        gkBroadcastUbTensor[row * FP32_VALUES_PER_BLOCK],
                        FP32_VALUES_PER_REPEAT, fullRepeats, gateMulParams);
                }
                if (tailElements > 0) {
                    uint32_t tailOffset = rowOffset +
                        fullRepeats * FP32_VALUES_PER_REPEAT;
                    AscendC::Mul(
                        calcUbTensor[tailOffset], calcUbTensor[tailOffset],
                        gkBroadcastUbTensor[row * FP32_VALUES_PER_BLOCK],
                        tailElements, 1, gateMulParams);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            GElementInput gLastVal =
                gInputThisSubBlock.GetValue(chunkSize-1);
            float gLastFloat = 0.0f;
            if constexpr(std::is_same<GElementInput, float>::value) {
                gLastFloat = gLastVal;
            } else if constexpr(std::is_same<GElementInput, half>::value) {
                gLastFloat = (float)gLastVal;
            } else if constexpr(std::is_same<GElementInput, bfloat16_t>::value) {
                gLastFloat = AscendC::ToFloat(gLastVal);
            }
            glastUbTensor.SetValue(0, gLastFloat);

            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::Exp(glastUbTensor, glastUbTensor, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float muls = glastUbTensor.GetValue(0);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::Muls(calcUbTensor, calcUbTensor, muls,
                          mActualThisSubBlock * nActual);
        }


        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::DataCopy(hUpdateUbTensor, hUpdateInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Add<float>(hUpdateUbTensor, calcUbTensor, hUpdateUbTensor, mActualThisSubBlock * nActual);

        if (isFinalState) {
            if constexpr(!std::is_same<FinalStateElement, float>::value) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast(finalOutputUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(finalStateThisSubBlock, finalOutputUbTensor, mActualThisSubBlock * nActual);
            } else {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(finalStateThisSubBlock, hUpdateUbTensor, mActualThisSubBlock * nActual);
            }
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(hOutputUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(hOutputThisSubBlock, hOutputUbTensor, mActualThisSubBlock * nActual);
        }
    }

private:
    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<HElementInput> hUbTensor;
    AscendC::LocalTensor<float> hUpdateUbTensor;

    AscendC::LocalTensor<HElementOutput> hOutputUbTensor;
    AscendC::LocalTensor<FinalStateElement> finalOutputUbTensor;

    AscendC::LocalTensor<float> glastUbTensor;
    AscendC::LocalTensor<float> gkBroadcastUbTensor;

};
}

#endif
