#ifndef MLA_CACHE_WRITE_V310_H
#define MLA_CACHE_WRITE_V310_H

#include "kernel_operator.h"
#include "mla_cache_write_v310_tiling_data.h"

namespace NsMlaCacheWrite {

using namespace AscendC;

constexpr int64_t NZ_INNER = 16;

class MlaCacheWriteV310 {
public:
    __aicore__ inline void Init(GM_ADDR cache, GM_ADDR rows,
                                GM_ADDR slotMapping, GM_ADDR cacheOut,
                                const MlaCacheWriteV310TilingData *tiling,
                                TPipe *pipe)
    {
        numRows_ = tiling->numRows;
        channelBlocks_ = tiling->channelBlocks;
        blockSize_ = tiling->blockSize;
        pageStride_ = tiling->pageStride;
        rowsPerCore_ = tiling->rowsPerCore;
        cacheInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cache));
        rowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(rows));
        slotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slotMapping));
        cacheOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cacheOut));
        pipe_ = pipe;
        pipe_->InitBuffer(rowBuf_, channelBlocks_ * NZ_INNER * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        const int64_t firstRow = GetBlockIdx() * rowsPerCore_;
        for (int64_t offset = 0; offset < rowsPerCore_; ++offset) {
            const int64_t row = firstRow + offset;
            if (row >= numRows_) {
                break;
            }
            WriteRow(row);
        }
    }

private:
    __aicore__ inline void WriteRow(int64_t row)
    {
        const int64_t slot = slotsGm_.GetValue(row);
        if (slot < 0) {
            return;
        }
        const int64_t physicalBlock = slot / blockSize_;
        const int64_t tokenOffset = slot % blockSize_;
        const int64_t headDim = channelBlocks_ * NZ_INNER;
        LocalTensor<half> localRow = rowBuf_.Get<half>();
        DataCopy(localRow, rowsGm_[row * headDim], headDim);
        PipeBarrier<PIPE_ALL>();

        const int64_t cacheOffset =
            physicalBlock * pageStride_ + tokenOffset * NZ_INNER;
        DataCopyParams copyParams{
            static_cast<uint16_t>(channelBlocks_),
            1,
            0,
            static_cast<uint16_t>(blockSize_ - 1),
        };
        DataCopy(cacheOutGm_[cacheOffset], localRow, copyParams);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> rowBuf_;
    GlobalTensor<half> cacheInGm_;
    GlobalTensor<half> rowsGm_;
    GlobalTensor<int64_t> slotsGm_;
    GlobalTensor<half> cacheOutGm_;
    int64_t numRows_ = 0;
    int64_t channelBlocks_ = 0;
    int64_t blockSize_ = 0;
    int64_t pageStride_ = 0;
    int64_t rowsPerCore_ = 0;
};

}  // namespace NsMlaCacheWrite

#endif
