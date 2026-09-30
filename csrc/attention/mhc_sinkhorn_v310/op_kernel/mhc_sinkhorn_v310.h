#ifndef MHC_SINKHORN_V310_H
#define MHC_SINKHORN_V310_H

#include "kernel_operator.h"
#include "mhc_sinkhorn_v310_tiling_data.h"

namespace NsMhcSinkhorn {

using namespace AscendC;

constexpr int64_t MHC_STREAMS = 4;
constexpr int64_t MATRIX_ELEMENTS = MHC_STREAMS * MHC_STREAMS;

class MhcSinkhornV310 {
public:
    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR mix,
                                const MhcSinkhornV310TilingData *tiling,
                                TPipe *pipe)
    {
        logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logits));
        mixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mix));
        numRows_ = tiling->numRows;
        rowsPerCore_ = tiling->rowsPerCore;
        iterations_ = tiling->iterations;
        epsilon_ = tiling->epsilon;
        pipe->InitBuffer(matrixBuf_, MATRIX_ELEMENTS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t first = GetBlockIdx() * rowsPerCore_;
        for (int64_t offset = 0; offset < rowsPerCore_; ++offset) {
            const int64_t row = first + offset;
            if (row >= numRows_) {
                break;
            }
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(int64_t row)
    {
        LocalTensor<float> local = matrixBuf_.Get<float>();
        DataCopy(local, logitsGm_[row * MATRIX_ELEMENTS], MATRIX_ELEMENTS);
        PipeBarrier<PIPE_ALL>();

        for (int64_t r = 0; r < MHC_STREAMS; ++r) {
            float maximum = local.GetValue(r * MHC_STREAMS);
            for (int64_t c = 1; c < MHC_STREAMS; ++c) {
                const float value = local.GetValue(r * MHC_STREAMS + c);
                maximum = maximum > value ? maximum : value;
            }
            for (int64_t c = 0; c < MHC_STREAMS; ++c) {
                const int64_t index = r * MHC_STREAMS + c;
                local.SetValue(index, local.GetValue(index) - maximum);
            }
        }
        PipeBarrier<PIPE_ALL>();
        Exp(local, local, MATRIX_ELEMENTS);
        PipeBarrier<PIPE_ALL>();

        float matrix[MATRIX_ELEMENTS];
        for (int64_t r = 0; r < MHC_STREAMS; ++r) {
            float total = 0.0f;
            for (int64_t c = 0; c < MHC_STREAMS; ++c) {
                const int64_t index = r * MHC_STREAMS + c;
                matrix[index] = local.GetValue(index);
                total += matrix[index];
            }
            for (int64_t c = 0; c < MHC_STREAMS; ++c) {
                const int64_t index = r * MHC_STREAMS + c;
                matrix[index] = matrix[index] / total + epsilon_;
            }
        }

        NormalizeColumns(matrix);
        for (int64_t iteration = 1; iteration < iterations_; ++iteration) {
            NormalizeRows(matrix);
            NormalizeColumns(matrix);
        }

        for (int64_t index = 0; index < MATRIX_ELEMENTS; ++index) {
            local.SetValue(index, matrix[index]);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(mixGm_[row * MATRIX_ELEMENTS], local, MATRIX_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NormalizeRows(float *matrix)
    {
        for (int64_t r = 0; r < MHC_STREAMS; ++r) {
            const int64_t base = r * MHC_STREAMS;
            const float denominator = matrix[base] + matrix[base + 1] +
                                      matrix[base + 2] + matrix[base + 3] +
                                      epsilon_;
            for (int64_t c = 0; c < MHC_STREAMS; ++c) {
                matrix[base + c] /= denominator;
            }
        }
    }

    __aicore__ inline void NormalizeColumns(float *matrix)
    {
        for (int64_t c = 0; c < MHC_STREAMS; ++c) {
            const float denominator = matrix[c] + matrix[MHC_STREAMS + c] +
                                      matrix[2 * MHC_STREAMS + c] +
                                      matrix[3 * MHC_STREAMS + c] + epsilon_;
            for (int64_t r = 0; r < MHC_STREAMS; ++r) {
                matrix[r * MHC_STREAMS + c] /= denominator;
            }
        }
    }

    TBuf<TPosition::VECCALC> matrixBuf_;
    GlobalTensor<float> logitsGm_;
    GlobalTensor<float> mixGm_;
    int64_t numRows_ = 0;
    int64_t rowsPerCore_ = 0;
    int64_t iterations_ = 0;
    float epsilon_ = 0.0f;
};

}  // namespace NsMhcSinkhorn

#endif
