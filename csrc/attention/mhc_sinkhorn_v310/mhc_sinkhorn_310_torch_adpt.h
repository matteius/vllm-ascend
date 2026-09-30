#ifndef MHC_SINKHORN_310_TORCH_ADPT_H
#define MHC_SINKHORN_310_TORCH_ADPT_H

#include <cmath>

namespace vllm_ascend {

at::Tensor mhc_sinkhorn_310(const at::Tensor& logits,
                             int64_t iterations,
                             double epsilon)
{
    TORCH_CHECK(logits.dim() == 3 && logits.size(1) == 4 && logits.size(2) == 4,
                "mHC Sinkhorn logits must be [tokens,4,4]");
    TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.is_contiguous(),
                "mHC Sinkhorn logits must be contiguous float32");
    TORCH_CHECK(iterations >= 1 && iterations <= 64,
                "mHC Sinkhorn iterations must be in [1,64]");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0,
                "mHC Sinkhorn epsilon must be finite and nonnegative");
    at::Tensor mix = at::empty_like(logits);
    EXEC_NPU_CMD(aclnnMhcSinkhornV310,
                 logits, iterations, epsilon, mix);
    return mix;
}

}  // namespace vllm_ascend

#endif
