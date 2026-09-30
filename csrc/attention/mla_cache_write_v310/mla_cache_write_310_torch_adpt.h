#ifndef MLA_CACHE_WRITE_V310_TORCH_ADPT_H
#define MLA_CACHE_WRITE_V310_TORCH_ADPT_H

namespace vllm_ascend {

void mla_cache_write_310(at::Tensor& cache,
                         const at::Tensor& rows,
                         const at::Tensor& slot_mapping)
{
    TORCH_CHECK(cache.dim() == 4,
                "cache must be [blocks,head_dim/16,block_size,16]");
    TORCH_CHECK(rows.dim() == 3, "rows must be [tokens,head_dim/16,16]");
    TORCH_CHECK(cache.scalar_type() == at::kHalf && rows.scalar_type() == at::kHalf,
                "310P MLA cache write requires float16 cache and rows");
    TORCH_CHECK(cache.size(3) == 16 && rows.size(2) == 16,
                "310P MLA cache write requires an inner NZ width of 16");
    TORCH_CHECK(rows.size(1) == cache.size(1),
                "rows and cache must have the same latent head width");
    TORCH_CHECK(slot_mapping.dim() == 1 &&
                    slot_mapping.size(0) == rows.size(0),
                "slot_mapping must contain one entry per row");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kLong,
                "310P MLA slot_mapping must be int64");
    TORCH_CHECK(cache.stride(3) == 1 && cache.stride(2) == 16 &&
                    cache.stride(1) == cache.size(2) * 16,
                "310P MLA cache inner dimensions must use explicit NZ order");

    const int64_t page_stride = cache.stride(0);
    EXEC_NPU_CMD(aclnnMlaCacheWriteV310,
                 cache,
                 rows,
                 slot_mapping,
                 page_stride);
}

}  // namespace vllm_ascend

#endif
