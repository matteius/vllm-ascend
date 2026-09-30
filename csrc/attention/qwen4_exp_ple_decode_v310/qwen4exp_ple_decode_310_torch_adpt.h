#ifndef QWEN4EXP_PLE_DECODE_V310_TORCH_ADPT_H
#define QWEN4EXP_PLE_DECODE_V310_TORCH_ADPT_H

namespace vllm_ascend {

void qwen4exp_ple_decode_310(
    const at::Tensor& projected,
    const at::Tensor& hidden,
    const at::Tensor& norm_key_weight,
    const at::Tensor& norm_query_weight,
    const at::Tensor& norm_conv_weight,
    const at::Tensor& current_conv_weight,
    at::Tensor& output,
    double norm_eps)
{
    const auto device = hidden.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
                "310P PLE decode requires NPU tensors");
    TORCH_CHECK(projected.device() == device &&
                    norm_key_weight.device() == device && norm_query_weight.device() == device &&
                    norm_conv_weight.device() == device && current_conv_weight.device() == device &&
                    output.device() == device,
                "310P PLE decode inputs and output must be on the same NPU");
    TORCH_CHECK(projected.is_contiguous() && hidden.is_contiguous() &&
                    norm_key_weight.is_contiguous() && norm_query_weight.is_contiguous() &&
                    norm_conv_weight.is_contiguous() && current_conv_weight.is_contiguous() &&
                    output.is_contiguous(),
                "310P PLE decode inputs and output must be contiguous");
    TORCH_CHECK(projected.dim() == 2 && hidden.dim() == 2,
                "projected and hidden must be rank-2");
    TORCH_CHECK(projected.size(0) == hidden.size(0) && projected.size(1) > hidden.size(1),
                "PLE projected and hidden dimensions do not match");
    TORCH_CHECK(projected.scalar_type() == at::kHalf && hidden.scalar_type() == at::kHalf,
                "310P PLE decode requires float16 activations");
    TORCH_CHECK(norm_key_weight.scalar_type() == at::kHalf &&
                    norm_query_weight.scalar_type() == at::kHalf &&
                    norm_conv_weight.scalar_type() == at::kHalf,
                "310P PLE decode requires float16 norm weights");
    TORCH_CHECK(current_conv_weight.scalar_type() == at::kFloat,
                "310P PLE decode requires a preformatted float32 convolution tap");
    const int64_t value_width = projected.size(1) - hidden.size(1);
    TORCH_CHECK(hidden.size(1) % value_width == 0,
                "PLE hidden width must be divisible by value width");
    TORCH_CHECK(norm_key_weight.numel() == hidden.size(1) &&
                    norm_query_weight.numel() == hidden.size(1) &&
                    norm_conv_weight.numel() == hidden.size(1) &&
                    current_conv_weight.numel() == hidden.size(1),
                "PLE weights must match hidden width");
    TORCH_CHECK(norm_eps > 0.0, "PLE norm epsilon must be positive");

    TORCH_CHECK(output.sizes() == hidden.sizes() && output.scalar_type() == at::kHalf,
                "PLE output must be a float16 tensor matching hidden");
    double norm_eps_value = norm_eps;
    EXEC_NPU_CMD(aclnnQwen4ExpPleDecodeV310,
                 projected,
                 hidden,
                 norm_key_weight,
                 norm_query_weight,
                 norm_conv_weight,
                 current_conv_weight,
                 norm_eps_value,
                 output);
}

}  // namespace vllm_ascend

#endif
