import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

torch.npu.set_device(0)
enable_custom_op()
torch.manual_seed(37)
logits = torch.randn(17, 4, 4).npu() * 4
reference = torch.softmax(logits, dim=-1) + 1e-6
reference = reference / (reference.sum(-2, keepdim=True) + 1e-6)
for _ in range(19):
    reference = reference / (reference.sum(-1, keepdim=True) + 1e-6)
    reference = reference / (reference.sum(-2, keepdim=True) + 1e-6)
actual = torch.ops._C_ascend.mhc_sinkhorn_310(logits, 20, 1e-6)
ref = reference.cpu()
got = actual.cpu()
delta = (got - ref).abs().amax(dim=(1, 2))
print("delta", delta)
for row in torch.nonzero(delta > 1e-3).flatten():
    print("row", row.item())
    print("input", logits[row].cpu())
    print("actual", got[row])
    print("expected", ref[row])
