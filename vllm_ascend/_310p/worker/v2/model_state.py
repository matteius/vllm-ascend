# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""MRV2 model state for Ascend 310P (dense/VL + hybrid/GDN)."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.mamba_utils import get_mamba_groups
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend._310p.ops.rotary_embedding import prepare_mrope_cos_sin_slices_from_runner
from vllm_ascend._310p.worker.v2.input_batch import Ascend310PInputBatch
from vllm_ascend._310p.worker.v2.rope import Ascend310PRopeState, get_310p_rope_state
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    Qwen4ExpDtypePolicy,
)
from vllm_ascend.models.qwen4_exp.qwen4exp_gdn import (
    GDNBackend,
    GDNConvStateLayout,
    Qwen4ExpGDNParams,
    Qwen4ExpGDNStateLayout,
    Qwen4ExpGDNStatePool,
    gdn_delta_rule,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

from .rejection_sampler import RejectionSampler310V2
from .sampler import Ascend310PSampler


class _Ascend310PModelStateMixin:
    """310P RoPE / FULL-graph seq_lens helpers shared by dense and hybrid.

    Attribute annotations below are provided at runtime by AscendModelState /
    DefaultModelState (or set in the concrete subclass ``__init__``). Declared
    here so mypy can type-check the mixin in isolation.
    """

    model_config: Any
    model: nn.Module
    max_num_reqs: int
    max_num_tokens: int
    max_model_len: int
    device: torch.device
    rope_state: Any
    mm_pruner: Any
    _capture_seq_lens_by_ptr: dict[int, torch.Tensor]

    def _replace_310p_rope_state(self, encoder_cache: EncoderCache | None) -> None:
        del encoder_cache  # EVS / mm_pruner unsupported on 310P (see review notes).
        self.rope_state = get_310p_rope_state(
            self.model_config,
            self.model,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            self.device,
        )
        # Clear any parent-created EVS pruner: 310P has no EVS path, and
        # Ascend310PRopeState lacks the read/update_prefill_positions EVS needs.
        self.mm_pruner = None

    def _record_capture_seq_lens(self, seq_lens: torch.Tensor) -> None:
        """Record the largest captured view for each physical buffer."""
        data_ptr = seq_lens.data_ptr()
        recorded = self._capture_seq_lens_by_ptr.get(data_ptr)
        if recorded is None or seq_lens.numel() > recorded.numel():
            self._capture_seq_lens_by_ptr[data_ptr] = seq_lens

    def _refresh_capture_seq_lens(self, runtime_seq_lens: torch.Tensor) -> None:
        """Copy runtime lengths to buffers read by FULL graph replay."""
        for capture_seq_lens in self._capture_seq_lens_by_ptr.values():
            num_seq_lens = min(capture_seq_lens.numel(), runtime_seq_lens.numel())
            capture_seq_lens[:num_seq_lens].copy_(runtime_seq_lens[:num_seq_lens], non_blocking=True)
            if num_seq_lens < capture_seq_lens.numel():
                capture_seq_lens[num_seq_lens:].zero_()

    def prepare_attn(
        self,
        input_batch: Ascend310PInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        if for_capture:
            self._record_capture_seq_lens(input_batch.seq_lens)
        elif cudagraph_mode == CUDAGraphMode.FULL:
            # Updating only input_batch.seq_lens is insufficient when replay is
            # bound to a different capture-time address.
            self._refresh_capture_seq_lens(input_batch.seq_lens)

        # Mixin sits before AscendModelState / AscendMambaHybridModelState in MRO.
        return super().prepare_attn(  # type: ignore[misc]
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
        )

    def prepare_inputs(self, input_batch: Ascend310PInputBatch, req_states):
        if self.rope_state is None:
            return super().prepare_inputs(input_batch, req_states)  # type: ignore[misc]

        assert isinstance(self.rope_state, Ascend310PRopeState)
        # Upstream RopeState.prepare_positions uses Triton; 310P builds positions
        # on CPU from staged prefill tables, then H2D copies.
        self.rope_state.prepare_positions_cpu(
            input_batch.idx_mapping_np,
            input_batch.query_start_loc_np,
            req_states.prefill_len.np,
            req_states.num_computed_tokens_np,
            input_batch.num_tokens_after_padding,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        if self.model_config.uses_mrope:
            prepare_mrope_cos_sin_slices_from_runner(self, positions)
        return {"positions": positions}

    def custom_sampler(self, sampler):
        # MTP propose/_dummy_run reads sampler.sampling_states.temperature/seeds.
        # ``object.__new__`` UT fixtures may omit attrs set in real ``__init__``.
        max_num_reqs = int(getattr(self, "max_num_reqs", 1) or 1)
        device = getattr(self, "device", torch.device("cpu"))
        vocab_size = getattr(getattr(sampler, "sampling_states", None), "vocab_size", None)
        if vocab_size is None:
            model_config = getattr(self, "model_config", None)
            if model_config is not None and hasattr(model_config, "get_vocab_size"):
                vocab_size = model_config.get_vocab_size()
        base_sampler = Ascend310PSampler(max_num_reqs, device, vocab_size=vocab_size)
        vllm_config = getattr(self, "vllm_config", None)
        spec_config = None if vllm_config is None else vllm_config.speculative_config
        if spec_config is None:
            return base_sampler, None
        method = getattr(spec_config, "method", None)
        if method != "mtp":
            raise NotImplementedError(f"310P MRv2 only supports MTP speculative decoding, got {method!r}.")
        return base_sampler, RejectionSampler310V2(base_sampler, spec_config, device)


class Ascend310PModelState(_Ascend310PModelStateMixin, AscendModelState):
    """Model state with Triton-free 310P sampler / MRoPE and encoder support."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Initialize the full Ascend/DefaultModelState contract first so
        # attributes such as ``prompt_embeds_state`` / ``encoder_runner`` exist,
        # then swap RoPE to the Triton-free 310P implementation.
        # follow-imports=skip hides DefaultModelState.__init__; ignore call-arg.
        AscendModelState.__init__(  # type: ignore[call-arg]
            self, vllm_config, model, encoder_cache, device
        )
        # ACLGraph replays the tensor addresses bound during capture. Keep every
        # captured seq_lens buffer so its contents can be refreshed before replay.
        self._capture_seq_lens_by_ptr = {}
        self._replace_310p_rope_state(encoder_cache)


class Ascend310PMambaHybridModelState(_Ascend310PModelStateMixin, AscendMambaHybridModelState):
    """310P hybrid/GDN state: keep Ascend hybrid contract, swap Triton RoPE."""

    def add_request(self, req_index: int, new_req_data) -> None:
        # Skip upstream Mamba GPU scalar fills; mirror their state on CPU.
        super(MambaHybridModelState, self).add_request(req_index, new_req_data)
        self._num_accepted_tokens_cpu[req_index] = 1
        if not self._align_mode:
            return
        # b2f685834a uses cache_config.block_size as the align page size.
        block_size = self.cache_config.block_size
        self._mamba_state_idx_cpu[req_index] = (new_req_data.num_computed_tokens - 1) // block_size

    def _ensure_mamba_copy_funcs_cpu(self, kv_cache_config: KVCacheConfig) -> None:
        if self._mamba_copy_funcs_by_group is not None:
            return
        mamba_groups = get_mamba_groups(kv_cache_config)
        copy_funcs_by_type = self.model.get_mamba_state_copy_funcs({spec.mamba_type for spec in mamba_groups})
        self._mamba_copy_funcs_by_group = {
            group_id: copy_funcs_by_type[spec.mamba_type]
            for spec, group_ids in mamba_groups.items()
            for group_id in group_ids
        }

    def _copy_mamba_state_from_cpu_plan(
        self,
        kv_cache_config: KVCacheConfig,
        block_tables_np: tuple[np.ndarray, ...],
        batch_idx: int,
        src_column: int,
        dst_column: int,
        token_bias: int,
    ) -> None:
        """CPU computes slices; NPU only executes required state copies."""
        assert self._mamba_copy_funcs_by_group is not None
        forward_context = self.vllm_config.compilation_config.static_forward_context
        for group_id in self._mamba_group_ids:
            block_ids = block_tables_np[group_id][batch_idx]
            src_block = int(block_ids[src_column])
            dst_block = int(block_ids[dst_column])
            group = kv_cache_config.kv_cache_groups[group_id]
            for layer_name in group.layer_names:
                attention = forward_context[layer_name]
                states: list[torch.Tensor] = attention.kv_cache
                for state, copy_func in zip(states, self._mamba_copy_funcs_by_group[group_id]):
                    if "conv" in copy_func.__name__:
                        src = state[src_block]
                        dst = state[dst_block]
                        width = src.shape[-1] if is_conv_state_dim_first() else src.shape[0]
                        copy_width = width - token_bias
                        if copy_width <= 0:
                            continue
                        if is_conv_state_dim_first():
                            dst[..., :copy_width].copy_(src[..., token_bias:].clone())
                        else:
                            dst[:copy_width].copy_(src[token_bias:].clone())
                    else:
                        actual_src_block = int(block_ids[src_column + token_bias])
                        state[dst_block].copy_(state[actual_src_block].clone())

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Initialize the complete upstream/Ascend hybrid contract first (e.g.
        # ``_align_mode`` / mamba metadata), then replace Triton RoPE.
        AscendMambaHybridModelState.__init__(  # type: ignore[call-arg]
            self, vllm_config, model, encoder_cache, device
        )
        self._capture_seq_lens_by_ptr = {}
        self._replace_310p_rope_state(encoder_cache)
        self._num_accepted_tokens_cpu = np.ones(self.max_num_reqs, dtype=np.int32)
        self._mamba_copy_funcs_by_group = None
        if self._align_mode:
            self._mamba_state_idx_cpu = np.zeros(self.max_num_reqs, dtype=np.int32)
            self._current_mamba_block_tables_np: tuple[np.ndarray, ...] | None = None

    def prepare_attn(
        self,
        input_batch: Ascend310PInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        """310P hybrid FULL: correct actual/padded token counts + uniform SpecDecoding pads.

        Upstream ``AscendMambaHybridModelState.prepare_attn`` passes only
        ``num_tokens`` (padded under FULL), so GDN treats pad tokens as actual.
        Pad rows also get ``draft_tokens=-1``, which turns a uniform SpecDecoding
        FULL batch into mixed Spec+Prefill — diverging from the captured uniform
        graph. Mirror AscendModelState's actual/padded split and keep pad rows on
        the SpecDecoding path (draft=K, accepted=1), with pad accepted=1.
        """
        assert ubatch_idx == 0, "DBO is not supported on Ascend"
        if for_capture:
            self._record_capture_seq_lens(input_batch.seq_lens)
        elif cudagraph_mode == CUDAGraphMode.FULL:
            self._refresh_capture_seq_lens(input_batch.seq_lens)

        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_input_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_input_tokens = input_batch.num_tokens
        num_actual_reqs = input_batch.num_reqs
        num_actual_tokens = input_batch.num_tokens

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[:num_actual_reqs] = torch.from_numpy(input_batch.is_prefilling_np)

        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            accepted_np = np.ones(num_reqs, dtype=np.int32)
            accepted_np[:num_actual_reqs] = self._num_accepted_tokens_cpu[input_batch.idx_mapping_np]
            num_accepted_tokens = torch.from_numpy(accepted_np).to(
                device=self.device,
                non_blocking=True,
            )

            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[:num_actual_reqs] = np.where(
                    spec_decode_mask,
                    num_draft_tokens_per_req,
                    -1,
                )
                # Align with upstream #15707: only promote pad rows to Spec when
                # every real request is SpecDecoding and pad query lens == 1+K.
                # Also keep pad rows Spec when attn_state is already SpecDecoding
                # (310P target FULL capture / concurrent pad).
                if cudagraph_mode == CUDAGraphMode.FULL and num_reqs > num_actual_reqs:
                    expected_query_len = int(self.vllm_config.num_speculative_tokens) + 1
                    padded_query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])[num_actual_reqs:]
                    attn_state = input_batch.attn_state
                    is_spec = attn_state is not None and getattr(attn_state, "name", "") == "SpecDecoding"
                    if (spec_decode_mask.all() or is_spec) and np.all(padded_query_lens == expected_query_len):
                        num_decode_draft_tokens_np[num_actual_reqs:] = padded_query_lens - 1
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        # Host seq_lens for pad rows must be 0 (GPU already zeroed in prepare_inputs).
        seq_lens_np = input_batch.seq_lens_np
        if seq_lens_np is not None and num_reqs > num_actual_reqs:
            seq_lens_np = seq_lens_np.copy()
            seq_lens_np[num_actual_reqs:num_reqs] = 0

        model_specific_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        self.attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_tokens=num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            is_prefilling=is_prefilling,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.num_scheduled_tokens.max().item(),
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            seq_lens_np=seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            model_specific_attn_metadata=model_specific_metadata,
            for_cudagraph_capture=for_capture,
        )
        return self.attn_metadata

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if not self.vllm_config.cache_config.enable_prefix_caching:
            # Recurrent state has no historic pages to retain when prefix
            # caching is disabled. Address it by vLLM's stable request index so
            # the backing allocation only needs max_num_reqs slots. Mutating
            # the gathered per-step tables is safe: scheduler-owned CPU block
            # tables are separate, and attention groups keep their original
            # paged ids.
            num_reqs = input_batch.num_reqs
            num_reqs_padded = input_batch.num_reqs_after_padding
            request_slots = input_batch.idx_mapping[:num_reqs].to(dtype=torch.int32)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
                if not isinstance(group.kv_cache_spec, MambaSpec):
                    continue
                table = block_tables[group_id]
                table[:num_reqs].copy_(request_slots[:, None].expand(-1, table.shape[1]))
                if num_reqs < num_reqs_padded:
                    table[num_reqs:num_reqs_padded].zero_()

        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )

    def preprocess_state(
        self,
        input_batch: Ascend310PInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Plan align transition on CPU; issue only required NPU state copies."""
        del block_tables, num_computed_tokens
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return

        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        self._ensure_mamba_copy_funcs_cpu(kv_cache_config)
        block_tables_np = input_batch.block_tables_np
        if block_tables_np is None:
            raise RuntimeError("310P CPU-first Mamba preparation requires CPU block tables.")
        self._current_mamba_block_tables_np = block_tables_np
        self._current_mamba_idx_mapping_np = input_batch.idx_mapping_np.copy()
        self._current_mamba_kv_cache_config = kv_cache_config

        for batch_idx, req_idx_value in enumerate(input_batch.idx_mapping_np):
            req_idx = int(req_idx_value)
            old_state_idx = int(self._mamba_state_idx_cpu[req_idx])
            token_bias = max(int(self._num_accepted_tokens_cpu[req_idx]) - 1, 0)
            computed_after = int(input_batch.num_computed_tokens_np[batch_idx]) + int(
                input_batch.query_start_loc_np[batch_idx + 1] - input_batch.query_start_loc_np[batch_idx]
            )
            new_state_idx = (computed_after + mamba_spec.block_size - 1) // mamba_spec.block_size - 1
            self._mamba_state_idx_cpu[req_idx] = new_state_idx
            if old_state_idx >= 0 and old_state_idx != new_state_idx:
                self._copy_mamba_state_from_cpu_plan(
                    kv_cache_config,
                    block_tables_np,
                    batch_idx,
                    old_state_idx,
                    new_state_idx,
                    token_bias,
                )
                self._num_accepted_tokens_cpu[req_idx] = 1

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        if num_reqs:
            idx_mapping_np = idx_mapping.numpy()
            if isinstance(num_sampled, int):
                sampled_np = np.full(num_reqs, max(num_sampled, 1), dtype=np.int32)
            else:
                sampled_np = np.maximum(num_sampled.numpy(), 1)
            valid = idx_mapping_np >= 0
            self._num_accepted_tokens_cpu[idx_mapping_np[valid]] = sampled_np[valid]

        if self.recoverssm is not None:
            # RecoverSSM is an NPU consumer. Materialize CPU-owned state only
            # at this boundary; normal 310P Mamba never pays these copies.
            self.num_accepted_tokens_gpu.copy_(torch.from_numpy(self._num_accepted_tokens_cpu), non_blocking=True)
            recover_idx_mapping = idx_mapping.to(self.device, non_blocking=True)
            recover_num_sampled = (
                num_sampled if isinstance(num_sampled, int) else num_sampled.to(self.device, non_blocking=True)
            )
            if self._align_mode:
                self._mamba_state_idx_gpu.copy_(torch.from_numpy(self._mamba_state_idx_cpu), non_blocking=True)
            self.recoverssm.commit_step(
                recover_num_sampled,
                recover_idx_mapping,
                state_indices=(self._mamba_state_idx_gpu if self._align_mode else None),
                num_accepted_tokens=self.num_accepted_tokens_gpu,
            )
            self._num_accepted_tokens_cpu[:] = self.num_accepted_tokens_gpu.cpu().numpy()
            if self._align_mode:
                self._mamba_state_idx_cpu[:] = self._mamba_state_idx_gpu.cpu().numpy()

        if not num_reqs:
            return

        if not self._align_mode or num_computed_tokens is None:
            return

        accepted = torch.clamp(num_sampled.masked_select(valid), min=1).to(self.num_accepted_tokens_gpu.dtype)
        self.num_accepted_tokens_gpu.index_copy_(0, valid_indices, accepted)


class Ascend310PQwen4ExpModelState(Ascend310PMambaHybridModelState):
    """310P Qwen4Exp hybrid state with rollback-safe PLE n-gram context.

    Ports the fork ``Qwen4ExpModelState`` PLE input maintenance onto the 310P
    Triton-free hybrid state (``Ascend310PMambaHybridModelState``):

    * an n-gram CONTEXT BUFFER rebuilt every step from the authoritative
      ``num_computed_tokens`` -- because the context is reconstructed from
      scratch rather than appended to, a rejected-speculative rollback (which
      shrinks ``num_computed_tokens``) needs no special handling and the S=0
      degenerate-first step falls out naturally as all-EOS padding;
    * EOS padding for the leading tokens of a request (positions before index
      0 clamp to EOS);
    * a fixed-shape ``ple_query_start_loc`` with unused capacity represented as
      trailing zero-length requests.

    Buffers keep a FIXED shape and address as the active request count changes
    between replays, so PLE runs inside graph-captured regions even in eager
    mode. PP=1 is enforced: non-first pipeline ranks never receive the raw
    ``input_ids`` that PLE needs.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Build the full 310P hybrid contract first (mamba metadata, Triton-free
        # RoPE), then add the rollback-safe PLE n-gram context buffers.
        super().__init__(vllm_config, model, encoder_cache, device)
        self._init_ngram_context(vllm_config)

    def _init_ngram_context(self, vllm_config: VllmConfig) -> None:
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(getattr(config, "ple_layer_ids", None))
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires pipeline_parallel_size=1 "
                "because non-first pipeline ranks do not receive the raw input_ids "
                "required by PLE. Please run with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def _prepare_ngram_context(
        self,
        input_batch: AscendInputBatch,
        req_states: Any,
    ) -> torch.Tensor:
        """Rebuild the n-gram context buffer from authoritative token state.

        Row ``i`` holds the ``ngram_context_len`` tokens preceding
        ``num_computed_tokens`` of request ``idx_mapping[i]`` (its next-token
        position), so it stays correct across chunk boundaries. Positions before
        index 0 are EOS-padded. Recomputed each step, so speculative rollbacks
        (a shrunk ``num_computed_tokens``) are handled with no stale carryover.
        """
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[request_indices.unsqueeze(1), token_indices]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def _fill_ple_query_start_loc(self, input_batch: AscendInputBatch) -> torch.Tensor:
        """Fixed-shape query_start_loc; pad with trailing zero-length requests."""
        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[num_reqs_padded])
        return query_start_loc

    def _fill_ple_query_start_loc_dummy(self, num_reqs: int, num_tokens: int) -> torch.Tensor:
        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)
        return query_start_loc

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states: Any) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs
        model_inputs.update(
            query_start_loc=self._fill_ple_query_start_loc(input_batch),
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs
        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=self._fill_ple_query_start_loc_dummy(num_reqs, num_tokens),
            ngram_context=ngram_context,
        )
        return model_inputs

    # ------------------------------------------------------------------
    # GDN state lifecycle (plan T5.2)
    #
    # Prefill / decode / preemption / reuse for the Gated DeltaNet conv +
    # recurrent (SSM) state. State is REPLICATED PER TP RANK, so one
    # ``Qwen4ExpGDNStatePool`` is created per rank; the copy / slot-remap
    # semantics (no aliasing, clean reuse, fail-closed preemption) live in the
    # pool and mirror the fork ``MambaAttentionBackendEnum.GDN_ATTN`` copy-func
    # pair. These thin methods fan a lifecycle event across all ranks.
    # ------------------------------------------------------------------
    def _init_gdn_state_pools(
        self,
        params: Qwen4ExpGDNParams,
        *,
        num_blocks: int,
        tp_size: int = 1,
        num_ranks: int = 1,
        num_spec: int = 0,
        layout: GDNConvStateLayout = "DS",
        policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        device: torch.device | str | None = None,
    ) -> list[Qwen4ExpGDNStatePool]:
        """Build one per-rank GDN state pool (state replicated per TP rank)."""
        if num_ranks <= 0:
            raise ValueError(f"num_ranks must be >= 1, got {num_ranks}.")
        self.gdn_state_layout = Qwen4ExpGDNStateLayout.from_params(
            params, tp_size=tp_size, num_spec=num_spec, policy=policy, layout=layout
        )
        self.gdn_num_ranks = num_ranks
        pool_device = device if device is not None else self.device
        self._gdn_state_pools = [
            Qwen4ExpGDNStatePool(num_blocks, self.gdn_state_layout, device=pool_device) for _ in range(num_ranks)
        ]
        return self._gdn_state_pools

    def gdn_pool(self, rank: int = 0) -> Qwen4ExpGDNStatePool:
        return self._gdn_state_pools[rank]

    def gdn_begin_request(self, request_id: int) -> list[int]:
        """Allocate a fresh (zeroed, non-aliasing) state block on every rank."""
        return [pool.allocate(request_id) for pool in self._gdn_state_pools]

    def gdn_complete_request(self, request_id: int) -> None:
        """Free a completed request's blocks on every rank (zeroed for reuse)."""
        for pool in self._gdn_state_pools:
            pool.free(request_id)

    def gdn_preempt_request(self, request_id: int) -> None:
        """Fail-closed preemption on every rank; resume must re-seed + recompute."""
        for pool in self._gdn_state_pools:
            pool.preempt(request_id)

    def gdn_is_resident(self, request_id: int) -> bool:
        return all(pool.is_resident(request_id) for pool in self._gdn_state_pools)

    def gdn_active_blocks(self, rank: int = 0) -> dict[int, int]:
        """``request_id -> block_id`` on ``rank`` (for aliasing audits)."""
        return self._gdn_state_pools[rank].active_blocks()

    def gdn_recurrent_state(self, request_id: int, rank: int = 0) -> torch.Tensor:
        return self._gdn_state_pools[rank].recurrent_state(request_id)

    def gdn_write_recurrent(self, request_id: int, state: torch.Tensor, rank: int = 0) -> None:
        self._gdn_state_pools[rank].write_recurrent(request_id, state)

    def gdn_step(
        self,
        request_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        rank: int = 0,
        chunked: bool,
        scale: float | None = None,
        use_qk_l2norm: bool = True,
        compute_dtype: torch.dtype | None = None,
        backend: GDNBackend = "eager",
    ) -> torch.Tensor:
        """Advance one request's GDN state by a prefill chunk or a decode step.

        Reads the request's resident recurrent state (fail-closed if the request
        was preempted/completed and not re-seeded), runs the gated delta rule
        continuing from it, then writes the new state back into the same block.
        """
        pool = self._gdn_state_pools[rank]
        initial_state = pool.recurrent_state(request_id)
        out, new_state = gdn_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            chunked=chunked,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            compute_dtype=compute_dtype,
            backend=backend,
        )
        pool.write_recurrent(request_id, new_state)
        return out
