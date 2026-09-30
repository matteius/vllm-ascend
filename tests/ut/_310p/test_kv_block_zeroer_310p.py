#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.base import TestBase
from vllm_ascend._310p.kv_block_zeroer import AscendKVBlockZeroer310


class TestAscendKVBlockZeroer310(TestBase):
    def setUp(self):
        self.zeroer = AscendKVBlockZeroer310(torch.device("cpu"), pin_memory=False)

    def test_zero_block_ids_noop_when_empty(self):
        kv = torch.ones(4, 2, 3)
        self.zeroer._kv_tensors = [kv]
        self.zeroer._logical_page_ratio = 1

        self.zeroer.zero_block_ids([])
        self.assertTrue(torch.all(kv == 1))

    def test_zero_block_ids_zeros_target_slices(self):
        kv = torch.ones(6, 2, 3)
        self.zeroer._kv_tensors = [kv]
        self.zeroer._logical_page_ratio = 2

        self.zeroer.zero_block_ids([1])

        self.assertTrue(torch.all(kv[:2] == 1))
        self.assertTrue(torch.all(kv[2:4] == 0))
        self.assertTrue(torch.all(kv[4:] == 1))

    def test_init_meta_deduplicates_kv_pointers(self):
        k_cache = torch.zeros(4, 2, 3)
        v_cache = k_cache
        layer_context = SimpleNamespace(kv_cache=(k_cache, v_cache))
        spec = FullAttentionSpec(
            block_size=128,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        )
        group = SimpleNamespace(
            kv_cache_spec=spec,
            kv_cache_group_id=0,
            layer_names=["layer_0"],
        )

        self.zeroer.init_meta(
            attn_groups_iter=[group],
            kernel_block_sizes=[[64]],
            cache_dtype="float16",
            runner_only_attn_layers=set(),
            static_forward_context={"layer_0": layer_context},
        )

        self.assertEqual(len(self.zeroer._kv_tensors), 1)
        self.assertEqual(self.zeroer._logical_page_ratio, 2)

    def test_host_hot_cache_is_not_zeroed_by_scheduler_ids(self):
        hot = torch.ones(2, 32, 32, 16)
        group = SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.float16,
            ),
            kv_cache_group_id=0,
            layer_names=["host_layer"],
        )
        layer = SimpleNamespace(kv_cache=(hot, hot), impl=SimpleNamespace(host_kv_layer=object()))
        self.zeroer.init_meta(
            attn_groups_iter=[group],
            kernel_block_sizes=[[32]],
            cache_dtype="float16",
            runner_only_attn_layers=set(),
            static_forward_context={"host_layer": layer},
        )
        self.zeroer.zero_block_ids([100])
        self.assertEqual(self.zeroer._kv_tensors, [])
        self.assertTrue(torch.all(hot == 1))
