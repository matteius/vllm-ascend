# SPDX-License-Identifier: Apache-2.0
"""Create an isolated experiment from the pinned live snapshot; never launch it."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from tools.qwen38_decode_study.affinity import cpu_topology, validate_groups

PINNED_FILES = (
    ("vllm_ascend/cpu_binding.py", "b2f29235a2eca792baf2f97cb6c5ad270973c0edf8138b4030dff9d8f0c9736f"),
    ("vllm_ascend/models/qwen4_exp/mtp.py", "6f92f223beb58c50f0ed2db9e73a04605675f727f3226611d806c9d180a32cc9"),
    ("vllm_ascend/models/qwen4_exp/moe.py", "f3f14677e2af54bdff8d493923032822ef7bccefa0889217c09588c443c61856"),
    ("vllm_ascend/models/qwen4_exp/model.py", "92115b5dba4837864d92e960fce0189b28fe29142f88aee575b9b1db57f1a29c"),
    (
        "vllm_ascend/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so",
        "f079d9a462a9ea9ffe6a47a68e7bdee2d771f27e6172550ba404c8331f98b943",
    ),
    (
        "vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py",
        "8d9adde2411c4d956413bc9de36c03676a51fb2df3332759a9724b720c286800",
    ),
)
LAUNCHER_SHA256 = "60ffca76b8e2f475efd09070189ca78621644faebf1c95c15ab04a2df70906e7"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one patch anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch_mtp(text: str, draft_eager: bool = True) -> str:
    tree = ast.parse(text)
    bank = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_MTPFP16MoE")
    eager = next(node for node in bank.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_eager")
    lines = text.splitlines(keepends=True)
    eager_text = "".join(lines[eager.lineno - 1 : eager.end_lineno])
    prefix = eager_text.split("        flat_ids = ids.flatten()\n")[0]
    tail = eager_text[eager_text.index("        if self.local_shared_intermediate:") :]
    grouped = prefix.replace("def _forward_eager(", "def _forward_grouped(")
    grouped += "        output = grouped_routed_experts(self, x, weights, ids)\n" + tail
    text = replace_once(text, eager_text, eager_text + "\n" + grouped)
    if draft_eager:
        text = replace_once(
            text, "            return self._forward_eager(x)\n", "            return self._forward_grouped(x)\n"
        )
        text = replace_once(
            text,
            "            weak_output.copy_(self._forward_eager(weak_x))\n",
            "            weak_output.copy_(self._forward_grouped(weak_x))\n",
        )
    else:
        text = replace_once(
            text,
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        capture =",
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            '        if x.device.type == "npu":\n'
            "            return self._forward_grouped(x)\n"
            "        capture =",
        )
    text = replace_once(
        text,
        "        return loaded\n",
        "        for layer in self.model.layers:\n"
        "            if isinstance(layer.mlp, _MTPFP16MoE) and layer.mlp.quantized_experts:\n"
        "                pack_draft_experts(layer.mlp)\n"
        "        return loaded\n",
    )
    text = replace_once(
        text,
        "from .dtype_policy import Qwen4ExpDtypePolicy\n",
        "from vllm_ascend._310p.qwen38_grouped_candidate import grouped_routed_experts, pack_draft_experts\n\n"
        "from .dtype_policy import Qwen4ExpDtypePolicy\n",
    )
    ast.parse(text)
    return text


def patch_target_routing(text: str) -> str:
    """Fix the same routing contract in the pinned target's small-batch path."""
    start = "        # The 310P routing kernel computes the expert permutation, inverse\n"
    end = "        local_rows = torch.arange(num_tokens * top_k, device=x.device) < group_list[-1]\n"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("Expected exactly one target routing patch anchor")
    begin = text.index(start)
    finish = text.index(end)
    if finish <= begin:
        raise ValueError("Target routing patch anchors are out of order")
    text = replace_once(
        text,
        text[begin:finish],
        "        # Include a real routing slot for peers, but no corresponding weights.\n"
        "        sorted_x, inverse_order, group_list = route_local_experts(\n"
        "            x, topk_ids, w13_weight.shape[0], expert_offset, torch_npu\n"
        "        )\n",
    )
    text = replace_once(
        text,
        "from .grouped_expert_dispatch import build_grouped_expert_dispatch\n",
        "from vllm_ascend._310p.qwen38_grouped_candidate import route_local_experts\n\n"
        "from .grouped_expert_dispatch import build_grouped_expert_dispatch\n",
    )
    ast.parse(text)
    return text


def patch_affinity(text: str, groups: list[list[int]]) -> str:
    pools = tuple(tuple(group) for group in groups)
    text = replace_once(
        text,
        "from vllm.logger import logger\n",
        "from vllm.logger import logger\n\nfrom vllm_ascend._310p.qwen38_affinity_candidate import bind_worker\n",
    )
    return replace_once(
        text,
        '        logger.info("CPU binding skipped: non-ARM CPU detected.")\n        return\n',
        f"        bind_worker(rank_id, {pools!r})\n"
        f'        logger.info("Qwen38 candidate affinity rank=%s cpus=%s", rank_id, {pools!r}[rank_id])\n'
        "        return\n",
    )


def prepare(
    source: Path, destination: Path, launcher: Path, arm: str, plan: Path | None, draft_eager: bool | None = None
) -> dict:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Destination must be a new directory outside the baseline")
    for name, expected in PINNED_FILES:
        if digest(source / name) != expected:
            raise ValueError(f"Live snapshot differs from reviewed source: {name}")
    if digest(launcher) != LAUNCHER_SHA256:
        raise ValueError("Launcher differs from the reviewed live baseline")
    if arm not in {"control", "affinity", "grouped", "combined"}:
        raise ValueError("Unknown study arm")
    grouped = arm in {"grouped", "combined"}
    if draft_eager is not None and not grouped:
        raise ValueError("Draft capture options require a grouped draft arm")
    grouped_eager = grouped and draft_eager is not False
    helpers = Path(__file__).resolve().parent
    groups = None
    if arm in {"affinity", "combined"}:
        if plan is None:
            raise ValueError("Affinity arms require --cpu-plan")
        groups = json.loads(plan.read_text())["groups"]
        if len(groups) != 4:
            raise ValueError("This pinned launcher requires four TP rank pools")
        validate_groups(groups, set(os.sched_getaffinity(0)), cpu_topology())
    # Copy file contents, never hardlink to the running checkout. Resolve
    # operator-library symlinks so future changes to the original don't leak in.
    dangling = [str(path.relative_to(source)) for path in source.rglob("*") if path.is_symlink() and not path.exists()]
    if any(name.startswith("vllm_ascend/_cann_ops_custom/") for name in dangling):
        raise ValueError("Active custom operator bundle contains a dangling symlink")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
        ignore_dangling_symlinks=True,
    )
    changes = []
    if groups is not None:
        path = destination / "vllm_ascend/cpu_binding.py"
        path.write_text(patch_affinity(path.read_text(), groups))
        helper_path = destination / "vllm_ascend/_310p/qwen38_affinity_candidate.py"
        shutil.copy2(helpers / "affinity.py", helper_path)
        changes.extend([path, helper_path])
    if grouped:
        path = destination / "vllm_ascend/models/qwen4_exp/mtp.py"
        path.write_text(patch_mtp(path.read_text(), grouped_eager))
        target_path = destination / "vllm_ascend/models/qwen4_exp/moe.py"
        target_path.write_text(patch_target_routing(target_path.read_text()))
        helper_path = destination / "vllm_ascend/_310p/qwen38_grouped_candidate.py"
        shutil.copy2(helpers / "grouped_draft.py", helper_path)
        changes.extend([path, target_path, helper_path])
    for path in changes:
        ast.parse(path.read_text())
    shutil.copy2(helpers / "guard.py", destination / "candidate_guard.py")
    # Preserve all model/kernel settings from the live recipe. Only change the
    # source root, listener/log locations, and add a refusal-only startup guard.
    recipe = launcher.read_text()
    recipe = replace_once(
        recipe,
        "QWEN38_PLUGIN_ROOT=${QWEN38_PLUGIN_ROOT:-/srv/ai/src/qwen4exp-serving-20260923}",
        f"QWEN38_PLUGIN_ROOT={shlex.quote(str(destination))}",
    )
    recipe = replace_once(recipe, "PORT=${PORT:-8001}", "PORT=${PORT:-8002}")
    recipe = replace_once(recipe, '--host 0.0.0.0 --port "$PORT"', '--host 127.0.0.1 --port "$PORT"')
    recipe = replace_once(
        recipe, 'LOG="$HOME/logs/serve_qwen38_mtp_graph_${PORT}.log"', 'LOG="${QWEN38_PLUGIN_ROOT}/serve.log"'
    )
    recipe = replace_once(
        recipe,
        'WATCHDOG_LOG="$HOME/logs/watchdog_qwen38_mtp_graph.log"',
        'WATCHDOG_LOG="${QWEN38_PLUGIN_ROOT}/watchdog.log"',
    )
    recipe = replace_once(
        recipe, '[[ -d "$MODEL" ]]', 'python3 "${QWEN38_PLUGIN_ROOT}/candidate_guard.py"\n\n[[ -d "$MODEL" ]]'
    )
    launch_path = destination / "launch-candidate.sh"
    launch_path.write_text(recipe)
    subprocess.run(["bash", "-n", str(launch_path)], check=True)
    changes.extend([destination / "candidate_guard.py", launch_path])
    manifest = {
        "arm": arm,
        "baseline": str(source),
        "destination": str(destination),
        "baseline_files": dict(PINNED_FILES),
        "baseline_launcher_sha256": LAUNCHER_SHA256,
        "changes": {str(path.relative_to(destination)): digest(path) for path in changes},
        "cpu_groups": groups,
        "skipped_dangling_build_links": dangling,
        "draft_activation_quantization": "W8A8 candidate" if arm in {"grouped", "combined"} else "W8A16 baseline",
        "draft_grouped_eager_callback": grouped_eager,
        "local_expert_routing": "virtual peer expert for target and draft" if grouped else "baseline",
        "device_validation": "NOT_RUN: real-weight gates required before promotion",
    }
    (destination / "candidate-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "affinity", "grouped", "combined"), required=True)
    parser.add_argument("--cpu-plan", type=Path)
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument(
        "--draft-eager", dest="draft_eager", action="store_true", help="Retain the draft eager callback (default)"
    )
    capture.add_argument(
        "--draft-capture", dest="draft_eager", action="store_false", help="Opt into unvalidated full draft capture"
    )
    parser.set_defaults(draft_eager=None)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source, args.destination, args.launcher, args.arm, args.cpu_plan, args.draft_eager), indent=2
        )
    )


if __name__ == "__main__":
    main()
