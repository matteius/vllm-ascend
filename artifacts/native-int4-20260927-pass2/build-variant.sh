#!/usr/bin/env bash
set -eo pipefail
variant=${1:?variant label required}
[[ "$variant" =~ ^[a-z0-9-]+$ ]] || exit 2
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
candidate_root=/srv/ai/src/native-int4-w4a8.KiuhBN
variant_dir=$candidate_root/ops-pass2-$variant
[[ ! -e "$variant_dir" ]] || { echo "Preserve existing variant: $variant_dir"; exit 2; }
cd "$candidate_root/csrc"
op=qwen_w4_a8_int4_matmul_v310
mkdir "$candidate_root/pass2/source-$variant"
cp "gmm/$op/op_kernel/"* "$candidate_root/pass2/source-$variant/"
cp "gmm/$op/op_kernel/"* "build/binary/ascend310p/src/$op/op_kernel/"
for stamp in build/binary/ascend310p/gen/${op}_*.done; do
    mv "$stamp" "$stamp.pass2-$variant-saved"
done
cmake --build build --target package -j4 > "$candidate_root/pass2/build-$variant.log" 2>&1
# The CANN packaging wrapper can exit successfully after an inner compile failure.
if grep -E 'error:|Exception occur{1,2}ed|ERROR.*compile' "$candidate_root/pass2/build-$variant.log"; then exit 1; fi
bash build/cann-ops-transformer-native_int4_w4a8_linux-x86_64.run --install-path="$variant_dir" \
    > "$candidate_root/pass2/install-$variant.log" 2>&1
sed "s/ops-delivery/ops-pass2-$variant/g" "$candidate_root/run-python.sh" \
    > "$candidate_root/pass2/run-python-$variant.sh"
sed "s/ops-delivery/ops-pass2-$variant/g" "$candidate_root/run-device-tests.sh" \
    > "$candidate_root/pass2/run-device-tests-$variant.sh"
