# Route-128 c3/c4 full-graph isolate

The prior four-graph attempt did not fail because 9- or 12-token decode is
unsupported. MTP2 rounds the configured `[1,2,3,4,6,8,12]` list to
`[3,6,9,12]`, and full graphs are captured largest first. The log completed
two of four captures, so the 12- and 9-token graphs had already succeeded.
The third capture failed in TP `all_reduce` with:

- `The event resources are insufficient`
- `Create capture event failed ... error=117571609`
- `aclrtAllocatorGetByStream ... stream is not registered with any allocator`
- the later `rtStreamEndCapture` error `507903`

This is cumulative graph/HCCL capture-event exhaustion. The old route-80
grouped fallback was active for 12 and 9 tokens, but both of those shapes
captured before the failure. The failure occurred while adding the third,
smaller graph. Capture ordering exposed the limit; it did not cause the
underlying exhaustion.

`serve-route128-fullgraph-c34.sh` therefore retains only `[9,12]`. This is the
smallest useful c3/c4 experiment and exactly matches the known-successful
prefix of the failed capture. It uses route-128, task queue mode 1, and leaves
HCCL AIV expansion unset. It is an experiment profile: c1/c2 pad to the
9-token graph and must be measured separately before this can replace the
general `[3,6]` service profile.

## Run

Create a fresh log so the error-signature gate cannot match an older run:

```bash
result=/srv/ai/src/native-int4-w4a8.KiuhBN/pass3/fullgraph-c34
mkdir -p "$result"
nohup bash "$result/serve-route128-fullgraph-c34.sh" \
  /srv/ai/src/native-int4-w4a8.KiuhBN/runtime-route128 0,1,2,3 8004 262144 \
  >"$result/server.log" 2>&1 &
echo $! >"$result/server.pid"
```

The launcher refuses to start if its runtime does not contain
`MAX_CUBE_ROUTES = 128`.

Generate the eager/current `[3,6]` reference first, while that service is
running:

```bash
python "$result/validate-c34-replay.py" \
  --base-url http://127.0.0.1:8001 \
  --output "$result/reference-c34.json"
```

After the `[9,12]` service is healthy, run the candidate gate:

```bash
python "$result/validate-c34-replay.py" \
  --base-url http://127.0.0.1:8004 \
  --server-log "$result/server.log" \
  --expect-full-graphs \
  --reference "$result/reference-c34.json" \
  --output "$result/candidate-c34.json"
```

The validator requires:

1. Both retained graphs finish capture, with no event-resource, allocator,
   `507903`, or engine-start failure signature.
2. c3 and c4 metrics contain positive 9- and 12-token FULL replay counts and
   zero NONE dispatches for those shapes.
3. Alternating inputs produce different outputs, while repeated inputs remain
   byte-identical across graph replays.
4. Every candidate output is byte-identical to the current/eager reference.
5. MTP drafts tokens and candidate acceptance remains within three percentage
   points of the same-workload reference.

Then collect three c3 and c4 throughput repeats with the existing harness:

```bash
for concurrency in 3 4; do
  for repeat in 1 2 3; do
    python /srv/ai/src/native-int4-w4a8.KiuhBN/pass3/bench-parallel.py \
      --base-url http://127.0.0.1:8004 \
      --concurrency "$concurrency" --max-tokens 512 \
      --output "$result/c${concurrency}-r${repeat}.json"
  done
done
```

Promotion gates are median c3 and c4 aggregate decode throughput above the
same-card `[3,6]` baseline, no reduction in maximum context from graph-pool
memory, and no regression in the fixed 228-question model evaluation. Keep
`[9,12]` as an isolate if c1/c2 padding regresses their latency materially.
