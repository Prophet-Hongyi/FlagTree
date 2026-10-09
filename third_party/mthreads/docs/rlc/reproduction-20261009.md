# MUSA RLC reproduction, 2026-10-09

This change makes the shared phased RemoveLayoutConversions implementation
available to MThreads builds. `FLAGTREE_MUSA_RLC_ENHANCE` remains **off by default**.
The table compares off/on in the same compiler build, with phase mask 15 and
default backend policy values. It does not compare two independently tuned kernels.

## Observed performance

| Operator | dtype | Input/output shape | Parameters | Off ms | On ms | Speedup |
|---|---|---|---|---:|---:|---:|
| bernoulli_ | float16 | 4096x4096 | p=0.5 | 0.06544 | 0.055 | 18.9818% |
| dropout | float16 | 4096x4096 | p=0.5,training=True | 0.13264 | 0.09692 | 36.8551% |
| masked_select | float16 | 4096x4096 | bool mask same shape; independent random input < 0.3 | 0.73612 | 0.46466 | 58.4212% |
| normal_tensor_tensor | float32 | 64x64 | mean=3,std=10 (both tensors) | 0.0119 | 0.01072 | 11.0075% |
| normal_tensor_tensor | float32 | 4096x4096 | mean=3,std=10 (both tensors) | 0.36628 | 0.32896 | 11.3448% |
| rand | float16 | 4096x4096 | uniform [0,1); contiguous output | 0.06836 | 0.05588 | 22.3336% |
| rand_like | float16 | 4096x4096 | uniform [0,1); contiguous output | 0.0684 | 0.05588 | 22.4052% |
| randn | float16 | 4096x4096 | normal mean=0,std=1; contiguous output | 0.1156 | 0.1156 | 0.0% |
| randn_like | float16 | 4096x4096 | normal mean=0,std=1; contiguous output | 0.1156 | 0.11556 | 0.0346% |
| topk | float16 | 128x32768 | k=256,dim=-1 | 0.4514 | 0.35652 | 26.6128% |
| topk | float16 | 64x1024 | k=32,dim=-1 | 0.02392 | 0.02092 | 14.3403% |
| topk | float16 | 64x8192 | k=128,dim=-1 | 0.11856 | 0.09452 | 25.4338% |
| topk | float16 | 16x1024x256 | k=256,dim=-1 | 0.58628 | 0.55764 | 5.1359% |
| uniform_ | float16 | 4096x4096 | from=0,to=1 | 0.06508 | 0.055 | 18.3273% |

Speedup is `(off_ms / on_ms - 1) * 100`; it is not percent latency reduction.
These are **one A/B round**, direct device-event measurements with the maintained
runner's repeat setting of 3. Six-round testing was not run. Cross-round stability
and model-level benefit are not established. The two FP16 normal generators show
no material gain and are included to avoid presenting historical percentages as
current results. The 5.14% three-dimensional topk result is marginal in one round.

All tensors are contiguous. For random generators, shape is the output shape.
For normal_tensor_tensor, mean and standard-deviation tensors both have the listed
shape, values 3 and 10, and output dtype FP32. For masked_select, input and bool mask
have the listed shape; output length is data-dependent. For topk, k is independent
of the input dimensions, dim=-1, largest=True, sorted=True; values are FP16 and
indices int64 with the last output dimension replaced by k.

The normal measurements cover the public multi-kernel operator, including RNG
and the affine transform; masked_select likewise includes its public multi-kernel
path. These are operator latency measurements, not timings of an arbitrarily
selected sub-kernel. Generic topk results do not establish MoE-routing gains.

See [performance CSV](performance-20261009.csv) and
[kernel/configuration CSV](kernel-configurations-20261009.csv). The latter lists
every timed kernel variant, constexpr, signature, warps/stages, layout conversions,
shared-memory bytes and executable `.text` identity. Its `shape_index` preserves
the harness label: topk's label includes k, and `161024256x256` denotes input
`[16,1024,256]`, k=256; it is not a tensor dimension of 161024256.

## Source, device and validation

- Compiler source: `8ef46b61b2d2ef496ef0d9a714b6680c51ec3003`, rebased on
  upstream `0d4c7b4a3db375693bc5f9844312c49e40430bb8`, including the newer
  component-resolution implementation. Later changes only add these receipts
  and remove a trailing blank line from an MLIR test.
- Loaded `libtriton.so` SHA256:
  `f3325ba664dd1615a19af3a571898f38c4ea89fba0fe397b6d9ef55fa3051cc9`.
- MTT S5000, architecture 31, physical GPU 7, driver 3.3.5-server,
  torch 2.7.1, Triton 3.6.0. The device was admitted idle before the serial campaign.
  Clocks were not locked.
- Existing FlagGems product source snapshot, source-manifest bound. The snapshot
  has no `.git` directory; its contents, rather than an assumed clean Git HEAD,
  define the tested product. Maintained local harness commit
  `fd047973b89dfa4dac4967b79380b834890fd10b`.
- Offline MThreads LLVM22 build passed. Optional FlagPrism and Proton were
  disabled for this compiler build; this is not full package qualification.
- MLIR phase-mask and scalar-log FileCheck checks passed. Cache-key, tails/masks,
  and MMA-selection pytest checks passed: 20 per initial switch setting, 40 total.
- Public FlagGems `--quick` correctness tests passed on both sides. Additional
  full representative outputs passed, including seeded RNG off/on bitwise
  equality at 32-bit and 64-bit Philox offsets, masked_select against a CPU oracle,
  and all four topk values, gathered indices, bounds and index uniqueness.
  The collected correctness command receipts contain 58 successful exits.
- All 52 timed kernel/shape/side records were bound to matching compile cache
  identities and TTGIR. Binaries and lower IR were recovered from correctness
  compilation caches because the maintained benchmark runner deletes its cache.
  For four generated affine-transform records only source-location definitions
  differ; the remaining TTGIR is identical. This was correctness-only artifact
  recovery, not another performance round. Executable `.text`, rather than whole
  ELF metadata, is compared in the configuration CSV.

The first bernoulli timing attempt used a nonexistent pytest node, and the first
topk attempt selected `[4096,4096], k=5`. Both diagnostic attempts are excluded
from the table. Only corrected measurements are claimed.

## Reproduction contract

Build this branch with `FLAGTREE_BACKEND=mthreads` and offline LLVM22 dependencies.
Set `PYTHONPATH` so this build's Triton and the intended FlagGems snapshot load;
verify their module paths before running. Select one idle S5000 device and use
separate Triton and FlagGems caches for each switch side.

For each operator, run its FlagGems correctness tests with the switch set to 0 and
1, then its native benchmark through the maintained runner using:

```text
SWITCH=FLAGTREE_MUSA_RLC_ENHANCE
FLAGTREE_MUSA_RLC_PHASE_MASK=15
USE_FLAGTUNE=0
ZL_BACKEND=musa
ZL_ENABLE_CUDAGRAPH=0
ZL_ISOLATE_TUNER_CACHE=1
ZL_DIRECT_REPEAT=3
BENCH_LEVEL=core
SHAPE_MODE=file
```

Use off/record then on/replay. These selected paths have no tuner, so replay is
N/A and both sides use the same source heuristics; warps/stages/constexprs are
checked in the configuration CSV. Do not replace a missing record with fresh
candidate tuning. Pass topk shape records as `[128,32768,256]`, `[64,1024,32]`,
`[64,8192,128]`, and `[[16,1024,256],256]`. The bernoulli_ benchmark node is
`benchmark/test_bernoulli.py::test_bernoulli_inplace`; normal uses
`benchmark/test_normal.py::test_normal_tensor_tensor`. Set the dtype explicitly.
The local maintained runner/plugins are a separate dependency, not shipped by
this PR; the native FlagGems default timer is not an identical timing protocol.

## Historical scope and remaining qualification

The previous 196-row history is an inventory of dated experiments, not 196
accepted gains. This PR reproduces the primary included operator families and
the two recent production normal shapes above; it does not claim to reproduce
every historical shape, dtype or protocol. Historical width8/grouped BatchNorm,
FLA state/store-lane rules, FFT budget changes and isolated router-GEMM policies
are not included. Earlier instance_norm/sort claims had already failed later
product qualification. These cannot be added to this PR's gain total.

The shared pass and JIT code also need NVIDIA-side CI/review. Native NVIDIA tests,
other MUSA devices, full production package qualification and real-model A/B
were not run here. Default enablement requires a separate acceptance decision.
