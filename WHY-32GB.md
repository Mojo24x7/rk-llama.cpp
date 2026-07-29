# Why 32 GB matters on RK3588

Measurements from a four-node Radxa ROCK 5B+ cluster running modern LLMs on the
RK3588 NPU. Every number below was taken on **16 GB** boards, and the point of
the document is narrow: to identify precisely where 16 GB is the binding
constraint and what 32 GB would change.

**Hardware.** 4x Radxa ROCK 5B+ (RK3588, 16 GB LPDDR5-4800, NVMe), Debian 12,
kernel 6.1.84, rknpu driver 0.9.8, CPU/DRAM/NPU governors pinned to
`performance`, inference threads pinned to the four Cortex-A76 cores.

**Method.** Warm runs only, first run discarded, A/B/A ordering where two
configurations are compared. NVMe read volume is sampled per request from
`/proc/diskstats`, because on this platform it is the variable that explains
almost everything.

---

## 1. The headline: the same model, the same board, 52% apart

`Qwen3-30B-A3B-Instruct-2507` at `Q4_0` is **17.3 GB**. Usable page cache on a
16 GB board is roughly 12 GB. The model therefore cannot be resident, and MoE
sparsity means each token routes to a different subset of experts.

| condition | context | decode | NVMe read **per request** |
|---|---|---|---|
| repeated identical prompt, fully warm | 9 tok | **9.15 t/s** | **3.2 MB** |
| repeated identical prompt, fully warm | 670 tok | **6.90 t/s** | **9.2 MB** |
| **unique prompt each request** | 675 tok | **5.42 / 5.88 / 6.00 t/s** | **11308 / 4347 / 6002 MB** |

The third row is what normal use looks like: every request has different text,
so it routes to different experts, so the working set never converges. The
system reads **4-11 GB from NVMe on every single request**, indefinitely.

The first row is the same model and the same board with the working set
resident. **9.15 vs 6.00 t/s is a 52% difference, and it is caused by nothing
except the model not fitting in RAM.**

At 32 GB the 17.3 GB model is fully resident, every request behaves like row 1,
and the 4-11 GB of per-request NVMe traffic disappears.

## 2. Dense models are worse, and the CPU is provably idle waiting

`Qwen3.6-27B` at `Q4_0` is **15.8 GB** and **dense** - every parameter is read
for every token, so sparsity cannot help.

```
sustained NVMe read during decode : 987 MB/s   (~76% of the NVMe ceiling)
CPU utilisation                   : 133% of 400%   (four A76 cores available)
decode                            : ~0.5 t/s
cold first token                  : > 4 minutes
```

The CPU is not the bottleneck; it is waiting on storage. Reading 15.1 GB per
token against a measured 23 GB/s memory bandwidth gives a **1.46 t/s ceiling if
resident** - roughly 3x the measured figure. That gap is storage.

## 3. Larger models already run, so RAM buys capability directly

The mmap plus MoE-sparsity approach scales well beyond RAM:

| model | file size | vs 16 GB | decode |
|---|---|---|---|
| `gpt-oss-120b` Q8_0 | 60 GB | 4x | 0.88 t/s |
| `Qwen3-30B-A3B` Q8_0 | 31 GB | 2x | ~2.3 t/s |
| `Qwen3.6-35B-A3B` Q4_0 | 20.8 GB | 1.4x | 4.07 t/s @ 676 tok |
| `Qwen3-30B-A3B` Q4_0 | 17.3 GB | 1.15x | 5.4-9.2 t/s (see 1) |

Nothing here is a soft limit. Speed tracks how much of the model is resident,
and residency is set by RAM.

## 4. Memory bandwidth is not the constraint - residency is

Measured with concurrent-master streaming benchmarks:

```
DRAM             LPDDR5-4800, 64-bit, dmc pinned 2400 MHz = 38.4 GB/s theoretical
4x Cortex-A76    22.94 / 22.80 GB/s
NPU (3 cores)    23.24 / 23.08 GB/s   (peak 29.71 at large matrices)
both concurrent  14.91 + 12.38 = 27.29 GB/s aggregate (~70% of theoretical)
```

Two things follow. The NPU is **not** memory-starved relative to the CPU, so the
usual "the accelerator has a narrow memory path" explanation does not apply
here. And the fabric sustains ~27 GB/s, which is far more than the ~1 GB/s the
NVMe delivers - so any workload forced to stream from storage is leaving most of
the machine idle.

## 5. This backend's own published benchmarks require 32 GB

The `rk-llama.cpp` RKNPU2 backend README publishes its reference figures from a
**32 GB** board. On 16 GB hardware those numbers are not reproducible for the
larger models, because they cannot be resident. There is currently no published,
methodologically careful 16 GB vs 32 GB comparison for on-device LLM inference
on RK3588.

The backend also allocates NPU memory across IOMMU domains with a per-domain
limit of roughly 4 GB, striping across them as needed. More physical RAM is
therefore not merely more page cache - it is directly more addressable resident
model.

## 6. Also measured, for completeness

Findings from the same work that bear on how this platform should be used, and
which are not documented elsewhere as far as we can tell:

- **RK3588 INT4 matmul does not batch.** Latency is linear in M and caps at
  ~120 GFLOPS, i.e. no benefit from batching at all. INT8 scales properly:
  19 GFLOPS at M=1, 1134 at M=128, 1491 at M=512, 1563 at M=1024. Consequence:
  a `Q4_0` GGUF should be run through the INT8 pipeline, not the INT4 one, and
  quantisation choices for this chip should be made accordingly.
- **The hardware implements only symmetric matmul precisions** - W16A16, W8A8
  and W4A4. There is no W8A16, W4A16 or W4A8 (probed directly against
  `rknn_matmul_create`). 4-bit weights cannot be paired with higher-precision
  activations, so running 4-bit weights on the NPU costs 2x the bytes of the
  CPU path.
- **Decode is memory-bound; prefill is compute-bound.** NPU load measures 18-27%
  across three cores through a 702-token prefill and ~0% during decode. High NPU
  utilisation is not a goal in itself: forcing the FP16 pipeline raises NPU
  utilisation from 41-43% to 51-57% while making prefill 31% and decode 36%
  *slower*.

## What we would do with 32 GB boards

Publish the missing comparison, with the code and the methodology, as a
follow-on to this repository:

1. **16 GB vs 32 GB, same board, same models, same harness** - decode, prefill
   and NVMe read volume per request, across `Qwen3-30B-A3B`, `Qwen3.6-35B-A3B`,
   a dense model and `gpt-oss-120b`. Quantifying exactly which models cross from
   unusable to usable, rather than asserting it.
2. **Reproduction of the backend's published benchmarks on the hardware they
   were taken on**, closing a documentation gap in the RKNPU ecosystem.
3. **Multi-board work that 16 GB currently makes uninteresting** - tensor
   parallelism across boards is bandwidth-bound on 2.5 GbE at these model sizes
   (measured: 512 FLOP/byte offered against a 536 FLOP/byte machine balance),
   but with resident models per board the trade changes and is worth measuring
   properly.

Contact and full working notes: this repository, branch `rknpu2`.
