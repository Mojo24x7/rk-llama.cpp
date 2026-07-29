# Memory residency and LLM throughput on RK3588

Measurements from a four-node Radxa ROCK 5B+ cluster running modern LLMs on the
RK3588 NPU. The subject is narrow: on this platform, **whether the model is
resident in RAM dominates every other variable**, including quantisation,
thread count, accelerator choice and model architecture.

**Hardware.** 4x Radxa ROCK 5B+ (RK3588, 16 GB LPDDR5-4800, NVMe), Debian 12,
kernel 6.1.84, rknpu driver 0.9.8, CPU/DRAM/NPU governors pinned to
`performance`, inference threads pinned to the four Cortex-A76 cores.

**Method.** Warm runs only, first run discarded, A/B/A ordering where two
configurations are compared. NVMe read volume is sampled per request from
`/proc/diskstats`, because on this platform it is the variable that explains
almost everything.

---

## 1. The same model, the same board, 52% apart

`Qwen3-30B-A3B-Instruct-2507` at `Q4_0` is **17.3 GB**. Usable page cache on a
16 GB board is roughly 12 GB. The model therefore cannot be fully resident, and
MoE sparsity means each token routes to a different subset of experts.

| condition | context | decode | NVMe read **per request** |
|---|---|---|---|
| repeated identical prompt, fully warm | 9 tok | **9.15 t/s** | **3.2 MB** |
| repeated identical prompt, fully warm | 670 tok | **6.90 t/s** | **9.2 MB** |
| **unique prompt each request** | 675 tok | **5.42 / 5.88 / 6.00 t/s** | **11308 / 4347 / 6002 MB** |

The third row is what varied use looks like: every request has different text, so
it routes to different experts, so the working set never converges. The system
reads **4-11 GB from NVMe on every single request**, indefinitely.

The first row is the same model on the same board with its working set resident.
**9.15 versus 6.00 t/s is a 52% difference attributable to residency alone.**

This has a practical consequence for anyone benchmarking on this platform: a
harness that repeats one prompt measures the resident case and will report
roughly 50% higher throughput than varied real use. Both numbers are valid; they
answer different questions. Stating which one is being reported matters.

## 2. Dense models are storage-bound, and the CPU is provably idle waiting

`Qwen3.6-27B` at `Q4_0` is **15.8 GB** and **dense** - every parameter is read
for every token, so sparsity cannot help.

```
sustained NVMe read during decode : 987 MB/s   (~76% of the NVMe ceiling)
CPU utilisation                   : 133% of 400%   (four A76 cores available)
decode                            : ~0.5 t/s
cold first token                  : > 4 minutes
```

The CPU is not the bottleneck; it is waiting on storage. Reading 15.1 GB per
token against a measured 23 GB/s memory bandwidth implies a **1.46 t/s ceiling
when resident** - roughly 3x the measured figure. The difference is storage.

## 3. Throughput tracks residency across model sizes

The mmap plus MoE-sparsity approach works well beyond physical RAM, and the
resulting throughput tracks how much of the model can stay resident:

| model | file size | vs 16 GB | decode |
|---|---|---|---|
| `gpt-oss-120b` Q8_0 | 60 GB | 4x | 0.88 t/s |
| `Qwen3-30B-A3B` Q8_0 | 31 GB | 2x | ~2.3 t/s |
| `Qwen3.6-35B-A3B` Q4_0 | 20.8 GB | 1.4x | 4.07 t/s @ 676 tok |
| `Qwen3-30B-A3B` Q4_0 | 17.3 GB | 1.15x | 5.4-9.2 t/s (see 1) |

None of these fail to run. Speed is set by residency.

## 4. Memory bandwidth is not the constraint

Measured with concurrent-master streaming benchmarks:

```
DRAM             LPDDR5-4800, 64-bit, dmc pinned 2400 MHz = 38.4 GB/s theoretical
4x Cortex-A76    22.94 / 22.80 GB/s
NPU (3 cores)    23.24 / 23.08 GB/s   (peak 29.71 at large matrices)
both concurrent  14.91 + 12.38 = 27.29 GB/s aggregate (~70% of theoretical)
```

Two conclusions. The NPU is **not** memory-starved relative to the CPU, so the
common "the accelerator has a narrow memory path" explanation does not apply on
this SoC. And the fabric sustains ~27 GB/s against roughly 1 GB/s from NVMe, so a
workload forced to stream from storage leaves most of the machine idle.

An earlier measurement of ours put NPU bandwidth at ~11 GB/s and was wrong: it
used a 1.57 MB weight matrix, small enough that per-dispatch overhead dominated.
At realistic matrix sizes the NPU sustains 23 GB/s. Bandwidth benchmarks on this
platform need matrices of at least ~8 MB to be meaningful.

## 5. Two hardware constraints worth knowing

Probed directly rather than inferred:

- **RK3588 implements only symmetric matmul precisions** - `W16A16`, `W8A8` and
  `W4A4`. There is no `W8A16`, `W4A16` or `W4A8`; `rknn_matmul_create` rejects
  them on this platform. 4-bit weights therefore cannot be paired with
  higher-precision activations, so running a `Q4_0` model through the NPU costs
  2x the bytes of the CPU path. That is a hardware property, not a tuning
  oversight.
- **INT4 matmul does not batch.** Latency is linear in M and caps around
  120 GFLOPS, so there is no benefit from batching at all. INT8 scales properly:
  19 GFLOPS at M=1, 1134 at M=128, 1491 at M=512, 1563 at M=1024. In practice a
  `Q4_0` GGUF is better run through the INT8 pipeline
  (`RKNPU_HYBRID=W8A8_STANDARD`) than the INT4 one.

## 6. Decode is memory-bound; prefill is compute-bound

NPU load measures 18-27% across three cores through a 702-token prefill, and
approximately 0% during decode. That is expected: decode reads the active weights
once per token and does little arithmetic with them, so it is bound by memory
rather than compute, and the NPU's 23.2 GB/s does not beat the CPU's 22.9 GB/s.

High NPU utilisation is therefore not a useful target in itself. Forcing the FP16
pipeline raises NPU utilisation from 41-43% to 51-57% while making prefill 31%
and decode 36% **slower**, because each matmul moves twice the bytes for no gain
in precision.

## Open questions

Things this work did not settle, recorded so they are not mistaken for
conclusions:

1. **No controlled 16 GB versus 32 GB comparison exists.** Section 1 infers the
   resident case from the same board's warm-cache behaviour, which is sound but
   is not the same as measuring a board that can hold the model outright. The
   RKNPU2 backend's own published reference figures were produced on a 32 GB
   board, so for the larger models they are not reproducible on 16 GB hardware -
   a gap in the ecosystem's documentation rather than in the backend.
2. **The backend allocates NPU memory across IOMMU domains** with a per-domain
   limit of roughly 4 GB, striping as needed. How addressable resident capacity
   scales with physical RAM across domains has not been characterised.
3. **Cross-board tensor parallelism is bandwidth-bound at these model sizes.**
   Measured on 2.5 GbE: 512 FLOP/byte offered against a 536 FLOP/byte machine
   balance, so the fabric is the limit rather than the implementation. Whether
   that trade changes when each board holds a resident model is untested.

Working notes and reproduction scripts: this repository, branch `rknpu2`.
