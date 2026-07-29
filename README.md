# llama.cpp on the Rockchip RK3588 NPU

Run local LLMs on the NPU of an RK3588 board (Radxa ROCK 5B+, Orange Pi 5, etc.),
using current upstream llama.cpp.

Branch **`rknpu2`** = [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
master + 4 commits.

---

## Credits — who wrote what

**The RKNPU2 backend is not our work.** It was written by
**[@invisiofficial](https://github.com/invisiofficial)** in
**[invisiofficial/rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp)**,
with contributions from
[@Polarnik](https://github.com/Polarnik) (zero-copy weights),
[@hvalev](https://github.com/hvalev),
[@woefulwabbit](https://github.com/woefulwabbit) (cross-compilation) and
[@MartinoMensio](https://github.com/MartinoMensio).
12 commits, 2025-10-17 to 2026-05-19. **Upstream llama.cpp has no Rockchip NPU
backend at all** — without their work none of this exists.

It is vendored here in commit
[`c180473d4`](../../commit/c180473d4), authored to Invisi with `Co-authored-by:`
trailers for the others, so `git blame` on the backend points at them and not at
us.

**[@danielferr85](https://github.com/danielferr85)** independently rebased the
same backend onto llama.cpp of 2026-07-14 in
[danielferr85/rk-llama.cpp](https://github.com/danielferr85/rk-llama.cpp).
That work is not in our history, but his commit `71fa525e` is what identified
which backend interface slots had changed since May, which saved us the search.

**What we added** is the rebase onto current upstream plus backend engineering —
MoE experts on the NPU, cross-board tensor parallelism, quantisation fixes and a
measurement study. Commits [`95a113dc7`](../../commit/95a113dc7) and
[`51f3ed6a6`](../../commit/51f3ed6a6). Full provenance in **[LINEAGE.md](LINEAGE.md)**.

---

## Why this fork

`invisiofficial/rk-llama.cpp` was last updated 2026-05-20, so it sits on llama.cpp
of 2026-05-19. This fork carries the same backend on **current** upstream, which
brings newer model architectures, vision/audio graphs and upstream fixes that
RK3588 users otherwise cannot reach.

Staying current is cheap by construction: the backend is **96 % self-contained**
(4133 of 4187 lines live in `ggml/src/ggml-rknpu2/`), so a rebase touches
**15 lines** of shared code plus any new backend-interface slots.

```bash
git fetch origin && git rebase origin/master
cd build && make -j3
```

## Build

```bash
cmake -B build -DLLAMA_RKNPU2=ON -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -C build -j3 llama-server llama-cli
ulimit -n 1000000          # each resident matmul context holds DMA handles
```

```console
$ ./build/bin/llama-cli --list-devices
Available devices:
  RKNPU: Rockchip NPU (0 MiB, 0 MiB free)
```

`0 MiB` is correct — the backend reports as a compute accelerator with no memory
of its own, so llama.cpp keeps the KV cache on the host where it belongs.

## Running modes — pick one

| mode | when | boards | example result (Qwen3-30B-A3B Q4_0) |
|---|---|---|---|
| **A. single board** | model fits RAM | 1 | **PP 21.1 · TG 9.6** ← best overall |
| **B. single board, larger than RAM** | MoE bigger than RAM | 1 | 60 GB model → TG 0.88 |
| **C. multi-board serial** (`-sm layer`) | dense model too big; simplest | 2-4 | PP **26.6** · TG 2.5 |
| **D. multi-board parallel** (`rktp`) | dense model too big; better decode | 2-4 | 27B dense: TG **1.05** vs 0.53 serial |

**If the model fits one board, use mode A.** Measured repeatedly: adding boards to
a model that already fits does not help (gemma-4-12B: 1 board 1.12-1.54,
2 boards 1.12, 3 boards 1.29). Only distribute when you must.

### A. Single board

```bash
RKNPU_HYBRID=W8A8_STANDARD RKNPU_GLUE=1 \
ulimit -n 1000000 && \
taskset -c 4-7 ./build/bin/llama-server -m model.gguf \
    -ngl 99 -t 4 -c 16384 -fa on --host 0.0.0.0 --port 8080
```

### B. Single board, MoE larger than RAM

Only the routed experts are read per token, so the model may be several times RAM.

```bash
RKNPU_HYBRID=W8A8_STANDARD RKNPU_GLUE=1 \
ulimit -n 1000000 && \
taskset -c 4-7 ./build/bin/llama-server -m big-moe.gguf \
    -ngl 99 --cpu-moe --no-repack -fit off \
    -t 4 -c 16384 -fa on --host 0.0.0.0 --port 8080
```

Add `--override-kv <arch>.expert_used_count=int:4` to halve the experts read per
token: **+32 % decode, +14.6 % perplexity**. A real trade, not a free win.

### C. Multi-board, serial (layer pipeline)

Boards take turns, one active at a time. Uses upstream `ggml-rpc`.

**On each remote board:**
```bash
cd rk-llama.cpp
LD_LIBRARY_PATH=build/bin:ggml/src/ggml-rknpu2/libs \
RKNPU_AS_GPU=1 ulimit -n 65536 && \
./build/bin/rpc-server -H 0.0.0.0 -p 50060 -d RKNPU
```

**On the coordinator:**
```bash
RKNPU_AS_GPU=1 RKNPU_HYBRID=W8A8_STANDARD RKNPU_GLUE=1 \
ulimit -n 1000000 && \
taskset -c 4-7 ./build/bin/llama-server -m model.gguf \
    --rpc <boardB>:50060,<boardC>:50060 -sm layer \
    -ngl 99 -fit off -nkvo -t 4 -c 4096 -fa on --host 0.0.0.0 --port 8080
```

★ `RKNPU_AS_GPU=1` must be set on **both** the coordinator **and** every
`rpc-server` — device type and memory are evaluated in the remote process. Without
it the scheduler gives the remote board no layers and the split silently does
nothing.

### D. Multi-board, parallel (tensor parallel)

Every board computes part of each matmul simultaneously. Better decode than mode C
for dense models.

**On each shard board:**
```bash
cd rk-llama.cpp
LD_LIBRARY_PATH=build/bin:ggml/src/ggml-rknpu2/libs \
RKNPU_HYBRID=W8A8_STANDARD ulimit -n 65536 && \
./build/bin/tp_shard --role shard --port 48200
```

**On the coordinator:**
```bash
RKNPU_TP_SHARD=<shardA>:48200,<shardB>:48200 \
RKNPU_TP_MINN=2048 RKNPU_TP_LOCFRAC=0.34 \
RKNPU_TP_OFFLOAD=0.5 RKNPU_TP_FULLNPU=1 \
RKNPU_HYBRID=W8A8_STANDARD RKNPU_GLUE=1 \
ulimit -n 1000000 && \
taskset -c 4-7 ./build/bin/llama-server -m model.gguf \
    -ngl 99 -t 4 -c 4096 -fa on --host 0.0.0.0 --port 8080
```

| variable | meaning |
|---|---|
| `RKNPU_TP_SHARD` | comma-separated shard endpoints. One shard = 2 boards, two = 3 boards. |
| `RKNPU_TP_LOCFRAC` | fraction of output rows kept locally. **0.5 for 2 boards, 0.34 for 3.** |
| `RKNPU_TP_MINN` | minimum output width to split. 2048 splits most projections. |
| `RKNPU_TP_OFFLOAD` | fraction of non-splittable tensors shipped whole, for memory balance. |
| `RKNPU_TP_FULLNPU=1` | keep all weights on the NPU rather than spilling to CPU. |

★ **`RKNPU_TP_LOCFRAC` must land on the backend's alignment boundary.** A value
like `0.6` produces plausible token rates and **silently wrong output** — it was
measured at "1.06 t/s" before anyone read the text. Use 0.5 or 0.34.

★ `RKNPU_HYBRID` must be **identical** on coordinator and shards. A coordinator on
`W8A8_STANDARD` with a shard defaulting `Q4_0` to `W4A4_HADAMARD` aborts on a
missing Hadamard vector.

★ After changing `tp_shard.cpp` or the wire protocol, rebuild the shard **and
rsync its ggml libraries** — `tp_shard` is not a CMake target, so `make` will not
rebuild it, and a fresh binary against stale `libggml-base.so` crashes inside
`graph_compute`. Use `rebuild_tp_shard.sh`.

**Do not expect decode to improve on 2.5 GbE.** Tensor parallelism offers
512 FLOP/byte against a ~536 FLOP/byte machine balance, so the link is the binding
constraint — measured on the 30B: prefill **-15 %**, decode **-23 %** versus a
single board. It wins on the dense 27B only because that case is latency-bound at
M=1 and the alternative uses one board at a time. Full arithmetic in
[MULTI-BOARD.md](MULTI-BOARD.md).

## Flag reference

Flags that matter on this platform, and why.

| flag | effect | notes |
|---|---|---|
| `-ngl 99` | offload all layers | the backend takes what it can run |
| `-t 4` | 4 threads | **never more** — A55 cores cost 56 % |
| `taskset -c 4-7` | pin to Cortex-A76 | big.LITTLE barrier sync otherwise |
| `-fa on` | flash attention | large free prefill win; `-fa` off cost 21.1 → 8.2 |
| `--cpu-moe` | MoE experts on CPU | required for MoE larger than RAM |
| `--no-repack` | keep experts memory-mapped | **mandatory** with `--cpu-moe` above RAM; a repack materialises every expert in RAM |
| `-fit off` | skip the memory-fit check | required above free RAM — upstream aborts, not treating mmap as reclaimable |
| `-nkvo` | KV cache on host | needed with `RKNPU_AS_GPU=1`; not otherwise |
| `-c N` | context | KV is host-side; 16384 is comfortable on 16 GB |
| `--override-kv <arch>.expert_used_count=int:N` | fewer experts per token | +32 % decode at +14.6 % perplexity for 8 → 4 |
| `-ot <regex>=CPU` | force tensors to a buffer type | this fork also allows naming extra bufts, e.g. `CPU_REPACK` |
| `--jinja` | use the model's chat template | needed for tool-calling models |

**Governors first, always.** `performance` on CPU, DRAM and NPU is worth **+32 %**
and costs nothing:

```bash
for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor \
         /sys/class/devfreq/*/governor; do echo performance | sudo tee $p; done
```

## Results at a glance

16 GB ROCK 5B+, single board unless noted. PP = prefill t/s, TG = decode t/s.

| model | quant | PP | TG |
|---|---|---|---|
| Qwen3-30B-A3B (3.3 B active) | Q4_0 17 GB | **21.1** | **9.6** |
| Qwen3.6-35B-A3B | Q4_0 21 GB | 18.5 | 4.1 |
| gemma-4-E4B | Q8_0 | 38.2 | 4.0 |
| gemma-3-1B | Q8_0 | 201.6 | 17.0 |
| gpt-oss-120b (**4× board RAM**) | Q8_0 60 GB | — | 0.88 |
| Qwen3.6-27B **dense**, 3 boards | Q8_0 | — | 1.05 |

Full tables — 14 models, every configuration, perplexity, and **12 things that did
not work** — in **[BENCHMARKS.md](BENCHMARKS.md)**.

## Documentation

| document | contents |
|---|---|
| **[BENCHMARKS.md](BENCHMARKS.md)** | all result tables, perplexity, raw NPU matmul sweep, platform constants, speculative decoding, and the levers that failed |
| **[MULTI-BOARD.md](MULTI-BOARD.md)** | distributing a model across boards — 3 schemes with diagrams, why decode does not improve on 2.5 GbE, and the 7 fixes needed to make it work |
| **[MEMORY-RESIDENCY.md](MEMORY-RESIDENCY.md)** | why residency dominates throughput, and the benchmarking trap it creates |
| **[LINEAGE.md](LINEAGE.md)** | provenance, with commands to verify every claim |
| [ggml/src/ggml-rknpu2/README.md](ggml/src/ggml-rknpu2/README.md) | the backend's own documentation, by @invisiofficial |
| [README-llama.cpp.md](README-llama.cpp.md) | upstream llama.cpp's README |

## Environment variables added by this fork

| variable | effect |
|---|---|
| `RKNPU_AS_GPU=1` | report GPU device type and real system memory instead of accelerator/0. Needed only for multi-board weight distribution; also restores the meaning of `-ngl 0`. |
| `RKNPU_GLUE=0/1` | let the backend claim cheap non-matmul ops and run them on-device, cutting graph splits 482 → 196. Now honours its **value** (previously enabled on mere presence). |
| `RKNPU_GLUE_THREADS=N` | thread count for that path. |
| `RKNPU_PERCHAN=1` | per-output-channel quantisation scales. Takes INT4 attention from **+43 % to +5.0 %** perplexity versus int8. |
| `RKNPU_TP_SHARD=host:port[,host:port]` | cross-board tensor parallelism — see [MULTI-BOARD.md](MULTI-BOARD.md). |
| `LLAMA_RECURRENT_ON_CPU=1` | keep recurrent layers on the CPU so the fused Gated Delta Net kernel stays enabled on hybrid models. Redundant on current upstream; retained pending removal. |

Upstream's `RKNPU_HYBRID`, `RKNPU_CORES`, `RKNPU_DOMAINS` and `RKNPU_DEVICE` are
documented in [the backend's own README](ggml/src/ggml-rknpu2/README.md).

## What this fork adds to the backend

Commit [`51f3ed6a6`](../../commit/51f3ed6a6). All figures measured on
ROCK 5B+ / 16 GB / driver 0.9.8.

- **MoE experts on the NPU** via `MUL_MAT_ID`, grouping routed tokens into one
  matmul per expert instead of one dispatch per (token, expert) pair — **9.7×**
  on the expert matmuls of a layer.
- **Accelerator device type by default** rather than GPU-with-memory, restoring
  correct KV placement and removing the previously mandatory `-nkvo`/`-fit off`:
  **7.2× decode on gemma-3-1B** (2.39 → 17.32 t/s).
- **Glue-op locality** — graph splits 482 → 196, inter-device traffic
  13.9 → 3.3 GB per request under RPC.
- **`supports_op` guard** so only genuine preloaded weights are claimed
  (KV-cache views were being claimed, tripping an assert), with CPU fallback.
- **Per-channel quantisation scales** (see `RKNPU_PERCHAN`).
- **Cross-board tensor parallelism** — Qwen3.6-27B across three boards at
  **0.79–1.05 t/s** versus 0.53 for the sequential layer split.
- **`ggml-rpc` `set_tensor` fast path** — removes a byte-at-a-time hash over every
  tensor above 10 MB. Raw wire 280 MB/s, RPC was achieving 38. Backend-agnostic.
- **`-ot` / `--override-tensor` can name a device's extra buffer types**
  (e.g. `CPU_REPACK`), which the argument parser omitted.
- **Arm Mali (Valhall) OpenCL support**, so `ggml-opencl` stops rejecting the GPU.
  Enablement only — measured **22× slower than the CPU** on this SoC.

## Status

Validated against the previous base on an idle board, warm runs, A/B/A ordering:
performance-neutral within noise (30B: PP 25.5 vs 24.7, TG 6.35 vs 6.36). The
rebase is for capability and maintainability, not speed.

## Licence

MIT, inherited from llama.cpp. The vendored backend is under the same terms as its
origin repository.
