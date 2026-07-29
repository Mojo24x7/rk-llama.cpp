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
[`51f3ed6a6`](../../commit/51f3ed6a6). Full credits in **[CREDITS.md](CREDITS.md)**, full provenance in **[LINEAGE.md](LINEAGE.md)**.

---

## What this work adds — measured

Every figure below was measured on a 16 GB ROCK 5B+, warm runs, first run
discarded. Full tables in **[BENCHMARKS.md](BENCHMARKS.md)**.

### Headline

| improvement | before | after | gain |
|---|---|---|---|
| **MoE expert matmuls on the NPU** (batched `MUL_MAT_ID`, int8) | 5.02 s/layer | **0.518 s/layer** | **9.7×** |
| **Correct device type + KV placement** (gemma-3-1B decode) | 2.39 t/s | **17.32 t/s** | **7.2×** |
| **`ggml-rpc` weight transfer** | 38 MB/s | **280 MB/s** (wire-limited) | **7.4×** |
| **Production 30B prefill** | 8.66 t/s | **21.1 t/s** | **2.4×** |
| **Hybrid model decode** (Qwen3.6-35B, fused GDN kept alive) | 4.35 t/s | **7.99 t/s** | **+84 %** |
| **Tensor parallel vs layer pipeline** (27B dense, 3 boards) | 0.53 t/s | **1.05 t/s** | **2.0×** |
| **INT4 attention quality** (per-channel scales) | +43 % perplexity | **+5.0 %** | **8.6× less loss** |
| **Inter-device traffic** (glue-op locality) | 13.9 GB/req | **3.3 GB/req** | **4.2× less** |
| **Production 30B decode** | 6.5 t/s | **9.6 t/s** | **1.5×** |
| **Governor tuning alone** | — | — | **+32 %** |

### Capabilities that did not exist before

| capability | evidence |
|---|---|
| **MoE experts run on the NPU at all** | `MUL_MAT_ID` implemented for this backend; **9.7×** over the naive per-(token,expert) dispatch |
| **Models 4× larger than board RAM** | `gpt-oss-120b`, **60 GB on a 16 GB board**, 0.88 t/s, coherent |
| **Multi-board inference that is correct** | seven fixes; before them the remote silently no-op'd norms and RoPE and produced fluent nonsense |
| **Cross-board tensor parallelism** | `rktp` — row-split matmuls with balanced memory and N-shard support, **2× the layer pipeline** on dense models |
| **Best-in-class multi-board prefill** | 30B across 3 boards: **26.6 t/s**, vs 21.1 single board |
| **Qwen3.6-35B loads at all** | GGUFs whose `block_count` includes the MTP layer previously failed with a missing-tensor error |
| **Arm Mali GPU is recognised** | `ggml-opencl` rejected Valhall outright with `Unsupported GPU`; now enumerates and runs |
| **`-ot` can target extra buffer types** | e.g. `CPU_REPACK`, which the argument parser could not name |
| **Backend runs on current upstream** | rebased across **1514 upstream commits**; the origin branch is still on llama.cpp of 2026-05-19 |

### Hardware facts established by measurement

Not documented elsewhere as far as we can find, and each one changes how the
platform should be used:

| finding | consequence |
|---|---|
| **INT4 matmul does not batch** — linear in M, ~120 GFLOPS ceiling; INT8 reaches **1563 GFLOPS** | run `Q4_0` through the **int8** pipeline: **9.7×** on quantised matmuls |
| **Only symmetric precisions exist** — W16A16 / W8A8 / W4A4; no W8A16, W4A16, W4A8 | 4-bit weights cannot pair with higher-precision activations, by hardware |
| **NPU bandwidth is 23.2 GB/s, not ~11** | the NPU is *not* memory-starved relative to the CPU's 22.9 GB/s |
| **CPU + NPU do not sum** — 27.29 GB/s aggregate vs 23 alone | cross-device concurrency is capped at ~+19 % on this SoC |
| **A55 cores cost 56 %** (`-t 8` = 4.26 vs `-t 4` = 9.66) | pin to the four A76 cores, always |
| **Residency dominates everything** — 9.15 vs 6.00 t/s, same model, same board | see **[MEMORY-RESIDENCY.md](MEMORY-RESIDENCY.md)** |
| **Tensor parallelism is bandwidth-bound on 2.5 GbE** — 512 FLOP/byte offered vs 536 machine balance | predicted before measuring; a 10 GbE fabric would flip it |

### Also fixed

- **`supports_op` guard** — KV-cache views were being claimed as weights, tripping
  an assert. Now only genuine preloaded weights are claimed, with CPU fallback.
- **`RKNPU_GLUE` honours its value** — it previously enabled on mere *presence*, so
  `RKNPU_GLUE=0` did nothing and two of our own A/B comparisons were
  flag-on versus flag-on.
- **Uninitialised `rknn_matmul_io_attr`** — worked by luck in one process, failed
  on an RPC server thread's stack.
- **Per-weight metadata keyed by tensor pointer** — the tensor object at load
  differs from the one at compute under RPC. Re-keyed by buffer offset.
- **Oversized tensors excluded** above the ~2 GB IOMMU domain limit.
- **`ggml-vulkan` builds against Vulkan headers older than 1.3.272.**

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

## Running modes — four, all measured

This fork supports every distribution scheme the hardware allows, and each was
measured rather than assumed. Pick by what you need.

| mode | boards | Qwen3-30B-A3B Q4_0 | what it unlocks |
|---|---|---|---|
| **A. single board** | 1 | **PP 21.1 · TG 9.6** | fastest end-to-end; the deployed configuration |
| **B. single board, above RAM** | 1 | 60 GB model → 0.88 t/s | **models 4× board RAM** |
| **C. multi-board serial** (`-sm layer`) | 2-4 | **PP 26.6** · TG 2.5 | **highest prefill of any mode** |
| **D. multi-board parallel** (`rktp`) | 2-4 | 27B dense **1.05** vs 0.53 serial | **2× the layer pipeline** on dense models |

Modes C and D both let a model run that does not fit one board. Mode D is this
fork's own contribution and beats the standard layer pipeline **2×** on dense
models, because the pipeline activates one board at a time while tensor
parallelism has every board computing on every matmul.

### A. Single board

```bash
RKNPU_HYBRID=W8A8_STANDARD RKNPU_GLUE=1 \
ulimit -n 1000000 && \
taskset -c 4-7 ./build/bin/llama-server -m model.gguf \
    -ngl 99 -t 4 -c 16384 -fa on --host 0.0.0.0 --port 8080
```

### B. Single board, MoE larger than RAM

Only the routed experts are read per token, so the model may be several times RAM.
**A 60 GB model runs on a 16 GB board.**

```bash
RKNPU_HYBRID=W8A8_STANDARD RKNPU_GLUE=1 \
ulimit -n 1000000 && \
taskset -c 4-7 ./build/bin/llama-server -m big-moe.gguf \
    -ngl 99 --cpu-moe --no-repack -fit off \
    -t 4 -c 16384 -fa on --host 0.0.0.0 --port 8080
```

Add `--override-kv <arch>.expert_used_count=int:4` for **+32 % decode** at
**+14.6 % perplexity** — a documented trade rather than a hidden one.

### C. Multi-board serial — highest prefill

Layers are divided across boards, over upstream `ggml-rpc`. **30B: PP 26.6 t/s**,
the best prefill of any configuration here.

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
`rpc-server` — device type and memory are evaluated in the remote process.
Without it the scheduler gives the remote board no layers and the split silently
does nothing.

### D. Multi-board parallel — tensor parallelism

Every board computes part of each matmul simultaneously. **2× the layer pipeline**
on dense models (27B: 1.05 vs 0.53 t/s).

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

### Choosing between them

- **Model fits one board** → mode A. Distribution adds network cost with nothing to
  win, and we measured it: gemma-4-12B is 1.12-1.54 t/s on one board and 1.29 on
  three.
- **Model too big, want throughput** → mode B if MoE (a 60 GB model runs), mode D
  if dense (**2×** the layer pipeline).
- **Prompt-heavy workload** (RAG, long context, batch) → mode C, **PP 26.6**.

Tensor parallelism is bandwidth-bound on 2.5 GbE — 512 FLOP/byte offered against a
~536 FLOP/byte machine balance — which is why it wins where the alternative uses
one board at a time, and why a 10 GbE fabric would change the picture entirely.
Arithmetic and per-scheme diagrams in **[MULTI-BOARD.md](MULTI-BOARD.md)**.

### Operational notes worth knowing

Each of these cost real debugging time and is invisible from the outside:

- **`RKNPU_TP_LOCFRAC` must land on the backend's alignment boundary.** `0.6`
  produces plausible token rates and **silently wrong output** — recorded as
  "1.06 t/s" before anyone read the text. Use 0.5 or 0.34.
- **`RKNPU_HYBRID` must be identical** on coordinator and shards, or the shard
  aborts on a missing Hadamard vector.
- **`tp_shard` is not a CMake target**, so `make` will not rebuild it. After
  changing it or the wire protocol, rebuild **and** rsync its ggml libraries —
  a fresh binary against a stale `libggml-base.so` crashes inside
  `graph_compute`. Use `rebuild_tp_shard.sh`.

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

## Which GGUF quantisations work on the NPU

**This matters more than anything else when choosing a model file.** The backend
requantises weights into the NPU's native formats at load, and it can only do that
for a specific set of GGUF types. Everything else silently falls back to the CPU —
it will run, but the NPU contributes nothing.

| GGUF type | NPU-eligible | maps to | notes |
|---|---|---|---|
| **`Q4_0`** | ✅ | `W4A4_HADAMARD` by default | **run it as int8 instead** — see below |
| **`Q8_0`** | ✅ | `W8A8_STANDARD` | safest default |
| **`Q6_K`** | ✅ | alternates int8 / int4 per tensor | half its tensors take the int4 path |
| **`F16`** | ✅ | `W16A16` | 2× the bytes, no quality gain over int8 in practice |
| `Q4_K_M`, `Q4_K_S`, `Q3_K`, `Q5_K`, `IQ*` | ❌ | — | **CPU only**; the NPU is unused |
| `MXFP4` (e.g. native gpt-oss) | ❌ | — | **CPU only** |

### Recommended pairings

| what you have | what to run | why |
|---|---|---|
| `Q4_0` | `RKNPU_HYBRID=W8A8_STANDARD` | INT4 does not batch (~120 GFLOPS ceiling, linear in M); int8 reaches **1563 GFLOPS**. Worth **~9.7×** on quantised matmuls. Costs 2× the bytes per weight — accepted, because there is no 4-bit-weight pipeline that keeps activation precision. |
| `Q8_0` | default (`W8A8_STANDARD`) | already matched |
| `Q6_K` / `F16` | `W8A8_STANDARD` | feeding a higher-precision file into a lower-precision pipeline gives the requantiser an accurate reference and the best quality of any option |
| any K-quant | reconsider | works, but on the CPU only. For a **MoE with `--cpu-moe`** this is fine — the experts are CPU-side anyway — and `Q3_K_M` measured ~7.6 t/s on the 30B. |

If you need INT4 on the NPU, set **`RKNPU_PERCHAN=1`**. Per-output-channel scales
take INT4 attention from **+43 % to +5.0 %** perplexity versus int8; the default
per-block scales (one scale per ~2.8 M weights) are too coarse for 4-bit and
produce visibly wrong answers.

### Quantisations measured here

| model | quant | file | PP | TG | coherent |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | **Q4_0** | 17.3 GB | **21.1** | **9.6** | ✅ deployed |
| Qwen3-30B-A3B | Q8_0 | 31 GB | — | 1.2-2.3 | ✅ |
| Qwen3-30B-A3B | Q3_K_M (CPU only) | 14.7 GB | — | ~7.6 | ✅ |
| Qwen3.6-35B-A3B | Q4_0 | 20.8 GB | 18.5 | 4.07 | ✅ |
| Qwen3.6-27B dense | Q8_0 | 28.6 GB | — | 1.05 (3 boards) | ✅ |
| Qwen3.6-27B dense | Q4_0 | 15.8 GB | — | ~0.5 (I/O bound) | ✅ |
| gemma-4-12B dense | Q8_0 | 13 GB | — | 1.12-1.54 | ✅ |
| gemma-4-E4B | Q8_0 | 8.2 GB | 38.2 | 4.04 | ✅ |
| gemma-3-1B | Q8_0 | 1 GB | 201.6 | 17.0 | ✅ |
| gpt-oss-120b | Q8_0 | 60 GB | — | 0.88 | ✅ |
| gpt-oss-20b | MXFP4 (CPU only) | 12 GB | 13.1 | 6.1-8.0 | ❌ incoherent |
| gpt-oss-20b | Q4_0 requantised | 11.5 GB | 13.7 | 7.5 | ❌ incoherent |

Perplexity across five quantisation and routing configurations is in
**[BENCHMARKS.md](BENCHMARKS.md)** §5.

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
| **[CREDITS.md](CREDITS.md)** | who wrote which part, with links |
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

## Status

Validated against the previous base on an idle board, warm runs, A/B/A ordering:
performance-neutral within noise (30B: PP 25.5 vs 24.7, TG 6.35 vs 6.36). The
rebase is for capability and maintainability, not speed.

## Licence

MIT, inherited from llama.cpp. The vendored backend is under the same terms as its
origin repository.
