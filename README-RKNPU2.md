# llama.cpp with Rockchip RK3588 NPU support

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) carrying the
**RKNPU2 backend**, kept rebased onto current upstream master.

> The RKNPU2 backend was written by [@invisiofficial](https://github.com/invisiofficial)
> and contributors in [invisiofficial/rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp).
> Upstream llama.cpp has no Rockchip NPU backend. See [LINEAGE.md](LINEAGE.md) for
> exact provenance and the commands to verify it.

Branch: **`rknpu2`** — upstream master + 3 commits.

## Why this fork exists

`invisiofficial/rk-llama.cpp` has not been updated since 2026-05-20, so it sits
on llama.cpp of 2026-05-19. This fork carries the same backend on **current**
upstream, which brings newer model architectures, `mtmd` vision/audio graphs and
upstream fixes that RK3588 users otherwise cannot reach.

Keeping it current is cheap by construction. The backend is **96% self-contained**:
of its 4187 lines, 4133 live in `ggml/src/ggml-rknpu2/`, a directory upstream never
touches. Only **15 lines** cross into shared files:

| file | change |
|---|---|
| `.gitignore` | allow `librknnrt.so` to be tracked |
| `CMakeLists.txt` | `option(LLAMA_RKNPU2)` → `GGML_RKNPU2` |
| `ggml/CMakeLists.txt` | `option(GGML_RKNPU2)` |
| `ggml/src/CMakeLists.txt` | `ggml_add_backend(RKNPU2)` |
| `ggml/src/ggml-backend-reg.cpp` | include + `register_backend` |

Plus whatever backend-interface slots upstream has added since the last rebase
(most recently four: `set/get_tensor_2d` on `ggml_backend_buffer_i` and
`set/get_tensor_2d_async` on `ggml_backend_i`, all `NULL` for this backend).

So a rebase is a routine operation:

```bash
git fetch origin && git rebase origin/master   # expect only vtable churn
cd build && make -j3
```

## Build

```bash
cmake -B build -DLLAMA_RKNPU2=ON -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -C build -j3 llama-server llama-cli
```

Confirm the backend is present:

```console
$ ./build/bin/llama-cli --list-devices
Available devices:
  RKNPU: Rockchip NPU (0 MiB, 0 MiB free)
```

`0 MiB` is correct — this backend reports as a compute accelerator with no
memory of its own by default, so llama.cpp places the KV cache on the host
where it belongs. See `RKNPU_AS_GPU` below.

At runtime the process needs a raised file-descriptor limit, because each
resident matmul context holds DMA handles:

```bash
ulimit -n 1000000
```

## Running

Pin to the four Cortex-A76 cores and set the governors to `performance`; both
matter measurably on this SoC.

```bash
taskset -c 4-7 ./build/bin/llama-server -m model.gguf \
    -ngl 99 -t 4 -fa on --host 0.0.0.0 --port 8080
```

### Models larger than RAM

MoE models substantially larger than physical RAM run well, because only the
routed experts are read per token. Keep the expert weights memory-mapped rather
than repacked into RAM:

```bash
--cpu-moe --no-repack -fit off
```

- `--no-repack` is **required**. A full repack materialises every expert in RAM
  and will exhaust a 16 GB board on a 17 GB model.
- `-fit off` is currently required for any model larger than free system memory:
  upstream's memory-fit heuristic aborts rather than proceeding, because it does
  not model a memory-mapped buffer as reclaimable.

Measured on 16 GB: a 20.8 GB model loads in 35 s; a 60 GB model runs at
0.88 t/s. See [MEMORY-RESIDENCY.md](MEMORY-RESIDENCY.md) for how residency affects
throughput.

## Quantisation on this hardware

RK3588 implements only **symmetric** matmul precisions — `W16A16`, `W8A8`,
`W4A4`. There is no `W8A16`, `W4A16` or `W4A8` (probed directly against
`rknn_matmul_create`). 4-bit weights therefore cannot be paired with
higher-precision activations.

Separately, **INT4 matmul does not batch on this chip**: latency is linear in M
and caps around 120 GFLOPS. INT8 scales properly — 19 GFLOPS at M=1 rising to
1563 at M=1024. In practice a `Q4_0` GGUF is best run through the INT8 pipeline:

```bash
RKNPU_HYBRID=W8A8_STANDARD
```

This costs 2x the bytes per weight on the NPU relative to the CPU path, which is
a real trade rather than an oversight — there is no 4-bit-weight pipeline that
keeps activation precision.

## Environment variables added by this fork

| variable | effect |
|---|---|
| `RKNPU_AS_GPU=1` | report GPU device type and real system memory instead of accelerator/0. Needed only for multi-board weight distribution; also restores the meaning of `-ngl 0`. |
| `RKNPU_GLUE=0/1` | let the backend claim cheap non-matmul ops and run them on-device, cutting graph splits. Now honours its **value** (it previously enabled on mere presence). |
| `RKNPU_GLUE_THREADS=N` | thread count for that path. |
| `RKNPU_PERCHAN=1` | per-output-channel quantisation scales. Takes INT4 attention from +43% to +5.0% perplexity versus INT8 on `Qwen3-30B-A3B`. |
| `RKNPU_TP_SHARD=host:port[,host:port]` | cross-board tensor parallelism (see below). |
| `LLAMA_RECURRENT_ON_CPU=1` | keep recurrent layers on the CPU device so the fused Gated Delta Net kernel stays enabled on hybrid models. Now redundant on current upstream; retained pending removal. |

Upstream `RKNPU_HYBRID`, `RKNPU_CORES`, `RKNPU_DOMAINS` and `RKNPU_DEVICE` are
documented in [`ggml/src/ggml-rknpu2/README.md`](ggml/src/ggml-rknpu2/README.md)
(Invisi's own document, vendored verbatim).

## What this fork adds beyond the vendored backend

Commit `51f3ed6a6`. All figures measured on ROCK 5B+ / 16 GB / driver 0.9.8.

- **MoE experts on the NPU** via `MUL_MAT_ID`, with per-expert routing grouped
  into one matmul per routed expert and fixed-height chunking to stay inside the
  ~64k DMA handle space.
- **Accelerator device type by default** rather than GPU-with-memory. Restores
  correct KV placement and removes the previously mandatory `-nkvo`/`-fit off`:
  **7.2x decode on `gemma-3-1b`** (2.39 → 17.32 t/s).
- **Glue-op locality**, cutting graph splits 482 → 196 and inter-device traffic
  13.9 → 3.3 GB per request under RPC.
- **`supports_op` guard** so only genuine preloaded weights are claimed —
  KV-cache views were being claimed, tripping an assert — with graceful CPU
  fallback.
- **Per-channel quantisation scales** (see `RKNPU_PERCHAN` above).
- **Cross-board tensor parallelism** (`rktp.h`, `tp_shard.cpp`): row-split
  matmuls over TCP with balanced memory and N-shard support.
  `Qwen3.6-27B` across three boards: 0.79 t/s versus 0.53 for the sequential
  layer split. Note it is a **loss on prefill** — 2.5 GbE offers 512 FLOP/byte
  against a 536 FLOP/byte machine balance, so tensor parallelism over this fabric
  is bandwidth-bound by arithmetic rather than by implementation.
- **`ggml-rpc` `set_tensor` fast path**: removes a byte-at-a-time hash over every
  >10 MB tensor, an extra round trip and a zero-initialised staging copy. Raw
  wire was 280 MB/s while RPC achieved 38. Backend-agnostic.
- **`-ot` / `--override-tensor` can name a device's extra buffer types**
  (e.g. `CPU_REPACK`), which the argument parser omitted.
- **Arm Mali (Valhall) OpenCL support**, so `ggml-opencl` stops rejecting the GPU
  outright. Enablement only: measured 22x slower than the CPU on this SoC.

## Validation

Against the previous base, otherwise-idle board, warm runs only, A/B/A ordering:

| model | prefill | decode |
|---|---|---|
| `Qwen3-30B-A3B-Q4` | 25.51 vs 24.70 t/s | 6.35 vs 6.36 t/s |
| `Qwen3.6-35B-A3B-Q4` | 18.51 vs 18.97 t/s | 4.07 vs 3.95 t/s |

Performance-neutral within noise — the rebase is for capability and
maintainability, not speed.

## Licence

MIT, inherited from llama.cpp. The vendored backend is under the same terms as
its origin repository.
