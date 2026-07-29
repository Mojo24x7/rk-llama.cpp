# Credits

## The RKNPU2 backend

Rockchip NPU support for llama.cpp was written in
**[invisiofficial/rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp)**.
**Upstream llama.cpp contains no Rockchip NPU backend** — without this work,
nothing in this repository would exist.

| author | contribution |
|---|---|
| **[Invisi](https://github.com/invisiofficial)** | the RKNPU2 backend itself; quantisation support; hardware pipelines and hybrid quantisation; IOMMU domain management; the caching system; environment variables |
| **[Polarnik](https://github.com/Polarnik)** | zero-copy for weights, weight pre-packing on load |
| **[hvalev](https://github.com/hvalev)** | zero-length segment handling (#2) |
| **[Gerald Tan](https://github.com/woefulwabbit)** | cross-compilation fixes (#8) |
| **[Martino Mensio](https://github.com/MartinoMensio)** | build fixes (#16) |

12 commits, 2025-10-17 to 2026-05-19. Vendored here in commit
[`c180473d4`](../../commit/c180473d4), authored to Invisi with `Co-authored-by:`
trailers for the others, so `git blame` on the backend attributes to them rather
than to this repository.

Their own documentation ships unmodified in
[`ggml/src/ggml-rknpu2/README.md`](ggml/src/ggml-rknpu2/README.md) and
[`CONTRIBUTING.md`](ggml/src/ggml-rknpu2/CONTRIBUTING.md).

## Upstream llama.cpp

[**ggml-org/llama.cpp**](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov
and contributors. This repository is a fork of it and tracks its master branch.

## Independent parallel work

**[danielferr85](https://github.com/danielferr85)** rebased the same backend onto
llama.cpp of 2026-07-14 in
[danielferr85/rk-llama.cpp](https://github.com/danielferr85/rk-llama.cpp),
independently of this repository. His commit
[`71fa525e`](https://github.com/danielferr85/rk-llama.cpp/commit/71fa525e)
identified which backend interface slots had changed since May — which is the
information that made our rebase a short job rather than a search. That work is
not present in our history, but the finding is his.

He has also integrated the backend into
[rkllama](https://github.com/NotPunchnox/rkllama) as an orchestration layer, so
that GGUF requests spawn a `llama-server` worker while `.rkllm` requests take the
native path.

## This repository

The rebase onto current upstream, and the backend engineering listed under
*What this work adds* in the [README](README.md) — MoE experts on the NPU,
cross-board tensor parallelism, the quantisation and correctness fixes, and the
measurement study in [BENCHMARKS.md](BENCHMARKS.md),
[MULTI-BOARD.md](MULTI-BOARD.md) and [MEMORY-RESIDENCY.md](MEMORY-RESIDENCY.md).

Commits [`95a113dc7`](../../commit/95a113dc7) and
[`51f3ed6a6`](../../commit/51f3ed6a6). Full provenance, with commands to verify
every claim above, in [LINEAGE.md](LINEAGE.md).

## Hardware

Measurements were taken on **Radxa ROCK 5B+** boards (RK3588, 16 GB LPDDR5).
