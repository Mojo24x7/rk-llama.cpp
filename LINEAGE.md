# Lineage and provenance

This fork exists to run llama.cpp on the **Rockchip RK3588 NPU**. It is assembled
from three sources, and this document states exactly which parts come from where,
with the commands to verify each claim yourself.

## Summary

```
ggml-org/llama.cpp                    upstream llama.cpp.
        |                             Contains NO Rockchip NPU backend.
        v
invisiofficial/rk-llama.cpp           Where RK3588 NPU support was written.
   branch rknpu2                      12 commits, 2025-10-17 .. 2026-05-19,
                                      by Invisi with Polarnik, hvalev,
                                      Gerald Tan and Martino Mensio.
        |
        v
Mojo24x7/rk-llama.cpp                 This repository.
   branch rknpu2                      = current ggml-org/llama.cpp master
                                      + that backend, vendored with attribution
                                      + our engineering on top.
```

The RKNPU2 backend is **not our work**. It was written by
[@invisiofficial](https://github.com/invisiofficial) and contributors in
[invisiofficial/rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp).
Our contribution is the third commit on this branch, plus keeping the whole
thing rebased onto current upstream.

## Verify it

**Upstream llama.cpp has no Rockchip NPU support.**

```console
$ git ls-tree -r --name-only origin/master | grep -c rknpu
0
$ git grep -c rknpu origin/master -- ggml/src/CMakeLists.txt ggml/src/ggml-backend-reg.cpp
0
```

**This branch sits directly on upstream master, nothing in between.**

```console
$ git merge-base rknpu2 origin/master
f5b9bd39b56c7a7839a9795a100b6a00b84ac961
$ git rev-parse origin/master
f5b9bd39b56c7a7839a9795a100b6a00b84ac961     # identical
$ git rev-list --count origin/master..rknpu2
3
```

**The backend is credited to its authors in git itself.**

```console
$ git log --format='%h %an <%ae>  %s' -- ggml/src/ggml-rknpu2/ggml-rknpu2.cpp
51f3ed6a6 homelab <lab@rocklabs>              rknpu2: RK3588 NPU backend work from the Rocklabs lab
95a113dc7 homelab <lab@rocklabs>              ggml-rknpu2: wire the vendored backend into build and ABI
c180473d4 Invisi <invisiofficial@gmail.com>   ggml-rknpu2: vendor the RK3588 NPU backend from invisiofficial
```

Commit `c180473d4` is authored to Invisi and carries `Co-authored-by:` trailers
for Polarnik, hvalev, Gerald Tan and Martino Mensio. It contains their work only
and does not build on its own; the build wiring and the current backend ABI are
added separately in `95a113dc7` so that the boundary between their work and ours
is unambiguous in `git blame`.

## The original 12 commits

Preserved here as attribution. They are not replayed individually because the
surrounding build files moved too far upstream in the intervening nine months
for a per-commit rebase to be meaningful; the end state is identical.

| commit | date | author | subject |
|---|---|---|---|
| `192959805` | 2025-10-17 | Invisi | RKNPU2 backend is implemented |
| `9cb5c4f79` | 2025-10-17 | Invisi | .gitignore is updated |
| `dbe05d30e` | 2025-10-20 | Polarnik | Implement Zero-Copy for Weights |
| `74d51a2f6` | 2025-11-02 | Invisi | Quantization support is implemented |
| `27df34fd5` | 2025-11-22 | Invisi | README.md is created |
| `5bfeb5072` | 2026-03-22 | hvalev | Zero-length segments skip (#2) |
| `14d39ea9c` | 2026-03-22 | Invisi | Hardware Pipelines & Hybrid Quantization |
| `1d3c0f582` | 2026-04-04 | Invisi | IOMMU Domain Management is implemented |
| `6362b6585` | 2026-04-11 | Gerald Tan | Fix cmake for cross-compiling (#8) |
| `d90f5fea7` | 2026-04-12 | Invisi | Caching System is fixed |
| `8df5be13c` | 2026-04-17 | Invisi | Several Environment Variables for QoL |
| `81eff6a45` | 2026-05-19 | Martino Mensio | Add includes for algorithm (#16) |

`ggml/src/ggml-rknpu2/README.md` and `CONTRIBUTING.md` in this tree are Invisi's
own documents, vendored verbatim.

## Also acknowledged

[danielferr85/rk-llama.cpp](https://github.com/danielferr85/rk-llama.cpp)
independently rebased the same backend onto llama.cpp master of 2026-07-14. That
work is not in our history and was never fetched, but commit `71fa525e` there is
what identified which backend vtable slots had changed since May, which saved us
the search. Credit where due.

## Why a fork of ggml-org/llama.cpp rather than of invisiofficial

GitHub renders a fork's diff and network relative to its parent. Forking
upstream makes this branch read as **three commits on current master**, which is
what it is. Forking `invisiofficial` would have made it read as ~1500 commits of
upstream churn against a branch last updated 2026-05-20, burying both their work
and ours.

The technical base and the credit are therefore kept separate on purpose: the
fork parent reflects what the code is built on, and authorship in `git log` and
`git blame` reflects who wrote it. The second of those is the one that survives
cloning, bisecting and blaming.
