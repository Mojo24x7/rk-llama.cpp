# Speculative decoding on RK3588

Measured on a **Radxa ROCK 5B+** (RK3588, 4x Cortex-A76 @ 2.35 GHz, 16 GB
LPDDR5-4800, NVMe), Debian 12, rknpu driver 0.9.8. Model
**Qwen3-30B-A3B-Instruct-2507-Q4_0** (16.04 GiB, 128 experts, top-4 routing),
`--cpu-moe --no-repack`, attention on the NPU under `W8A8_STANDARD`,
`taskset -c 4-7 -t 4`, `-fa off`, `ulimit -n 1000000`.

Every table below is a **within-sweep, drift-controlled** comparison: the
baseline is re-measured **last**, and the reported drift is the A1-to-A2 delta.
Read section 6 before quoting any absolute number.

---

## 1. Why prompt-lookup, and not a draft model

Decode on this board is byte-bound. Per token the model reads roughly
1.85 GB (int8 attention on the NPU + top-4 experts on the CPU + output head),
against a measured **22.9 GB/s** CPU and **23.2 GB/s** NPU. Speculation helps
only if it reduces the number of times that traffic is paid, and hurts if the
draft itself adds traffic.

On a **sparse MoE** that asymmetry is severe. Verifying `K` drafted tokens in
one batch activates the **union** of experts across those `K` positions, so a
*rejected* draft makes the target read *more* weights than a plain single-token
step. A hit saves one token's weight read; a miss widens the batch. That single
fact explains every result in this document.

Measured consequences on this model:

| drafting method | decode | note |
|---|---|---|
| none | baseline | |
| independent draft model (Qwen3-0.6B-Q8, vocab-compatible) | **-41 %** | 0.64 GB of extra resident weights evicting the target's page cache |
| trained MTP head (1.19 GB, `n_layer_nextn = 1`) | **-7.8 %** | drafts on *every* step, so the union cost is paid every step |
| **prompt-lookup (`ngram-simple`)** | **+22 %** | drafts only on a confident context match, so the union cost is paid rarely |

The draft-model and MTP results are **not** a statement about speculation in
general — they are a statement about sparse MoE. For a **dense** model,
batch-verifying `K` tokens reads the weights once for the whole batch, which is
genuine amortisation; published results on dense Gemma-4-12B with an MTP head
reach +65 %. The model class decides the sign of the result.

> Acceptance rate is not the deciding variable. Our 0.6B draft achieved ~56 %
> acceptance and lost 41 %. What differs between methods is **the price of a
> wrong guess**, not how often they are right.

---

## 2. The speculation family, measured honestly

7 **distinct** passages, each requested exactly once (see section 6 for why
that matters), `n_predict = 96`, aggregate over 6 (first request discarded as
page-cache warming). `rag` = passage plus a question whose answer quotes it.
`chat` = short question, free-form answer.

| `--spec-type` | rag decode | chat decode | survives short-after-long | verdict |
|---|---|---|---|---|
| `none` | 7.98 | 7.94 | yes | reference |
| **`ngram-simple`** | **8.91** (+11.7 %) | **8.10** (+2.0 %) | yes | **use this** |
| `ngram-mod` | 8.92 (+11.8 %) | 7.99 (+0.6 %) | yes | ties, but keeps cross-request state |
| `ngram-cache` (no `-lcd` corpus) | 6.47 (**-18.9 %**) | 6.93 (-12.7 %) | yes | loses |
| `ngram-map-k` | not measurable | **crashes** | **NO** | do not use |
| `ngram-map-k4v` | not measurable | **crashes** | **NO** | do not use |

Drift control: baseline 7.98 measured first, 7.94 measured last = **-0.5 %**.

`ngram-simple` is the right choice for reasons beyond the number: it scans only
the **current context** and keeps no state between requests. That makes it
immune both to the crash below and to the measurement artefact in section 6,
and it means one client's prompt shape cannot affect another's request on a
shared endpoint. `ngram-mod` matches its throughput while carrying
cross-request state for no measured benefit.

`ngram-cache` loses because with no pre-populated corpus it drafts on weak
evidence, and on a sparse MoE every rejected draft is expensive. It would only
pay if fed a static corpus (`-lcs`) built offline from real traffic.

---

## 3. Bug: `ngram-map-k` / `ngram-map-k4v` abort on a short request after a long one

Both variants terminate the server process:

```
common_ngram_map_begin: refresh map: idx_last_draft=375, new begin=22, #keys_checked=3, ...
common/ngram-map.cpp:236: common_ngram_map_draft: map.idx_last_check > cur_len: 301 > 72
```

**Mechanism.** These variants maintain an n-gram map *across* requests, indexed
by absolute token position in the previous sequence. After a 302-token request
`idx_last_check` is 301. The next request is 22 tokens (`cur_len = 72` after
templating), the stale index exceeds the current sequence length, and the assert
aborts.

**Reproduction.** Start `llama-server --spec-type ngram-map-k`, send one long
request, then one short request. It aborts on the first short request. Observed
identically for `ngram-map-k4v`, at the same source line, with the same values.

**Impact.** For any multi-client or web-facing endpoint this is a remotely
reachable denial of service, since prompt length is entirely client-controlled
and no unusual input is required. It also means any benchmark of these two
variants that repeats a single prompt is measuring cross-request replay of the
harness's own earlier output rather than drafting quality.

A defensive fix would clamp or invalidate the map when `cur_len` is smaller than
the recorded index, rather than asserting.

---

## 4. `--spec-ngram-size-n` is tuned by your traffic, not by your model

`--spec-ngram-size-n` is the length of the exact token match required before a
draft is issued. Measured with `--spec-type ngram-simple`, distinct prompts,
drift **+0.2 %** (the cleanest control in this work, so these are directly
comparable):

| `--spec-ngram-size-n` | `--draft-max` | rag decode |
|---|---|---|
| 4 | 8 | 9.67 |
| **6** | **8** | **9.75** |
| 8 | 8 | 9.39 / 9.41 |
| 8 | 16 | 9.01 |
| 12 | 16 | 8.96 *(library defaults)* |
| 16 | 24 | 8.17 |

Two independent findings:

**Both parameters prefer smaller values, for different reasons.** Lowering `n`
raises the *hit rate*, because a 6-token exact repeat is far more findable than
a 16-token one when the retrieved passage appears only once. But `n = 4` is
worse than `n = 6`: below some length the match stops being specific enough, the
predicted continuation is wrong more often, and each rejected draft costs a
wider expert union. That is a genuine interior optimum. Separately, raising
`--draft-max` from 8 to 16 at fixed `n` **costs 4 %**: harvesting more tokens per
hit only pays if acceptance holds that far, and it does not, while every verify
batch grows regardless.

**There is no universal default, because the optimum is a property of the
context.** Highly redundant context — duplicated boilerplate, repeated log
lines, code reusing the same identifiers — makes long matches common and precise,
so large `n` wins. Single-occurrence retrieved context, which is what a
retrieval-augmented pipeline actually produces, makes long matches rare, so
small `n` wins. We found this the hard way: an early harness embedded its test
passage **twice** in the prompt, which guarantees a 12-token match exists and
therefore reported the library defaults as nearly twice as good as `n = 6`. On
realistic single-occurrence prompts the ordering reverses.

The chat column moves the opposite way (7.84 at `n = 6` to 8.10 at `n = 16`),
consistent with the same mechanism: on a short free-form prompt there is nothing
to match at any `n`, so a larger `n` simply drafts less often and wastes fewer
verifies. Tune against the workload you actually serve.

---

## 5. Deployed configuration

```
--spec-type ngram-simple --draft-max 8 --spec-ngram-size-n 6
```

+22 % on retrieval-style requests versus no speculation, neutral on short chat,
**lossless** (drafts are verified against the target, so output is unchanged),
and free — no draft model, no additional resident weights, no NPU tenancy. When
no match is found nothing is drafted and nothing is paid, which is why the chat
case is neutral rather than negative.

---

## 6. Measurement traps

Two independent artefact classes were hit during this work. A harness can be
free of one and still fail the other.

**Cross-request replay.** `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` and
`ngram-cache` learn across requests. A harness that sends the same prompt N
times at temperature 0 lets runs 2..N be drafted from run 1's own output. This
inflated `ngram-mod` from its true 8.92 to an apparent 10.42, and produced an
absurd 11.62 on a short free-form prompt with almost no internal redundancy.
`ngram-simple` is stateless across requests and therefore immune. **Check
whether the drafter keeps state before choosing the harness.**

**Intra-prompt redundancy.** Independently of the above, a prompt that contains
its own answer twice hands within-context lookup a guaranteed long match. This
is what produced the retracted "library defaults are twice as good" claim.
Construct prompts the way your real pipeline does.

**Benchmark absolutes are inflated relative to production.** The same deployed
configuration measured **9.75 t/s** on an exclusive-board sweep and **8.72 t/s**
on the live server for a byte-identical workload — an 11 % gap with identical
flags, thread count and CPU affinity (all three verified from
`/proc/<pid>/cmdline` and `taskset -pc`, not from the manifest). The cause is
that the sweep ran five servers over the *same* seven prompts, so by the third
configuration those specific experts had been faulted in about fourteen times
and the page cache was tuned to that exact prompt mix. On a model larger than
RAM, what matters is not *how much* of the model is resident (57 % here) but
*whether the resident part is the part this prompt routes to*. Per-prompt decode
ranged from 6.83 to 10.55 t/s across the seven prompts for this reason.

Relative, within-sweep comparisons are reliable. Cross-sweep absolutes need this
calibration applied. Always re-measure the baseline **last**: drift across the
sweeps in this document ranged from +0.2 % to -4.0 %, and in the -4.0 % case the
correction changed the ranking.

---

## 7. What is not settled

- Only `rag` and `chat` prompt shapes were measured. Code completion, where
  redundancy is high and long matches are common, would likely prefer a larger
  `n` and possibly a larger `--draft-max`; that is untested here.
- `ngram-cache` with a real static corpus (`-lcs`) built from production traffic
  was not attempted. It is the one configuration that could plausibly beat
  `ngram-simple`, since the measured loss is attributable to having no corpus.
- The dense-model case was not measured on this hardware. The physics argue an
  MTP head should win there, but the only dense hybrid we could test is
  bandwidth-bound at roughly 1.5 t/s for unrelated reasons, which makes it a
  poor vehicle.
