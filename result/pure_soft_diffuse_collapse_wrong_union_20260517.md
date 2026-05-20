# Pure-soft Diffuse Collapse on VStar Wrong-Union

## Setup

- Dataset: `data/vstar_exp1_wrong_union.jsonl`
- Samples: `102`
- Construction: union of wrong samples from exp1 VStar `cot`, `lead`, and `pure_soft`
- Base run dir: `output/experiments/20260517_181331/pure_soft_collapse_wrong_union_parallel`
- Decoding:
  - `method=pure_soft`
  - `cot_prompt_mode=orign`
  - `do_sample=False`
  - `temperature=0.6`
  - `top_p=0.95`
  - `top_k=20`
  - `max_new_tokens=1024`

## Intervention

Baseline pure-soft always uses:

```text
next_input_embedding = sum_i p_i * E_i
```

Collapse version keeps pure-soft by default, but when the current token is a diffuse high-entropy spike, it uses one-step discrete collapse:

```text
next_input_embedding = E[selected_token]
```

Trigger:

```text
H_t >= 1.0
H_t > local_mean(previous 16 tokens) + 2.0 * local_std(previous 16 tokens)
top1_prob < 0.20 or top1_prob - top2_prob < 0.05
```

## Results

| method | accuracy | mean length | median length | p90 length | max length | long >= 256 | maxed 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_soft baseline | 23/102 = 22.55% | 360.98 | 125 | 1024 | 1024 | 29 | 18 |
| diffuse collapse | 41/102 = 40.20% | 200.14 | 110 | 284 | 1024 | 14 | 6 |

Output changes:

- Changed outputs: `81/102`
- Fixed: `23`
- Damaged: `5`
- Net gain: `+18`

Collapse trigger stats:

- Total collapse tokens: `264`
- Samples with at least one collapse: `84/102`
- Mean collapses per sample: `2.59`
- Median: `2`
- p90: `6`
- Max: `10`

Top collapse-count samples:

```text
[(92, 10), (18, 8), (179, 8), (4, 7), (19, 6),
 (77, 6), (129, 6), (145, 6), (152, 6), (156, 6),
 (184, 6), (187, 6)]
```

## Interpretation

This is a strong positive signal for the router hypothesis.

The intervention does not add visual information. It only prevents soft embedding at diffuse low-confidence spikes. Accuracy improves substantially on the wrong-union subset, and output degeneration is reduced:

- p90 length drops from `1024` to `284`
- maxed-out samples drop from `18` to `6`
- long outputs drop from `29` to `14`

This supports the claim that a major pure-soft failure mode is not missing visual grounding, but representational noise caused by mixing many incompatible low-confidence candidate tokens.

Next steps:

1. Run the same collapse intervention on full VStar to measure damage rate on originally correct samples.
2. Try threshold variants:
   - stricter: `top1 < 0.15`, `margin < 0.03`
   - looser: `top1 < 0.25`, `margin < 0.08`
3. Add semantic guards so answer/format tokens are handled separately from diffuse reasoning tokens.
