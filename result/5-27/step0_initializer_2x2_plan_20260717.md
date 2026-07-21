# Step-0 Initializer 2x2 Plan

## Goal

Decompose the original LEAD step-0 early path: determine whether its benefit comes from the soft state, the weak newline cue, or their interaction. The `0.9 soft + 0.1 newline` mixture is inherited from the original LEAD implementation; this experiment separates it from the subsequent step-1 bridge.

## Fixed protocol

- R1-Onevision-7B-RL; VStar full and MMVP full.
- Greedy decoding, seed 42, 1024 new tokens, origin prompt.
- Step 1 is forced to return to the direct hard embedding (`beta=1`); all later steps use normal COT.
- The matched `soft + newline` cell is reused from the completed direct-hard run.

## Matrix

| Step-0 route | Newline | Run | Purpose |
|---|---|---|---|
| hard | off | `hard_no_newline` | wrapper/COT-equivalence control |
| hard | on | `hard_with_newline` | newline-only effect |
| soft | off | `soft_no_newline` | latent-initialization-only effect |
| soft | on | `soft_with_newline` | proposed full initializer; reused matched run |

At step 0, soft is the raw-vocabulary expectation `s0=sum_v p0(v)e(v)`, while hard is the selected-token embedding `e(y0)`. The newline-on rows use `0.9 * route + 0.1 * e(newline)`.

## Decision rules

- Full cell exceeds both single-factor cells: support a soft/newline interaction.
- Soft/no-newline exceeds hard/no-newline: support latent initialization without relying on format cue.
- Hard/newline explains the gain: call it boundary/format steering, not latent reasoning.
- Hard/no-newline differs from matched COT: audit wrapper equivalence before interpretation.

Report accuracy, MMVP pair accuracy, failed extraction, output length, fixed/damaged, paired McNemar, bootstrap CI, and the interaction `A_soft,newline - A_soft,no-newline - A_hard,newline + A_hard,no-newline`.
