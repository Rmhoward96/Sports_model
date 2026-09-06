# CFB Forward-Looking Priors — Design Spec

**Status:** Draft for review
**Date:** 2026-09-05
**Sport:** CFB first (method later adapts to NFL where data allows)

## Problem

The CFB prediction model is 100% backward-looking. Elo carries the prior
season's final rating forward with a decay (`carryover`), SRS and the points
ratings are computed only from games already played. Early in the season those
inputs are last year's team. The model therefore cannot see offseason change —
transfer portal, recruiting, coaching hires, returning/departing production —
and mis-rates teams that reloaded or regressed. The LSU–Clemson opener (both
"improved," LSU clearly the better *current* team, model blind to it) is the
canonical failure. The error is concentrated in roughly Weeks 1–5; after that
the ratings are mostly built on this year's games and self-correct.

## Goal

Give the model a **forward-looking preseason prior** so Week-0 ratings reflect
the *current* roster/staff, then let real games take over as they accumulate.
Keep the model **independent of the betting market** (no shrinking toward the
line) so its disagreements with the line are its own — and **validate** whether
those disagreements are a real edge, not just noise.

## Non-goals

- **No market blending.** The Vegas line stays a *comparison* (the existing
  lean display + grading), never a model input. (This is the deliberate
  "independent + edge-finding" choice.)
- **NFL** — out of scope for v1. No CFBD-equivalent data source; revisit after
  the CFB method is proven.
- **In-game / weekly injury** modeling — later increment.
- Shipping any factor as an "edge" before it clears the validation gate below.

## Data sources (all via the CFBD API — key already provisioned as
`CFBD_API_KEY`)

Pulled per team-season into a committed snapshot, mirroring `build_cfb_lines.py`:

| Factor | CFBD endpoint(s) | Use |
|---|---|---|
| Base preseason rating | `/ratings/sp` (SP+), fallback `/ratings/fpi` | Prior base. **SP+ preseason already includes recruiting + returning production + regressed prior performance**, so it is the base, not double-counted. |
| Returning production | `/player/returning` | Diagnostic / fallback if SP+ preseason unavailable for a team-season. |
| Recruiting class | `/recruiting/teams` | Already in SP+ base; kept for the fallback prior and as an explanatory field. |
| Transfer portal | `/player/portal` | **Explicit adjustment:** net portal talent = Σ(incoming player ratings) − Σ(outgoing). Data horizon ~2021+. |
| Coaching change | `/coaches` | **Explicit adjustment:** detect HC change by comparing consecutive seasons; flag first-year HC (and optionally coordinator change if derivable). |
| Talent composite | `/teams/talent` | Optional supporting feature for the portal/coaching adjustments. |

## Model design

### Preseason prior

For each team-season:

```
R_pre = base_rating(SP+ preseason, in Elo-scale units)
        + w_portal   * net_portal_talent_z
        + w_coach    * coaching_change_flag
        + w_recruit  * recruiting_z        # only in the SP+-absent fallback prior
```

- `base_rating` is SP+ (or FPI) preseason, mapped onto the model's Elo scale via
  a fitted linear transform so it is comparable to the existing rating units.
- `net_portal_talent_z`, `recruiting_z` are z-scored across FBS for the season.
- `coaching_change_flag` ∈ {0,1} (first-year HC); a scheme-change refinement is
  a later increment.
- All `w_*` are **fitted by the backtest**, not hand-set. A weight the backtest
  can't justify (no accuracy/edge improvement) is dropped.

### Integration into the existing engine (decaying prior)

The prior replaces the naive Elo season carryover as the Week-0 seed, and acts
as a decaying regularization target for SRS/points while few games have been
played (reusing the existing `games_played` tracking and `shrink` machinery):

```
effective_rating(team, week) =
    blend( R_pre, in_season_rating, weight = f(games_played) )
```

`f` starts ~1.0 on the prior at Week 0 and decays toward the in-season rating as
games accumulate (fitted decay, same shape family as the current shrink curve).
By mid-season the prior's weight is ~0 and the model behaves as today. This keeps
the change **surgical and market-independent** — it only changes how the season
*starts*, not the in-season math.

## Validation gate (non-negotiable — nothing ships without passing)

Walk-forward backtest across available seasons (portal-inclusive window ~2021+;
longer for SP+/recruiting-only variants), extending
`backtest_cfb_ratings.py` / `backtest_cfb_gameline.py`:

1. **Accuracy:** does the prior-seeded model reduce margin MAE and total MAE
   versus the current model, *especially Weeks 1–5*? Report per-week deltas.
2. **Edge (the real bar):** bucket games by |model_margin − closing_line|. Do the
   model's biggest disagreements beat the closing line at a profitable rate
   (ATS > ~52.4%, and/or positive CLV)? A signal that improves accuracy but does
   **not** beat the line is reported as "better predictor, no edge" — and we do
   not label or sell it as an edge.
3. **Ablation:** fit `w_*` with each factor in/out to confirm portal and coaching
   each carry independent, non-spurious signal before inclusion.

## Increments

1. **CFBD priors ingest** → `scripts/build_cfb_priors.py` writing
   `assets/cfb/priors.parquet` (team-season: base rating, returning production,
   recruiting, net portal, coaching-change flag). Idempotent, retried, like
   `build_cfb_lines`. Manual/seasonal workflow.
2. **Prior assembly + Elo-scale mapping** (pure, unit-tested): components →
   `R_pre` per team-season.
3. **Decaying-prior integration** into the ratings/gameline engine (pure,
   unit-tested), behind the existing config JSONs.
4. **Backtest + edge validation** — run the gate above, fit `w_*` and the decay,
   report accuracy + edge tables. **Decision point:** ship only what passes.
5. Later: coordinator/scheme change, injuries, momentum/recency weighting — each
   through the same gate.

## Risks / open questions

- **Portal data horizon.** CFBD portal data is thin before ~2021, limiting the
  portal-inclusive backtest to a few seasons — small sample for the portal
  weight. Mitigation: validate SP+-base + coaching first (longer history), add
  portal weight on the shorter window and treat its weight conservatively.
- **Double-counting.** SP+ preseason already embeds recruiting + returning
  production; the design uses SP+ as base and only layers portal/coaching to
  avoid it. If we ever build the from-scratch composite (SP+-absent fallback),
  recruiting re-enters there only.
- **Coaching signal is noisy.** First-year-HC as a flat flag is crude; its
  weight may not survive ablation. That's an acceptable, reportable outcome.
- **The honest likely result:** priors should measurably improve early-season
  *accuracy*. Whether they yield a *beatable edge* vs the line is genuinely
  open — the backtest decides, and the result is reported straight either way.
