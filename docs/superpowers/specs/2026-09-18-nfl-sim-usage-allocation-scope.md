# Sub-project B.4 — NFL sim usage allocation fix (scope / design draft)

**Status:** scope for review · **Date:** 2026-09-18
**Program:** A(done) → B(dormant) → B.2(usage model) → B.3(honest volume; DONE) → **B.4 (this: allocation fix — the real path to featured player-prop calibration)** → C(props productization).

## The problem B.3 exposed (settled, evidence-backed)

With honest per-game volume (B.3: attempts ≈ real 34, sacks excluded, rush ≈ real),
the featured player-prop markets are **under-projected**, walk-forward propable:

| market | sim mean | actual mean | bias | p50 |
|---|---|---|---|---|
| rec_yds | 35.9 | 44.0 | **−8.1** | .31 |
| rush_yds | 40.7 | 51.2 | **−10.5** | .30 |
| receptions | 3.14 | 3.95 | −0.81 | .38 |
| pass_yds (the sum) | 249 | 230 | +19 | .54 (≈calibrated) |

Featured players get too little; the roster **tail** absorbs the excess (which is
why the *sum*, pass_yds, is fine). Proven NOT to be volume, efficiency
(completion-anchoring made it worse), or dispersion (concentration re-tune did
nothing). It is an **allocation** problem. B.2's earlier "calibrated" receptions
(.456) was a volume-inflation artifact (38 vs real 34 attempts) masking the same
under-allocation.

## The open question (needs a spike before designing the fix)

Three candidate mechanisms, not yet distinguished:
1. **Flat mean shares** — the active-usage model (recent-average target/carry
   share, renormalized over the active set) simply gives featured players a
   smaller share than they truly command; the tail's shares are too high.
2. **Selection effect** — "propable" (targets≥3 / carries≥5) selects the weeks a
   player actually had a role; a player's propable-week usage exceeds their
   all-weeks recent average (which includes low-script / limited games). The sim
   projects the average; grading samples the selected higher-usage weeks.
3. **Insufficient per-player usage variance structure** — the sim's game-to-game
   share variance (Dirichlet, concentration 150) may be both too small AND the
   wrong shape, so it never reproduces a featured player's genuine high-usage
   games even though it has the right mean.

These imply different fixes, so **Phase 0 is a spike** (read-only analysis, no
model change):
- For propable player-weeks, compare the player's recent-average share (what the
  sim uses) to their ACTUAL share that week → quantifies selection (mechanism 2)
  vs persistent flatness (mechanism 1).
- Compare the distribution of a featured player's actual weekly shares to the
  sim's simulated share distribution → tests mechanism 3.
- Decompose the −8 rec_yds gap into share-gap × volume × ypr.

## Candidate approaches (chosen after the spike)

- **A. Share concentration with variance (not just mean):** replace flat mean
  shares with a per-player share *distribution* whose mean AND spread match the
  player's real weekly share history, so high-usage games appear at the right
  rate. (Addresses 1+3; avoids the mean-overshoot that killed naive sharpening —
  because it adds the right *variance*, not a higher mean.)
- **B. Role-continuation / depth-aware prior:** weight shares by depth-chart role
  + recent trend (a WR1 trending up gets projected up), reducing the lag that
  makes the recent average trail a rising role. (Addresses 2.)
- **C. Grade-honest target definition:** if the gap is mostly selection, the
  projection is fine and the fix is to only *claim* calibration on the population
  the sim targets (all starts), not the selected propable tail — a scope/claim
  correction, not a model change. (Fallback if 1/3 are small.)

The spike decides which of A/B/C (or a combination) the spec commits to.

## Gate

Same iterative discipline: walk-forward propable coverage (p50 .45–.55, p90
.85–.93) + signed bias per market, leakage-free. Target: rec_yds / rush_yds /
receptions biases → ~0 and p50 → band, **without regressing pass_yds** (B.3's
win) and without the mean-overshoot that sharpening/trim caused. Iterate to
shippable-or-diminishing-returns; a market reaching the bar earns C.

## Non-goals

- Not C. Not re-litigating B.3's volume model (kept). NFL only; branch stays
  dormant; no A/desk/+EV/CFB/MLB changes.

## Risk

This is the third calibration attempt on these markets; the selection effect
(mechanism 2) may be irreducible (you cannot project which week a player breaks
out). If the spike shows the gap is mostly selection, the honest outcome may be
"featured yardage props are not calibratable from pre-game info" — a real
finding, surfaced with evidence, and C would then target only markets that do
calibrate.
