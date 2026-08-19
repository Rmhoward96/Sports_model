# MLB Modeling Methodology — Full Formula Spec

Scope: the complete math for (A) the player-prop matchup engine covering the 7 locked props, and (B) the MLB game-level model (expected runs, total, win probability). Every formula here is build-ready. Constants marked **[tunable]** are defaults with reasoning, not settled truth — re-fit them from your own backfill.

Companion files: data structures in [`db/schema.sql`](../db/schema.sql); approach + roadmap in [`PLAN.md`](../PLAN.md).

---

## 0. Notation & conventions

A plate appearance (PA) resolves to exactly one of **7 mutually exclusive terminal outcomes**:

| symbol | outcome |
|---|---|
| `p_bb` | walk (incl. HBP) |
| `p_k`  | strikeout |
| `p_1b` | single |
| `p_2b` | double |
| `p_3b` | triple |
| `p_hr` | home run |
| `p_out`| in-play out (non-K out) |

They sum to 1: `p_bb + p_k + p_1b + p_2b + p_3b + p_hr + p_out = 1`.

Derived: `p_hit = p_1b + p_2b + p_3b + p_hr`. Per-PA expected total bases `tb1 = p_1b + 2·p_2b + 3·p_3b + 4·p_hr`.

Rates are always stored **split by handedness** (`vs_hand`): a batter's rate uses the *pitcher's* throwing hand; a pitcher's allowed-rate uses the *batter's* hitting hand.

---

## PART A — THE PLAYER-PROP MATCHUP ENGINE

The pipeline, per (game, batter, opposing starter):

```
raw counts ──► (1) shrink to talent prior ──► (2) odds-ratio blend batter×pitcher
           ──► (3) platoon [lives inside the rates] ──► (4) pitch-mix adj [Phase 4]
           ──► (5) park/weather context ──► normalize ──► per-PA vector p_i
           ──► (6) × projected PA ──► (Part C) prop distributions ──► P(over line)
```

Steps 1–5 + normalize produce and persist `feat_matchup_pa_outcomes`; every prop is then an aggregation of it.

### 1. Rate estimation with shrinkage (regression to the mean)

Raw per-PA rates are noisy at real sample sizes, so each outcome rate is an **empirical-Bayes** estimate that regresses observed counts toward a prior:

```
p̂_i = (x_i + k_i · prior_i) / (n + k_i)
```

- `x_i` = observed count of outcome `i` in `n` PA (in the relevant handedness split + time window).
- `prior_i` = the talent prior for outcome `i` (see §4 — use the projection-system rate; fall back to league rate if no projection).
- `k_i` = **[tunable]** regression constant = PA of prior-weight to add. Default to each outcome's reliability "stabilization point" (Carleton), because at `n = k_i` the estimate is 50% observed / 50% prior:

  | outcome | `k_i` default (PA) |
  |---|---|
  | K | 60 |
  | BB | 120 |
  | HR | 170 |
  | 1B | 290 |
  | 2B / 3B | 1100 (XBH split is very noisy) |
  | in-play out | 70 |

**Windows.** Maintain `season`, `30d`, and `career` estimates in `feat_batter_profile` / `feat_pitcher_profile`; blend as `0.6·season + 0.25·30d + 0.15·career` **[tunable]** before the odds-ratio step (recency weighting).

### 2. Odds-ratio (log5) combination — batter × pitcher

Combine the batter's rate `b_i`, the pitcher's allowed-rate `p_i`, and the league rate `l_i` (Tango's odds-ratio form) per outcome. With odds `O(r) = r / (1 − r)`:

```
O_i(match) = O(b_i) · O(p_i) / O(l_i)
r_i        = O_i(match) / (1 + O_i(match))
```

The seven `r_i` won't sum to 1, so **normalize** (done once, after context in §5):

```
p_i = r_i / Σ_j r_j
```

> This per-outcome-then-normalize form is the practical standard. The more principled upgrade is a **multinomial log-linear (softmax)** model with a baseline category — worth doing in Phase 4, but the odds-ratio approximation is well-behaved for these rates.

### 3. Platoon splits (L/R)

**No separate blend step — platoon lives inside the rates.** Because §2 draws `b_i` from the batter's "vs the pitcher's hand" split and `p_i` from the pitcher's "vs the batter's hand" split, the platoon effect is already in. This is the correct way and avoids double-counting.

The one trap: handedness splits have small samples. Do **not** trust a hitter's raw vs-LHP line. Build the split as league platoon multiplier × player overall, nudged only slightly by the player's own regressed split:

```
b_i(hand) = p̂_i(overall) · M_i(hand)_league · (1 + δ_i)
δ_i       = regressed personal platoon deviation, k_platoon = 1000 PA  [tunable, heavy]
```

`M_i(hand)_league` **[tunable]** = league-average platoon multiplier for outcome `i` (e.g. RHB see a modest K↓/power↑ vs LHP). Re-estimate annually from your data.

### 4. Talent prior from projection systems  (Phase 1) + Pitch-mix/batted-ball (Phase 4)

**Talent prior (Phase 1).** In §1, set `prior_i` = the projection-system per-PA rate from `ref_player_projection` (Steamer/ZiPS/THE BAT X, or a blend). Regressing toward projected talent beats regressing toward league average because it already encodes aging, park, and Statcast. If no projection exists for a player, `prior_i = l_i` (league rate).

**Pitch-mix / batted-ball adjustment (Phase 4 — highest double-counting risk, ship OFF first).** A pitcher's marginal allowed-rates already reflect his pitch mix, so this only adds value where the *batter's* per-pitch-type strength interacts with *this* pitcher's usage beyond the marginals. Compute an expected-matchup contact-quality factor from Statcast (`fact_mlb_pitch`):

```
xwOBA_mix = Σ_t  u_p(t) · w_b(t)          # pitcher usage % × batter xwOBA, by pitch type t
adj       = clamp( xwOBA_mix / xwOBA_b_overall , 0.90, 1.10 )        # [tunable clamp]
r_i'      = r_i · (1 + α·(adj − 1))   for i ∈ {1b,2b,3b,hr};  α = 0.25   # [tunable weight]
```

Also adjust `p_k` via whiff-rate-by-pitch-type analogously. **Default `α = 0` in Phase 1** (feature disabled) — turn it on only after you can measure whether it improves calibration/CLV.

### 5. Context multipliers — park & weather

Applied to `r_i` before the normalize in §2, multiplicatively on the relevant outcomes:

```
r_hr' = r_hr · PF_hr(venue, season) · W_hr
r_i'  = r_i  · PF_hits(venue, season)        for i ∈ {1b,2b,3b}
```

- `PF_*` from `ref_mlb_park_factors` (1.00 = neutral; e.g. Coors > 1 for HR/hits).
- Weather HR multiplier **[tunable]**, dominated by wind blowing out to center:
  ```
  W_hr = 1 + β_wind · wind_out_mph + β_temp · (temp_f − 70)
  β_wind = 0.015 per mph,  β_temp = 0.006 per °F
  ```
  `wind_out_mph` = wind speed projected onto the park's home-plate→CF axis (from `ref_game_weather` + venue orientation). Dome/closed-roof ⇒ `W_hr = 1`.

Then **normalize** (§2) to get the final per-PA vector `p_i`, persisted to `feat_matchup_pa_outcomes`.

### 6. Opportunity — projected plate appearances

The biggest lever for counting props. Base expected PA by lineup slot from `feat_batting_order_pa`, scaled by team offensive context:

```
proj_pa(slot) = base_pa[season, slot] · (team_exp_PA_game / league_avg_PA_game)
```

`base_pa` league defaults (≈): slot 1 ≈ 4.6, 2 ≈ 4.5, 3 ≈ 4.4, 4 ≈ 4.3, 5 ≈ 4.2, 6 ≈ 4.1, 7 ≈ 4.0, 8 ≈ 3.9, 9 ≈ 3.8. `team_exp_PA_game` derives from projected team OBP (more baserunners ⇒ more PA).

**Pitcher-change split.** Later PAs face the bullpen, not the starter. Split each batter's PA:
```
proj_pa = pa_vs_sp + pa_vs_bp
```
`pa_vs_sp` from the starter's projected batters-faced (§C, Outs Recorded) distributed across the order; `pa_vs_bp` uses a **bullpen composite** per-PA vector (team relievers' blended rates) in place of the starter matchup vector. Phase 1 may approximate with the starter vector for all PA; Phase 3+ adds the bullpen composite.

---

## PART C — FROM THE PER-PA VECTOR TO EACH PROP DISTRIBUTION

Let a batter have `N` projected PA, each PA `j` carrying its own vector `p_i,j` (starter vector for early PA, bullpen composite for late PA — heterogeneous, so use Poisson-binomial forms, not plain Binomial). For every prop, once we have the PMF: `P(over L) = Σ_{x > L} PMF(x)`. Use half-point lines to avoid pushes; on whole-number lines report the push mass separately.

### Hitter props

**1. Hits.** Each PA is a hit w.p. `p_hit,j`. Hits `H = Σ_j Bernoulli(p_hit,j)` → **Poisson-binomial**.
```
P(H = 0) = Π_j (1 − p_hit,j)
P(over 0.5) = 1 − P(H=0)
P(over 1.5) = 1 − P(H=0) − P(H=1)
```
`P(H=1) = Σ_j p_hit,j Π_{m≠j}(1−p_hit,m)`. For speed, compute the full Poisson-binomial PMF by DFT or the O(N²) recursion; N ≤ ~5 so it's trivial.

**2. Total Bases.** Per PA, bases `B_j ∈ {0,1,2,3,4}` with probs `{p_out+p_bb+p_k, p_1b, p_2b, p_3b, p_hr}` (a walk = 0 TB). `TB = Σ_j B_j` → **convolve** the per-PA base PMFs (exact; support is 0..4N, small):
```
PMF_TB = PMF_B,1 ⊛ PMF_B,2 ⊛ … ⊛ PMF_B,N
P(over 1.5) = Σ_{tb ≥ 2} PMF_TB(tb)
```
`dist_type = 'empirical'` (store the convolved PMF or its params).

**3. Home Run (Y/N).**
```
P(≥1 HR) = 1 − Π_j (1 − p_hr,j)      # dist_type = 'bernoulli'
```

**4. Hits + Runs + RBIs (HRR).** H is from prop 1; R and RBI depend on base-out state (who's on base, who bats behind), which marginal rates can't capture.
- **Phase 1 (marginal approx):** `E[R] ≈ P(reach base)·(runner-scores rate by slot)`, `E[RBI] ≈ Σ_j p_hit,j · E[runners on | slot]`, using league slot tables **[tunable]**. Model `HRR = H + R + RBI` as **Negative Binomial** fit to `(E[HRR], Var)` (see §C-NB).
- **Phase 4 (correct):** a **Markov base-out chain** / full lineup Monte-Carlo game sim yields the *joint* distribution of H, R, RBI directly — read HRR off the empirical sim. This also fixes the correlation the marginal approx ignores.

### Pitcher props

The starter faces the opposing lineup; aggregate that lineup's matchup vectors (batter `b` vs this starter). Let the set of PA the starter throws be `𝒫` (size = projected batters faced, §Outs).

**5. Strikeouts.** `K = Σ_{PA∈𝒫} Bernoulli(p_k)`. Two variance sources: the Bernoullis **and** uncertainty in how many batters he faces (`BF`). By law of total variance:
```
E[K]   = E[BF] · k̄            where k̄ = mean p_k over the order faced
Var[K] = E[BF]·k̄(1−k̄) + k̄²·Var[BF]
```
Fit a **Negative Binomial** to `(E[K], Var[K])`:
```
r = E² / (Var − E),   p = E / Var      # NB(r, p);  requires Var > E (overdispersed — it is)
```

**6. Hits Allowed.** Same structure with `p_hit` instead of `p_k`:
```
E[HA] = E[BF]·h̄,  Var[HA] = E[BF]·h̄(1−h̄) + h̄²·Var[BF]  → NB(r,p)  (or Poisson if Var≈E)
```

**7. Outs Recorded — the workload / hook model.** Not a pure matchup stat; the manager's hook and pitch-count limit cap it.
- **Phase 1 (Normal):**
  ```
  E[outs] = min( pitch_limit / pitches_per_out ,  matchup_expected_outs )
  outs ~ Normal( E[outs], σ_outs )      σ_outs from the pitcher's historical start-length SD  [tunable]
  ```
  `pitches_per_out` and `pitch_limit` (≈ 95–105 for a healthy starter) **[tunable]** from `feat_pitcher_profile`. `matchup_expected_outs` grows when the matchup is favorable (weak lineup ⇒ efficient ⇒ deeper).
- **Phase 4 (hazard):** an inning-by-inning **survival model** — hazard of being pulled after each inning as a function of pitch count, runs allowed, times-through-order penalty, and leverage. Gives a realistic, skewed outs distribution instead of a symmetric Normal.

`E[BF]` for props 5–6 follows from `E[outs]`: `E[BF] ≈ E[outs] + baserunners_allowed ≈ E[outs]/(1 − (p_bb+p_hit)_faced)`.

### Negative-Binomial helper (§C-NB)

Given target mean `μ` and variance `v > μ`: `r = μ²/(v−μ)`, `p = μ/v`; `PMF(x) = C(x+r−1, x)·(1−p)^x·p^r`. If `v ≤ μ` (underdispersed), fall back to Binomial/Poisson.

---

## PART B — MLB GAME-LEVEL MODEL (expected runs, total, win probability)

### B.1 Expected runs per team

Aggregate the lineup's per-PA outcome vectors (vs starter for early PA, bullpen composite for late PA) into an expected team wOBA, then convert to runs (standard wRC method — annually recalibrated, **[tunable]**):

```
wOBA_team = Σ_i  w_i · E[count_i] / PA_team          # wOBA linear weights w_i (2023-era):
   w_bb=0.69, w_1b=0.89, w_2b=1.27, w_3b=1.62, w_hr=2.10   (K and outs = 0)
E[count_i] = Σ_batters proj_pa_b · p_i,b
runs_team = ( (wOBA_team − lg_wOBA)/wOBA_scale + lg_runs_per_PA ) · PA_team · PF_runs
   wOBA_scale ≈ 1.24,  lg_wOBA ≈ 0.320,  lg_runs_per_PA ≈ 0.118       # [tunable, per season]
```

`PA_team ≈ 38` league-average, scaled by team OBP. `PF_runs` = venue run park factor.

> Phase 4 upgrade: replace the linear wOBA→runs conversion with a **base-out Markov chain** or **BaseRuns** (`R = A·B/(B+C) + D`) fed by the actual lineup order — captures sequencing and lineup construction the wOBA average blurs.

### B.2 Expected total

```
E[total] = runs_home + runs_away
```

### B.3 Score distributions & win probability

Team runs are **overdispersed** vs Poisson (variance > mean, from clustering/big innings). Model each team's runs as **Negative Binomial** with mean = `runs_team` and dispersion fit from your data (`Var ≈ 1.25·μ` is a reasonable **[tunable]** seed), or run the Markov/Monte-Carlo sim directly.

Assuming (approximate) independence of the two teams' runs:
```
P(home win) = Σ_{a>b} P(home=a)·P(away=b)  +  P(tie)·P(home wins | extras)
P(over T)   = Σ_{a+b > T} P(home=a)·P(away=b)
P(home cover s) = Σ_{a−b > s} P(home=a)·P(away=b)
```
Extra-innings tie split **[tunable]**: `P(home wins | tie) ≈ 0.5 + home_edge`, `home_edge ≈ 0.04`. A Monte-Carlo game sim (Phase 4) removes the independence assumption and yields all three jointly, consistent with the props.

### B.4 Calibration & evaluation (applies to props + game model)

- Score probabilities with **log-loss / Brier**; check **calibration plots**; recalibrate with Platt/isotonic if predicted 60% ≠ ~60% realized.
- Breakeven at −110 is **52.4%** (`110/210`). Compare your probabilities to the **no-vig** market prob: `no_vig_i = imp_i / Σ imp`, where `imp = 100/(odds+100)` for + odds and `−odds/(−odds+100)` for − odds.
- **Closing Line Value** is the real validation: store `odds_line_snapshot` closing rows and measure whether your prices consistently beat the close across a large sample. Stake with **fractional Kelly** once calibrated.
- **Leakage guard:** every rate carries an `as_of_date`; freeze all features at prediction time (no post-game info, no confirmed-after-lock lineups).

---

## Consolidated tunable-parameter table

| Param | Symbol | Default | Where |
|---|---|---|---|
| Regression constants | `k_i` | K 60 / BB 120 / HR 170 / 1B 290 / XBH 1100 / out 70 PA | §A.1 |
| Window blend | — | 0.60 season / 0.25 30d / 0.15 career | §A.1 |
| Platoon multipliers | `M_i(hand)` | re-fit annually from data | §A.3 |
| Platoon personal regression | `k_platoon` | 1000 PA | §A.3 |
| Pitch-mix weight / clamp | `α` / clamp | 0 (off in P1) / [0.90,1.10] | §A.4 |
| Weather HR coeffs | `β_wind`,`β_temp` | 0.015 /mph, 0.006 /°F | §A.5 |
| Slot base PA | `base_pa` | 4.6 → 3.8 by slot | §A.6 |
| Pitch limit / pitches-per-out | — | ~100 / player-specific | §C.7 |
| wOBA weights & scale | `w_i`,`wOBA_scale` | 2023-era; recalibrate per season | §B.1 |
| League run env | `lg_wOBA`,`lg_runs_per_PA` | 0.320, 0.118 | §B.1 |
| Runs overdispersion | `Var/μ` | 1.25 | §B.3 |
| Extra-inning home edge | `home_edge` | 0.04 | §B.3 |

Everything above is a starting point. The discipline that matters: re-fit these from your own backfill, freeze features at prediction time, and judge changes by calibration + CLV — never by whether a single night looked right.
