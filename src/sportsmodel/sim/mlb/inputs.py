"""Assemble a kernel GameSpec from the same inputs the analytic props path uses."""
from __future__ import annotations

from sportsmodel.model import game, rates
from .kernel import Batter, GameSpec, Pitcher


def _matchup(batter_vec, pitcher_vec, league, *, pf, hr_mult, opp_def):
    v = rates.matchup_vector(batter_vec, pitcher_vec, league)
    if opp_def != 1.0:
        v = game.apply_bip_defense(v, opp_def)
    if hr_mult != 1.0:
        v = game.apply_hr_multiplier(v, hr_mult)
    if pf != 1.0:
        v = game.apply_park_to_vector(v, pf)
    return v


def build_game_spec(home_order, away_order, home_sp_vec, away_sp_vec,
                    home_bp_vec, away_bp_vec, workload, context, league, adv,
                    home_starter_id, away_starter_id,
                    roe_p=0.0, wp_p=0.0, dispersion=None) -> GameSpec:
    # workload: {player_id: (avg_outs, sd_outs)} -- feeds Pitcher's outs-recorded hook.
    pf, hr_mult = context["home_pf"], context["hr_mult"]
    home_def, away_def = context["home_def"], context["away_def"]

    def order(pairs, opp_sp_vec, opp_bp_vec, opp_def):
        out = []
        for pid, bvec in pairs:
            vs_sp = _matchup(bvec, opp_sp_vec, league, pf=pf, hr_mult=hr_mult, opp_def=opp_def)
            vs_bp = _matchup(bvec, opp_bp_vec, league, pf=pf, hr_mult=hr_mult, opp_def=opp_def)
            out.append(Batter(pid, vs_sp, vs_bp))
        return out

    # missing bullpen vector -> fall back to the starter vector (same guard
    # backtest_game.py's predict() applies: `vec_bp = ... if opp_bp else vec_sp`)
    home = order(home_order, away_sp_vec, away_bp_vec or away_sp_vec, away_def)  # home bats vs away pitchers
    away = order(away_order, home_sp_vec, home_bp_vec or home_sp_vec, home_def)
    spec = GameSpec(home, away,
                    Pitcher(home_starter_id, *workload[home_starter_id]),
                    Pitcher(away_starter_id, *workload[away_starter_id]))
    spec.adv = adv  # consumed by kernel.simulate
    spec.roe_p = roe_p
    spec.wp_p = wp_p
    spec.dispersion = dispersion
    return spec
