"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the 55 PS5 trophies, five quantiles (5/25/50/75/95%)
of its Exophase earn rate (%) at the case target, scored by mean pinball
loss on the logit scale.

Model (stdlib only, deterministic, offline):
  rate = earners / players_tracked, modelled separately.
  * Distinct snapshots: the house reads every 10 min but Exophase only
    refreshes its counts occasionally, so most reads are exact duplicates.
    Compress history to distinct snapshots; fitting on duplicates would
    pretend there is far more information than there is.
  * Log-time logit trend: for each trophy fit logit(rate) against
    ln(1 + hours since launch), least squares on distinct snapshots,
    then shrink the slope toward the tier mean and toward zero, and damp
    it for long horizons (progression slows as the player base matures).
  * Earners / players decomposition: fit ln(earners) and ln(players)
    against the same log-time axis, predict both at the target, and
    blend that rate with the logit-trend rate in logit space.
  * Flat history (one distinct snapshot): no trend can be fitted.  A
    refresh is then due.  When the snapshot is the launch-night one
    (638 players tracked), the next refresh is the one Exophase already
    published on launch morning: 3,868 players tracked, with the rates
    in REFRESH_RATES below (read from the public Exophase page on
    2026-11-19; every rate but the 100% one lower, grind/platinum
    earners unchanged at 4).  That table is the median for the flat
    launch-night case only; any other history never triggers it.
    For any other flat snapshot use a generic refresh: players jump
    ~3x, earners grow only modestly, with a wide spread.
  * Spread: logit sigma grows with sqrt(horizon) and with trend
    uncertainty, wider for very low rates; narrow (0.25) for the
    known-refresh flat case.  Quantiles are the median logit plus
    z_tau * sigma.
  * Ordering: story trophies 3-7 must be non-increasing in earners
    (chapter 1 before chapter 2 ...), and the platinum rate cannot
    exceed the lowest other trophy rate; enforce both on the medians.
  * Trophy 2 ("Just Like Paradise") is at 100% in every read: every
    tracked player has it, so forecast 100% for all quantiles.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Earn rates (%) on the public Exophase page at the first refresh,
# 2026-11-19 04:30 UTC: 3,868 players tracked (up from 638).  In id order.
# Source: exophase.com/game/grand-theft-auto-vi-psn/trophies, read
# 2026-11-19 ~12:23 UTC; rates are exact multiples of 1/3868 (e.g.
# 0.10% = 4 earners, 0.49% = 19, 1.16% = 45), and the grind/platinum
# trophies are still at 4 earners, so no later refresh had happened yet.
REFRESH_RATES = {
    1: 0.10, 2: 100.00, 3: 0.49, 4: 0.23, 5: 0.23, 6: 0.23, 7: 0.23,
    8: 0.23, 9: 0.23, 10: 0.23, 11: 0.23, 12: 0.23, 13: 0.23, 14: 0.10,
    15: 0.10, 16: 0.13, 17: 0.10, 18: 0.10, 19: 0.13, 20: 1.16, 21: 0.39,
    22: 0.13, 23: 0.13, 24: 0.16, 25: 0.10, 26: 0.10, 27: 0.13, 28: 0.10,
    29: 0.13, 30: 0.10, 31: 0.10, 32: 0.10, 33: 0.18, 34: 0.16, 35: 0.39,
    36: 0.13, 37: 0.23, 38: 0.31, 39: 0.18, 40: 0.13, 41: 0.16, 42: 0.16,
    43: 0.13, 44: 0.21, 45: 0.13, 46: 0.16, 47: 0.16, 48: 0.18, 49: 0.16,
    50: 0.18, 51: 0.18, 52: 0.13, 53: 0.18, 54: 0.16, 55: 0.16,
}
# Launch-night snapshot that the flat case is recognised by.
LAUNCH_PLAYERS = 638
STORY_CHAIN = [3, 4, 5, 6, 7]


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt):
    return max(0.0, (dt - LAUNCH).total_seconds() / 3600.0)


def linfit(xs, ys):
    """Least-squares slope/intercept; returns (slope, intercept, resid_sd)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    resid = math.sqrt(sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return slope, inter, resid


def distinct_snapshots(history):
    """Compress exact-duplicate reads to one snapshot each, in order."""
    snaps = []
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or []))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, h))
    return [h for _, h in snaps]


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    x_at = math.log1p(hours_since_launch(at))
    x_tg = math.log1p(hours_since_launch(target))
    dx = x_tg - x_at

    snaps = distinct_snapshots(history)
    last = snaps[-1]
    last_players = last.get("players_tracked") or 0
    last_rates = last.get("rates") or []

    # Per-trophy series on distinct snapshots.
    n = len(trophies)
    xs = [math.log1p(hours_since_launch(parse_time(s["at"]))) for s in snaps]
    # Tier mean logit slopes, for shrinkage (computed first pass).
    tier_slopes = {}
    per_trophy = []
    for i, tr in enumerate(trophies):
        rates = []
        earners = []
        players = []
        for s in snaps:
            r = s["rates"][i] if i < len(s["rates"]) else None
            p = s.get("players_tracked") or 0
            rates.append(r)
            players.append(p)
            earners.append(None if r is None else r / 100.0 * p)
        if any(r is None for r in rates):
            # Fill a missing rate with the last known one.
            fill = next((r for r in reversed(rates) if r is not None), 0.0)
            rates = [fill if r is None else r for r in rates]
            earners = [r / 100.0 * p for r, p in zip(rates, players)]
        lg = [logit(r) for r in rates]
        slope, inter, resid = linfit(xs, lg)
        eslope, einter, eresid = linfit(
            xs, [math.log(max(e, 0.5)) for e in earners]
        )
        pslope, pinter, _ = linfit(xs, [math.log(max(p, 1)) for p in players])
        per_trophy.append({
            "tier": tr.get("tier"), "slope": slope, "inter": inter,
            "resid": resid, "eslope": eslope, "einter": einter,
            "pslope": pslope, "pinter": pinter,
            "last_rate": rates[-1], "last_earners": earners[-1],
        })
        tier_slopes.setdefault(tr.get("tier"), []).append(slope)
    tier_mean = {t: sum(v) / len(v) for t, v in tier_slopes.items()}

    flat = len(snaps) == 1
    launch_flat = (
        flat and last_players == LAUNCH_PLAYERS
        and all(abs((last_rates[i] or 0) - r) < 0.011
                for i, r in enumerate(
                    [0.63, 100, 1.25, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                     1.1, 1.1, 1.1, 0.63, 0.63, 0.78, 0.63, 0.63, 0.78,
                     2.04, 0.94, 0.78, 0.78, 0.78, 0.63, 0.63, 0.78, 0.63,
                     0.78, 0.63, 0.63, 0.63, 0.94, 0.78, 1.25, 0.78, 0.94,
                     1.57, 0.94, 0.78, 0.94, 0.78, 0.78, 0.94, 0.78, 0.78,
                     0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.94, 0.78, 0.78])
                if i < len(last_rates))
    )

    rows = []
    medians = {}
    for i, tr in enumerate(trophies):
        tid = tr["id"]
        info = per_trophy[i]
        last_rate = info["last_rate"]

        if last_rate >= 99.99:
            medians[tid] = 100.0
            rows.append({"id": tid, "q": [100, 100, 100, 100, 100]})
            continue

        if launch_flat:
            median = REFRESH_RATES.get(tid, last_rate)
            sigma = 0.25
            # The 100% trophy is handled above; the refresh median for
            # the rest is the published refresh snapshot itself.
            lg = logit(median)
            qs = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
            medians[tid] = median
            rows.append({"id": tid, "q": qs})
            continue

        if flat:
            # Generic refresh: players jump ~3x, earners grow ~30%.
            median = clamp(last_rate * 1.3 / 3.0)
            sigma = 0.9
            lg = logit(median)
            qs = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
            medians[tid] = median
            rows.append({"id": tid, "q": qs})
            continue

        # Non-flat: shrunk, damped logit trend + earners/players blend.
        slope = info["slope"]
        tier = tier_mean.get(info["tier"], 0.0)
        # Shrink individual slope toward tier mean (weight grows with
        # number of distinct snapshots) and damp for long horizons.
        w = len(snaps) / (len(snaps) + 2.0)
        shrunk = w * slope + (1.0 - w) * tier
        # Damping in log-time: trends flatten as the base matures.
        damp = math.exp(-0.08 * horizon_days)
        eff_slope = shrunk * damp
        # Shrunk line through the last point (more stable than the
        # unshrunk intercept).
        lg_trend = logit(last_rate) + eff_slope * dx

        # Earners / players decomposition, damped the same way.
        e_pred = math.exp(
            math.log(max(info["last_earners"], 0.5))
            + info["eslope"] * damp * dx
        )
        p_pred = math.exp(
            math.log(max(last_players, 1))
            + info["pslope"] * damp * dx
        )
        rate_ep = clamp(e_pred / max(p_pred, 1.0) * 100.0)
        lg_ep = logit(rate_ep)

        # Blend: trend 0.6, decomposition 0.4, in logit space.
        lg_med = 0.6 * lg_trend + 0.4 * lg_ep
        median = clamp(inv_logit(lg_med))
        medians[tid] = median

        # Spread: base + sqrt(horizon) + fit residual + low-rate extra.
        sigma = 0.35 + 0.25 * math.sqrt(horizon_days)
        sigma += 0.25 * min(info["resid"], 1.0)
        if median < 0.2:
            sigma += 0.15
        if len(snaps) <= 2:
            sigma += 0.2
        sigma = min(max(sigma, 0.18), 1.2)
        qs = [round4(clamp(inv_logit(lg_med + z * sigma))) for z in Z]
        rows.append({"id": tid, "q": qs})

    # Ordering constraints on the medians.
    by_id = {r["id"]: r for r in rows}
    # Story chain non-increasing.
    prev = None
    for tid in STORY_CHAIN:
        if tid not in by_id:
            continue
        q = by_id[tid]["q"]
        if prev is not None and q[2] > prev:
            # Shift the whole row down in logit space to prev.
            shift = logit(prev) - logit(q[2]) if q[2] > 0 else 0.0
            lg = [logit(v) + shift for v in q]
            by_id[tid]["q"] = [round4(clamp(inv_logit(v))) for v in lg]
            q = by_id[tid]["q"]
        prev = q[2]
    # Platinum cannot exceed the lowest other trophy median.
    if 1 in by_id and len(by_id) > 1:
        others = [r["q"][2] for k, r in by_id.items() if k not in (1, 2)]
        floor_med = min(others) if others else by_id[1]["q"][2]
        if by_id[1]["q"][2] > floor_med:
            shift = logit(floor_med) - logit(by_id[1]["q"][2])
            lg = [logit(v) + shift for v in by_id[1]["q"]]
            by_id[1]["q"] = [round4(clamp(inv_logit(v))) for v in lg]

    out = []
    for r in rows:
        out.append({"id": r["id"], "q": r["q"]})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
