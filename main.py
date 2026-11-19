#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model (per trophy, stdlib only, fully deterministic):
  * Work in logit space on x = ln(1 + hours since release 2026-11-19T00:00Z).
  * Fit logit(rate) = a + b*x + sum of step dummies, one step per detected
    Exophase refresh (a read where players-tracked jumps). Steps absorb the
    dilution drops when a wave of new players joins; the slope b captures
    the existing cohort actually earning trophies as it plays.
  * Slopes are shrunk toward their tier mean and the global mean.
  * Players-tracked N is projected by fitting ln N on x; the future mix of
    the progressed current cohort and the joining cohort is computed with a
    cohort model: a new player earns trophy i with probability
    q = 1 - (1 - r)^rho, rho learned from the refreshes in the history
    (aggregate trophies-earned calibration) or a launch-night prior when
    the history shows no refresh yet.
  * Spread combines fit residuals, a random-walk term growing with the
    horizon, and dilution uncertainty, so the five quantiles are honest
    about refresh risk.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Prior for a new cohort's progress relative to the current cohort, from the
# launch-night refresh documented in the kit rules (638 -> 3,868 players
# tracked, 943 -> 4,276 trophies earned): new joiners had earned almost
# nothing yet. Used only when the history shows no refresh to learn from.
RHO_PRIOR = 0.07
# Prior gap between Exophase refreshes (hours); the first came ~4.5 h after
# the first probe and the rules say refreshes come "hours apart".
REFRESH_GAP_PRIOR_H = 5.0
# Prior players-tracked jump factor of one refresh when none observed.
NJUMP_PRIOR = 4.0


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def clamp_pct(p):
    return min(99.99, max(0.01, p))


def logit_pct(p):
    p = clamp_pct(p) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 40.0:
        return 99.99
    if x <= -40.0:
        return 0.01
    return clamp_pct(100.0 / (1.0 + math.exp(-x)))


def solve_linear(a, b):
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / pv
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def wls(design, y, w):
    """Weighted least squares; returns (beta, residuals) or (None, None)."""
    k = len(design[0])
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for row, yi, wi in zip(design, y, w):
        for i in range(k):
            atb[i] += wi * row[i] * yi
            for j in range(k):
                ata[i][j] += wi * row[i] * row[j]
    # Tiny ridge for stability.
    for i in range(k):
        ata[i][i] += 1e-9
    beta = solve_linear(ata, atb)
    if beta is None:
        return None, None
    resid = [yi - sum(b * v for b, v in zip(beta, row))
             for row, yi in zip(design, y)]
    return beta, resid


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(branches, tau):
    """tau-quantile of a mixture of logit-normals.

    branches: list of (weight, median_logit, sd). Solved by bisection on
    the mixture CDF; deterministic.
    """
    lo = min(m - 8.0 * s for _, m, s in branches)
    hi = max(m + 8.0 * s for _, m, s in branches)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = sum(w * norm_cdf((mid - m) / s) for w, m, s in branches)
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def calibrate_rho(rates_before, n_before, n_after, earned_after):
    """Solve rho in q = 1-(1-r)^rho so the mixed earners match earned_after.

    rates_before are percents; earned_after is the total trophies earned in
    the read after the refresh. Returns None when unsolvable.
    """
    if n_after <= n_before or earned_after is None:
        return None
    dn = n_after - n_before
    base = sum(n_before * r / 100.0 for r in rates_before if r is not None)
    target = earned_after - base
    if target <= 0:
        return 0.0

    def total(rho):
        s = 0.0
        for r in rates_before:
            if r is None:
                continue
            p = r / 100.0
            q = 1.0 if p >= 1.0 else 1.0 - (1.0 - p) ** rho
            s += dn * q
        return s

    if total(2.0) < target:
        return 2.0
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    n_tr = len(trophies)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_h = max(0.0, (target - at).total_seconds() / 3600.0)
    horizon_d = horizon_h / 24.0
    x_target = math.log1p(hours_since_release(target))

    # --- Read series -------------------------------------------------------
    times = []          # x per read
    players = []        # N per read
    earned = []         # trophies_earned per read
    rates = []          # per-read list of percents (None where missing)
    for read in history:
        times.append(math.log1p(hours_since_release(parse_time(read["at"]))))
        players.append(read.get("players_tracked"))
        earned.append(read.get("trophies_earned"))
        rates.append(read.get("rates") or [None] * n_tr)

    # --- Detect refreshes: players-tracked jumps ---------------------------
    refresh_idx = []    # indices of reads that start a new refresh
    for i in range(1, len(players)):
        if players[i] and players[i - 1] and players[i] > 1.04 * players[i - 1]:
            refresh_idx.append(i)

    # --- Learn rho from each observed refresh ------------------------------
    rhos = []
    for i in refresh_idx:
        rho = calibrate_rho(rates[i - 1], players[i - 1], players[i],
                            earned[i])
        if rho is not None:
            rhos.append(rho)
    if rhos:
        rho_future = (median(rhos) * len(rhos) + 0.15) / (len(rhos) + 1.0)
    else:
        rho_future = RHO_PRIOR

    # --- Project players-tracked -------------------------------------------
    n_now = players[-1] if players else 0
    if len(times) >= 2 and n_now:
        ln_n = [math.log(p) for p in players if p]
        xs_n = [t for t, p in zip(times, players) if p]
        beta_n, _ = wls([[1.0, x] for x in xs_n], ln_n, [1.0] * len(xs_n))
        if beta_n is not None and beta_n[1] > 0:
            n_target = math.exp(beta_n[0] + beta_n[1] * x_target)
        else:
            n_target = float(n_now)
        # Sanity caps: no shrinkage, at most 8x the current count.
        n_target = min(max(n_target, float(n_now)), 8.0 * n_now)
    else:
        n_target = float(n_now) if n_now else 1.0

    # --- Per-trophy trend fit with refresh steps ---------------------------
    n_steps = len(refresh_idx)
    fits = []           # (a, b_shrunk, resid_sd, last_logit)
    slopes = []
    tier_slopes = {}
    for i in range(n_tr):
        xs, ys, ws, ridx = [], [], [], []
        for j, read_rates in enumerate(rates):
            r = read_rates[i]
            if r is None:
                continue
            xs.append(times[j])
            ys.append(logit_pct(r))
            ws.append(1.0)
            ridx.append(j)
        if not xs:
            fits.append(None)
            continue
        design = []
        for j, x in enumerate(xs):
            row = [1.0, x]
            # steps: 1 once this read is at/after that refresh
            row += [1.0 if ridx[j] >= refresh_idx[k] else 0.0
                    for k in range(n_steps)]
            design.append(row)
        beta, resid = wls(design, ys, ws)
        if beta is None:
            beta = [ys[-1], 0.0] + [0.0] * n_steps
            resid = [0.0] * len(ys)
        a, b = beta[0], beta[1]
        # Sum of the refresh-step coefficients: at the target every past
        # refresh has happened, so all steps are active and must be added
        # to the intercept (they are negative dilution drops; omitting
        # them would inflate the forecast).
        step_sum = sum(beta[2:])
        # Residual sd (floor keeps flat histories from claiming certainty).
        if len(ys) > len(beta):
            var = sum(r * r for r in resid) / (len(ys) - len(beta))
            sd = math.sqrt(max(var, 0.0))
        else:
            sd = 0.0
        fits.append((a, b, max(sd, 0.05), ys[-1], xs[-1], step_sum))
        slopes.append(b)
        tier = trophies[i]["tier"]
        tier_slopes.setdefault(tier, []).append(b)

    global_slope = median(slopes) if slopes else 0.0
    tier_mean = {t: median(v) for t, v in tier_slopes.items()}
    resid_sds = [f[2] for f in fits if f is not None]
    global_sd = median(resid_sds) if resid_sds else 0.05

    # Expected refreshes still to come before the target. When the history
    # shows no refresh, the chance of one grows with how long the page has
    # sat unchanged: refreshes come on a schedule "hours apart", so a long
    # silence means the next one is nearer. Hazard scale shrinks as the
    # silence eats into the prior gap.
    if refresh_idx:
        p_refresh = 1.0  # history shows refreshes happen; N model covers them
    else:
        silence_h = 0.0
        if history:
            last_rates = rates[-1]
            silence_h = (at - parse_time(history[0]["at"])).total_seconds() / 3600.0
            for j in range(len(history) - 2, -1, -1):
                if any(a is not None and b is not None and abs(a - b) > 0.005
                       for a, b in zip(rates[j], last_rates)):
                    silence_h = (at - parse_time(history[j]["at"])).total_seconds() / 3600.0
                    break
        remain = max(1.5, REFRESH_GAP_PRIOR_H - 0.5 * silence_h)
        p_refresh = min(0.9, 1.0 - math.exp(-horizon_h / remain))

    out_rows = []
    for i in range(n_tr):
        fit = fits[i]
        tier = trophies[i]["tier"]
        if fit is None:
            last = 1.0
            a, b, sd, last_logit, x_last, step_sum = logit_pct(last), 0.0, 0.05, logit_pct(last), times[-1] if times else 0.0, 0.0
        else:
            a, b_raw, sd, last_logit, x_last, step_sum = fit
            # Shrink the slope toward the tier mean, then the global mean.
            b = (b_raw * 2.0 + tier_mean.get(tier, global_slope) * 1.0) / 3.0
            b = (b * 2.0 + global_slope * 1.0) / 3.0

        # Progressed rate of the current cohort at the target. The intercept
        # was fit with all past refresh steps active, so a + b*x_target
        # plus the step coefficients carries every past dilution drop;
        # only future dilution is added on top.
        prog_logit = a + b * x_target + step_sum
        r_prog = inv_logit_pct(prog_logit)

        # Dilution by the joining cohort up to the target.
        if n_target > n_now and n_now:
            q = 1.0 - (1.0 - r_prog / 100.0) ** rho_future \
                if r_prog < 100.0 else 1.0
            r_mix = (n_now * r_prog + (n_target - n_now) * 100.0 * q) / n_target
            dil_shift = logit_pct(r_mix) - logit_pct(r_prog)
        else:
            r_mix = r_prog
            dil_shift = 0.0

        # Base spread: residual + trend-extrapolation + random walk +
        # dilution, with a floor so flat histories never claim certainty.
        var = sd * sd * (1.0 + max(0.0, x_target - x_last))
        var += (0.5 * math.sqrt(max(horizon_d, 0.0))) ** 2
        var += (0.5 * abs(dil_shift)) ** 2
        var += 0.05 * 0.05
        med_logit = logit_pct(r_mix)

        # No refresh observed yet: the honest forecast is a two-branch
        # mixture -- no refresh before the target, or one launch-style
        # refresh -- with quantiles taken from the mixture itself, so the
        # low quantiles reach into the refresh branch even though its
        # weight is below one half.
        if not refresh_idx and p_refresh > 0:
            q_prior = 1.0 - (1.0 - r_mix / 100.0) ** RHO_PRIOR \
                if r_mix < 100.0 else 1.0
            n_after = n_now * NJUMP_PRIOR if n_now else n_target
            r_after = (n_now * r_mix + (n_after - n_now) * 100.0 * q_prior) / n_after \
                if n_after > 0 else r_mix
            jump = logit_pct(r_after) - logit_pct(r_mix)
            s0 = math.sqrt(var)
            s1 = math.sqrt(var + (0.4 * abs(jump)) ** 2)
            branches = [(1.0 - p_refresh, logit_pct(r_mix), s0),
                        (p_refresh, logit_pct(r_after), s1)]
            levels = case.get("levels", [0.05, 0.25, 0.5, 0.75, 0.95])
            qs = []
            prev = None
            for tau in levels:
                v = round(inv_logit_pct(mixture_quantile(branches, tau)), 4)
                if prev is not None and v < prev:
                    v = prev
                qs.append(v)
                prev = v
            out_rows.append({"id": trophies[i]["id"], "q": qs})
            continue

        # Normal branch: quantiles around the median with spread s.
        s = math.sqrt(var)
        # Never narrower than the house random walk, never absurdly wide.
        s = min(max(s, 0.5 * math.sqrt(max(horizon_d, 1e-6))), 2.5)

        qs = []
        prev = None
        for z in Z:
            v = round(inv_logit_pct(med_logit + z * s), 4)
            if prev is not None and v < prev:
                v = prev
            qs.append(v)
            prev = v
        out_rows.append({"id": trophies[i]["id"], "q": qs})

    return {"trophies": out_rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        name = os.path.basename(path)[:-3] + ".out"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
