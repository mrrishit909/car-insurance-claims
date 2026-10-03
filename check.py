"""Step 5: the project's check. Recompute with different code, in plain Python straight from the downloaded file (the csv and math modules and loops: no pandas, no numpy, no scipy): every count in the audit; the capped claims and exposure;
the 52 columns of the design, rebuilt by hand; and for every one of the 60 fitted models in fits.csv (each model on all policies and on each of the five training sets) a check that its coefficients solve the score equations X'(y - mu) = 0 on the
rows it was fitted to. The log-likelihoods of the Poisson model with an offset and of the logistic model are strictly concave, so a coefficient vector that solves them is the maximum-likelihood fit: the check needs no solver, only the
claim that the stored numbers are the optimum. Then, from the out-of-fold predictions it rebuilds: the deviances, the Gini coefficients, the lift, the calibration by decile and by exposure band, the drop-one importance, and the coefficients'
standard errors (the Fisher information, accumulated row by row and inverted by Gauss-Jordan elimination).

    ./venv/bin/python check.py
"""
import csv
import math
import zlib
from pathlib import Path

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
AGE = [("18-20", 18, 20), ("21-25", 21, 25), ("26-30", 26, 30), ("31-40", 31, 40), ("41-50", 41, 50), ("51-70", 51, 70), ("71+", 71, 10 ** 6)]
BM = [("50", 50, 50), ("51-59", 51, 59), ("60-79", 60, 79), ("80-99", 80, 99), ("100+", 100, 10 ** 6)]
VAGE = [("0", 0, 0), ("1-5", 1, 5), ("6-10", 6, 10), ("11-15", 11, 15), ("16+", 16, 10 ** 6)]
GROUPS = ["driver age", "bonus-malus", "vehicle age", "vehicle power", "density", "vehicle brand", "fuel", "region"]
EXPOSURE_BANDS = [("up to 0.1", 0.0, 0.1), ("0.1 to 0.25", 0.1, 0.25), ("0.25 to 0.5", 0.25, 0.5), ("0.5 to 0.75", 0.5, 0.75), ("0.75 to under 1", 0.75, 0.999999), ("exactly 1", 0.999999, 1.0000001), ("over 1", 1.0000001, 10.0)]
n_ok = 0


def same(label, got, want, rel=1e-9):
    global n_ok
    n_ok += 1
    want = float(want) if want not in ("", "nan") else float("nan")
    if math.isnan(want):
        assert math.isnan(got), f"{label}: recount {got} != results nan"
    else:
        assert abs(got - want) <= rel * max(1.0, abs(want)), f"{label}: recount {got} != results {want}"


def eq(label, got, want):
    global n_ok
    n_ok += 1
    assert got == want, f"{label}: recount {got!r} != results {want!r}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv")))


def risk_key(v):
    return math.floor(v * 1e12 + 0.5)


def band(v, spec):
    for name, lo, hi in spec:
        if lo <= v <= hi:
            return name
    return ""


def invert(A):
    m = len(A); M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for c in range(m):
        p = max(range(c, m), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]; piv = M[c][c]; M[c] = [v / piv for v in M[c]]
        for r in range(m):
            if r != c and M[r][c]:
                f = M[r][c]; Mr, Mc = M[r], M[c]
                for j in range(2 * m):
                    Mr[j] -= f * Mc[j]
    return [row[m:] for row in M]


def main():
    lines = open(D / "freMTPL2freq.arff").read().splitlines(); start = lines.index("@data") + 1
    rows = list(csv.reader(lines[start:], quotechar="'")); N = len(rows); pull = __import__("json").load(open(R / "pull.json"))
    aud = {r["check"]: float(r["value"]) for r in table("audit")}
    def A(name, got):
        same(name, got, aud[name])
    pol, cn, ex = [int(float(r[0])) for r in rows], [int(r[1]) for r in rows], [float(r[2]) for r in rows]; area, vp, va, da, bm = [r[3] for r in rows], [int(r[4]) for r in rows], [int(r[5]) for r in rows], [int(r[6]) for r in rows], [int(r[7]) for r in rows]
    brand, gas, dens, reg = [r[8] for r in rows], [r[9] for r in rows], [int(r[10]) for r in rows], [r[11] for r in rows]
    Y = [float(min(c, 4)) for c in cn]; E = [min(x, 1.0) for x in ex]; ANY = [1.0 if c > 0 else 0.0 for c in cn]
    prof = ["|".join((area[i], str(vp[i]), str(va[i]), str(da[i]), str(bm[i]), brand[i], gas[i], str(dens[i]), reg[i])) for i in range(N)]; FOLD = [zlib.crc32(p.encode()) % 5 for p in prof]
    cnt = {}
    for p in prof:
        cnt[p] = cnt.get(p, 0) + 1
    A("policies in the file", N); A("columns", len(rows[0])); A("policies in OpenML's description", pull["openml_description_policies"]); A("duplicate policy ids", N - len(set(pol))); A("blank values", sum("" in r for r in rows))
    A("policies with a claim", sum(c > 0 for c in cn)); A("claims (raw)", sum(cn)); A("claims (capped at 4)", sum(Y)); A("policies with more than 4 claims", sum(c > 4 for c in cn)); A("most claims on one policy", max(cn))
    A("exposure in years (raw)", math.fsum(ex)); A("exposure in years (capped at 1)", math.fsum(E)); A("policies with an exposure over 1 year", sum(x > 1 for x in ex)); A("longest exposure", max(ex)); A("shortest exposure", min(ex))
    A("claims per exposure-year (capped)", sum(Y) / math.fsum(E)); A("policies with a vehicle age over 20", sum(v > 20 for v in va)); A("oldest vehicle", max(va)); A("policies with a driver over 90", sum(v > 90 for v in da)); A("policies with a bonus-malus over 150", sum(v > 150 for v in bm))
    A("distinct covariate profiles", len(cnt)); A("policies whose profile another policy shares", sum(v for v in cnt.values() if v > 1)); A("largest profile", max(cnt.values()))
    for k in range(5):
        A(f"fold {k} policies", sum(f == k for f in FOLD))
    for r in table("marginal"):
        spec, vals = {"driver age": (AGE, da), "bonus-malus": (BM, bm), "vehicle age": (VAGE, va)}[r["variable"]]; idx = [i for i in range(N) if band(vals[i], spec) == r["level"]]
        same(f"marginal {r['variable']} {r['level']} policies", len(idx), r["policies"]); same("exposure", math.fsum(E[i] for i in idx), r["exposure"]); same("claims", sum(Y[i] for i in idx), r["claims"]); same("frequency", sum(Y[i] for i in idx) / math.fsum(E[i] for i in idx), r["frequency"])
    # the design, by hand
    names = ["intercept"] + [f"driver age: {nm}" for nm, _, _ in AGE if nm != "41-50"] + [f"bonus-malus: {nm}" for nm, _, _ in BM if nm != "50"] + [f"vehicle age: {nm}" for nm, _, _ in VAGE if nm != "1-5"] + ["vehicle power: 4", "vehicle power: 5", "vehicle power: 7", "vehicle power: 8", "vehicle power: 9+"] \
        + ["density: log of inhabitants per km2"] + [f"vehicle brand: {b}" for b in sorted(set(brand)) if b != "B12"] + ["fuel: diesel"] + [f"region: {r}" for r in sorted(set(reg)) if r != "R24"]
    T = len(names); Z = T; ti = {t: i for i, t in enumerate(names)}; DI = ti["density: log of inhabitants per km2"]; assert T == 53
    group_of = [t.split(":")[0] for t in names]
    ROWS = []
    for i in range(N):
        def slot(g, level):
            return ti.get(f"{g}: {level}", Z)
        pwr = min(vp[i], 9); ROWS.append((slot("driver age", band(da[i], AGE)), slot("bonus-malus", band(bm[i], BM)), slot("vehicle age", band(va[i], VAGE)), slot("vehicle power", str(pwr) + ("+" if pwr == 9 else "")), slot("vehicle brand", brand[i]),
                                         Z if gas[i] != "Diesel" else ti["fuel: diesel"], slot("region", reg[i]), math.log(dens[i]), Y[i], ANY[i], math.log(E[i]), FOLD[i]))
    fitrows = table("fits"); coefs = {}
    for r in fitrows:
        coefs.setdefault((r["spec"], r["fold"]), {})[r["term"]] = float(r["coefficient"])
    specs = ["poisson all"] + [f"poisson without {g}" for g in GROUPS] + ["logistic all"]; preds = {s: [0.0] * N for s in specs}; maxgrad = 0.0; n_fits = 0
    for s in specs:
        fam = "logistic" if s.startswith("logistic") else "poisson"
        for k in (0, 1, 2, 3, 4, "all"):
            cf = coefs[(s, str(k))]; b = [0.0] * (T + 1)
            for t, v in cf.items():
                b[ti[t]] = v
            present = [ti[t] for t in cf]; g = [0.0] * (T + 1); b0, bd = b[0], b[DI]; pred = preds[s]; exp = math.exp
            for i, (i0, i1, i2, i3, i4, i5, i6, ld, y, a, off, f) in enumerate(ROWS):
                eta = b0 + b[i0] + b[i1] + b[i2] + b[i3] + b[i4] + b[i5] + b[i6] + bd * ld
                if fam == "poisson":
                    mu = exp(eta + off)
                else:
                    mu = 1.0 / (1.0 + exp(-eta))
                if k != "all" and f == k:
                    pred[i] = mu; continue
                r_ = (y if fam == "poisson" else a) - mu; g[0] += r_; g[i0] += r_; g[i1] += r_; g[i2] += r_; g[i3] += r_; g[i4] += r_; g[i5] += r_; g[i6] += r_; g[DI] += r_ * ld
            m = max(abs(g[j]) for j in present); maxgrad = max(maxgrad, m); assert m < 5e-5, (s, k, "score equations not met", m)
            n_fits += 1
    # scores
    def dev(mu):
        return 2 * math.fsum((y * math.log(y / m) if y > 0 else 0.0) - (y - m) for y, m in zip(Y, mu))
    null = [0.0] * N
    for f in range(5):
        tr = [i for i in range(N) if FOLD[i] != f]; fr_ = math.fsum(Y[i] for i in tr) / math.fsum(E[i] for i in tr)
        for i in range(N):
            if FOLD[i] == f:
                null[i] = fr_ * E[i]
    dev_null, dev_all = dev(null), dev(preds["poisson all"])
    def gini(pred):
        K = [risk_key(v) for v in pred]; order = sorted(range(N), key=lambda i: (K[i], i)); ce = cy = 0.0; xs, ys = [0.0], [0.0]; te, ty = math.fsum(E), sum(Y)
        for j, i in enumerate(order):
            ce += E[i]; cy += Y[i]
            if j == N - 1 or K[order[j + 1]] != K[i]:
                xs.append(ce / te); ys.append(cy / ty)
        return 1 - 2 * math.fsum((xs[a] - xs[a - 1]) * (ys[a] + ys[a - 1]) / 2 for a in range(1, len(xs)))
    def lift(pred):
        K = [risk_key(v) for v in pred]; order = sorted(range(N), key=lambda i: (-K[i], i)); te = math.fsum(E); ce = 0.0; se = sy = 0.0
        for i in order:
            ce += E[i]
            if (ce - E[i] / 2) / te < 0.1:
                se += E[i]; sy += Y[i]
        return (sy / se) / (sum(Y) / te)
    fp = [m / e for m, e in zip(preds["poisson all"], E)]; sc = {r["model"]: r for r in table("scores")}
    r0, r1, r2 = sc["one frequency for everyone"], sc["Poisson model with exposure"], sc["logistic model of any claim, no exposure"]
    same("null mean deviance", dev_null / N, r0["mean_deviance"]); same("null gini", gini([m / e for m, e in zip(null, E)]), r0["gini"], 1e-9); same("null lift", 1.0, r0["top_decile_lift"])
    same("poisson mean deviance", dev_all / N, r1["mean_deviance"]); same("poisson gini", gini(fp), r1["gini"], 1e-9); same("poisson lift", lift(fp), r1["top_decile_lift"], 1e-9)
    same("logistic gini", gini(preds["logistic all"]), r2["gini"], 1e-9); same("logistic lift", lift(preds["logistic all"]), r2["top_decile_lift"], 1e-9)
    for r in table("importance"):
        g = r["variable_group"]; dg = dev(preds[f"poisson without {g}"]); same(f"{g} columns", sum(x == g for x in group_of), r["columns"]); same(f"{g} deviance without", dg / N, r["mean_deviance_without"]); same(f"{g} increase", (dg - dev_all) / N, r["deviance_increase"], 1e-8)
        same(f"{g} share of gain lost", (dg - dev_all) / (dev_null - dev_all), r["share_of_gain_lost"], 1e-8)
    mu_p = preds["poisson all"]; Kd = [risk_key(v) for v in fp]; order = sorted(range(N), key=lambda i: (Kd[i], i)); te = math.fsum(E); ce = 0.0; dec = [0] * N
    for i in order:
        dec[i] = min(9, int((ce + E[i] / 2) / te * 10)); ce += E[i]
    for r in table("calibration_deciles"):
        d = int(r["decile"]) - 1; idx = [i for i in range(N) if dec[i] == d]; se = math.fsum(E[i] for i in idx)
        same(f"decile {d + 1} policies", len(idx), r["policies"]); same("decile exposure", se, r["exposure"]); same("decile predicted", math.fsum(mu_p[i] for i in idx) / se, r["predicted_frequency"]); same("decile actual", sum(Y[i] for i in idx) / se, r["actual_frequency"])
    for r, (nm, lo, hi) in zip(table("calibration_exposure"), EXPOSURE_BANDS):
        eq("band name", r["exposure_band"], nm); idx = [i for i in range(N) if (lo < ex[i] <= hi if lo > 0 else ex[i] <= hi)]; m = len(idx)
        same(f"{nm} policies", m, r["policies"]); same("actual claim share", sum(ANY[i] for i in idx) / m, r["actual_claim_share"]); same("poisson claim share", math.fsum(1 - math.exp(-mu_p[i]) for i in idx) / m, r["poisson_claim_share"])
        same("logistic claim share", math.fsum(preds["logistic all"][i] for i in idx) / m, r["logistic_claim_share"]); same("actual frequency", sum(Y[i] for i in idx) / math.fsum(E[i] for i in idx), r["actual_frequency"]); same("poisson frequency", math.fsum(mu_p[i] for i in idx) / math.fsum(E[i] for i in idx), r["poisson_frequency"])
    # standard errors of the full Poisson fit: the Fisher information, row by row
    cf = coefs[("poisson all", "all")]; b = [0.0] * (T + 1)
    for t, v in cf.items():
        b[ti[t]] = v
    Info = [[0.0] * T for _ in range(T)]; chi = 0.0
    for (i0, i1, i2, i3, i4, i5, i6, ld, y, a, off, f) in ROWS:
        eta = b[0] + b[i0] + b[i1] + b[i2] + b[i3] + b[i4] + b[i5] + b[i6] + b[DI] * ld; mu = math.exp(eta + off); chi += (y - mu) ** 2 / mu
        nz = [0] + [j for j in (i0, i1, i2, i3, i4, i5, i6) if j != Z] + [DI]; vals = {0: 1.0, DI: ld}
        for p in nz:
            vp_ = vals.get(p, 1.0); row = Info[p]
            for q in nz:
                row[q] += mu * vp_ * vals.get(q, 1.0)
    phi = chi / (N - T); inv = invert(Info); z = 1.959963984540054; lg = coefs[("logistic all", "all")]
    for r in table("coefficients"):
        j = ti[r["term"]]; se_ = math.sqrt(phi * inv[j][j]); bj = b[j]
        eq("term group", r["group"], "intercept" if j == 0 else group_of[j]); same(f"{r['term']} coefficient", bj, r["poisson_coefficient"], 1e-9); same("se", se_, r["se"], 1e-7); same("relative frequency", math.exp(bj), r["relative_frequency"], 1e-9)
        same("ci low", math.exp(bj - z * se_), r["ci_low"], 1e-7); same("ci high", math.exp(bj + z * se_), r["ci_high"], 1e-7); same("logistic coefficient", lg[r["term"]], r["logistic_coefficient"], 1e-9); same("odds ratio", math.exp(lg[r["term"]]), r["odds_ratio"], 1e-9)
    sm = {r["key"]: float(r["value"]) for r in table("summary")}
    same("summary policies", N, sm["policies"]); same("summary claim share", sum(ANY) / N, sm["claim_share"]); same("summary frequency", sum(Y) / math.fsum(E), sm["frequency"]); same("summary dispersion", phi, sm["dispersion"], 1e-9)
    same("summary deviance gain", 100 * (dev_null - dev_all) / dev_null, sm["deviance_gain_pct"], 1e-9); same("summary gini poisson", gini(fp), sm["gini_poisson"], 1e-9); same("summary gini logistic", gini(preds["logistic all"]), sm["gini_logistic"], 1e-9); same("summary lift", lift(fp), sm["top_decile_lift_poisson"], 1e-9)
    print(f"check.py: {n_ok} recomputed numbers agree with results/, and all {n_fits} fits solve their score equations (largest {maxgrad:.1e}); every prediction and statistic rebuilt in plain Python")


if __name__ == "__main__":
    main()
