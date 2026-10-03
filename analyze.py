"""Steps 2-4: audit and clean the policies, then ask what predicts a claim, with the length of time a policy was in force taken into account.

    ./venv/bin/python analyze.py
        -> results/audit.csv  marginal.csv  fits.csv  coefficients.csv  scores.csv  importance.csv  calibration_deciles.csv  calibration_exposure.csv  summary.csv

One row per policy, 678,013 in all. Cleaning, as in Noll, Salzmann and Wuthrich (2018): the claim count is capped at 4 and the exposure at 1 year (a yearly policy cannot be in force for 1.01 years). The target is the number of claims; a policy that was in force
for half a year has half the chance to claim, so the Poisson model of the claim count has the log of the (capped) exposure as an offset and its coefficients are relative claim frequencies per year. The "naive" model is what the course task does:
logistic regression of "made a claim or not", with no exposure. The 52 columns are bands and categories (driver age, bonus-malus, vehicle age, vehicle power, brand, fuel, region) and the log of the density (Area, a banded density, is left out).
Folds: 5-fold cross-validation with folds made of covariate profiles (the nine risk features joined as text, crc32 of it mod 5), because 38% of the policies share every risk feature with another policy (a split contract), and such twins must not be in
both a training and a test fold. Fits are by iteratively reweighted least squares; at the solution the score equations X'(y - mu) = 0 hold (asserted). Deviance, Gini and calibration come from out-of-fold predictions only. Risks are rounded to 1e-12 before ranking (floor(x * 1e12 + 0.5)), so ties and decile edges do not depend on floating-point noise.
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
COLS = ["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
PROFILE = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
AGE = [("18-20", 18, 20), ("21-25", 21, 25), ("26-30", 26, 30), ("31-40", 31, 40), ("41-50", 41, 50), ("51-70", 51, 70), ("71+", 71, 10 ** 6)]
BM = [("50", 50, 50), ("51-59", 51, 59), ("60-79", 60, 79), ("80-99", 80, 99), ("100+", 100, 10 ** 6)]
VAGE = [("0", 0, 0), ("1-5", 1, 5), ("6-10", 6, 10), ("11-15", 11, 15), ("16+", 16, 10 ** 6)]
REF = {"driver age": "41-50", "bonus-malus": "50", "vehicle age": "1-5"}
GROUPS = ["driver age", "bonus-malus", "vehicle age", "vehicle power", "density", "vehicle brand", "fuel", "region"]
EXPOSURE_BANDS = [("up to 0.1", 0.0, 0.1), ("0.1 to 0.25", 0.1, 0.25), ("0.25 to 0.5", 0.25, 0.5), ("0.5 to 0.75", 0.5, 0.75), ("0.75 to under 1", 0.75, 0.999999), ("exactly 1", 0.999999, 1.0000001), ("over 1", 1.0000001, 10.0)]


def band(v, spec):
    return np.select([(v >= lo) & (v <= hi) for _, lo, hi in spec], [name for name, _, _ in spec], default="")


def design(df):
    """-> sparse design matrix (with intercept), the column names and the variable group of each column"""
    cols, names, grp = [np.ones(len(df))], ["intercept"], ["intercept"]
    def add(name, v, g):
        cols.append(np.asarray(v, float)); names.append(name); grp.append(g)
    for gname, spec, ref, key in (("driver age", AGE, "41-50", "DrivAge"), ("bonus-malus", BM, "50", "BonusMalus"), ("vehicle age", VAGE, "1-5", "VehAge")):
        b = band(df[key].to_numpy(), spec)
        for nm, _, _ in spec:
            if nm != ref:
                add(f"{gname}: {nm}", b == nm, gname)
    pw = np.minimum(df["VehPower"].to_numpy(), 9)
    for p in (4, 5, 7, 8, 9):
        add(f"vehicle power: {p}" + ("+" if p == 9 else ""), pw == p, "vehicle power")
    add("density: log of inhabitants per km2", np.log(df["Density"].to_numpy()), "density")
    for b_ in sorted(df["VehBrand"].unique()):
        if b_ != "B12":
            add(f"vehicle brand: {b_}", df["VehBrand"].to_numpy() == b_, "vehicle brand")
    add("fuel: diesel", df["VehGas"].to_numpy() == "Diesel", "fuel")
    for r_ in sorted(df["Region"].unique()):
        if r_ != "R24":
            add(f"region: {r_}", df["Region"].to_numpy() == r_, "region")
    return sparse.csr_matrix(np.column_stack(cols)), names, grp


def irls(X, y, offset, family, tol=1e-11, maxit=60):
    n, k = X.shape; beta = np.zeros(k)
    beta[0] = np.log(y.sum() / np.exp(offset).sum()) if family == "poisson" else np.log(y.mean() / (1 - y.mean()))
    for _ in range(maxit):
        eta = X @ beta + offset
        if family == "poisson":
            mu = np.exp(eta); w = mu; z = eta - offset + (y - mu) / mu
        else:
            mu = 1 / (1 + np.exp(-eta)); w = mu * (1 - mu); z = eta - offset + (y - mu) / w
        XtW = X.T.multiply(w).tocsr(); new = np.linalg.solve((XtW @ X).toarray(), XtW @ z)
        done = np.max(np.abs(new - beta)) < tol; beta = new
        if done:
            break
    eta = X @ beta + offset; mu = np.exp(eta) if family == "poisson" else 1 / (1 + np.exp(-eta)); grad = X.T @ (y - mu); assert np.max(np.abs(grad)) < 1e-6, ("score equations not met", np.max(np.abs(grad)))
    return beta


def deviance(y, mu):
    return 2 * float((np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu)).sum())


def risk_key(v):
    return np.floor(v * 1e12 + 0.5)                         # rounded to 1e-12, so equal risks are ties whatever the floating-point noise


def gini(pred, e, y):
    pred = risk_key(pred); order = np.lexsort((np.arange(len(pred)), pred)); p, ee, yy = pred[order], e[order], y[order]; ce, cy = np.cumsum(ee), np.cumsum(yy); last = np.r_[p[1:] != p[:-1], True]
    xs = np.r_[0.0, ce[last] / ce[-1]]; ys = np.r_[0.0, cy[last] / cy[-1]]; return 1 - 2 * float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2))


def main():
    R.mkdir(exist_ok=True)
    text = (D / "freMTPL2freq.arff").read_text().splitlines(); start = text.index("@data") + 1
    df = pd.read_csv(D / "freMTPL2freq.arff", skiprows=start, header=None, names=COLS, quotechar="'"); n = len(df); pull = json.load(open(R / "pull.json"))
    claims = np.minimum(df["ClaimNb"].to_numpy(), 4).astype(float); expo = np.minimum(df["Exposure"].to_numpy(), 1.0); anyc = (df["ClaimNb"].to_numpy() > 0).astype(float)
    prof = df[PROFILE].astype(str).agg("|".join, axis=1); vc = prof.value_counts(); fold = prof.map(lambda s: zlib.crc32(s.encode()) % 5).to_numpy()
    audit = [("policies in the file", n), ("columns", df.shape[1]), ("policies in OpenML's description", pull["openml_description_policies"]), ("duplicate policy ids", int(df["IDpol"].duplicated().sum())), ("blank values", int(df.isna().sum().sum())),
             ("policies with a claim", int(anyc.sum())), ("claims (raw)", int(df["ClaimNb"].sum())), ("claims (capped at 4)", int(claims.sum())), ("policies with more than 4 claims", int((df["ClaimNb"] > 4).sum())), ("most claims on one policy", int(df["ClaimNb"].max())),
             ("exposure in years (raw)", float(df["Exposure"].sum())), ("exposure in years (capped at 1)", float(expo.sum())), ("policies with an exposure over 1 year", int((df["Exposure"] > 1).sum())), ("longest exposure", float(df["Exposure"].max())), ("shortest exposure", float(df["Exposure"].min())),
             ("claims per exposure-year (capped)", float(claims.sum() / expo.sum())), ("policies with a vehicle age over 20", int((df["VehAge"] > 20).sum())), ("oldest vehicle", int(df["VehAge"].max())), ("policies with a driver over 90", int((df["DrivAge"] > 90).sum())), ("policies with a bonus-malus over 150", int((df["BonusMalus"] > 150).sum())),
             ("distinct covariate profiles", len(vc)), ("policies whose profile another policy shares", int(vc[vc > 1].sum())), ("largest profile", int(vc.max()))]
    for k, v in enumerate(np.bincount(fold)):
        audit.append((f"fold {k} policies", int(v)))
    pd.DataFrame(audit, columns=["check", "value"]).to_csv(R / "audit.csv", index=False)

    # marginal claim frequencies by band (raw, not adjusted)
    marg = []
    for gname, spec, key in (("driver age", AGE, "DrivAge"), ("bonus-malus", BM, "BonusMalus"), ("vehicle age", VAGE, "VehAge")):
        b = band(df[key].to_numpy(), spec)
        for nm, _, _ in spec:
            m = b == nm; marg.append({"variable": gname, "level": nm, "policies": int(m.sum()), "exposure": float(expo[m].sum()), "claims": float(claims[m].sum()), "frequency": float(claims[m].sum() / expo[m].sum())})
    pd.DataFrame(marg).to_csv(R / "marginal.csv", index=False)

    # fits: the full Poisson model, each model without one variable group, the null, and the naive logistic model; on all policies and on each training set
    X, names, grp = design(df); grp = np.array(grp); off = np.log(expo); specs = {"poisson all": ("poisson", np.ones(len(names), bool))}
    for g in GROUPS:
        specs[f"poisson without {g}"] = ("poisson", grp != g)
    specs["logistic all"] = ("logistic", np.ones(len(names), bool)); fits, oof = [], {}
    for sname, (fam, keep) in specs.items():
        Xs = X[:, np.flatnonzero(keep)]; y = claims if fam == "poisson" else anyc; o = off if fam == "poisson" else np.zeros(n); pred = np.zeros(n)
        for f in list(range(5)) + ["all"]:
            tr = np.ones(n, bool) if f == "all" else fold != f; b = irls(Xs[tr], y[tr], o[tr], fam)
            for t, c in zip(np.array(names)[keep], b):
                fits.append({"spec": sname, "fold": f, "term": t, "coefficient": c})
            if f != "all":
                te = fold == f; eta = Xs[te] @ b + o[te]; pred[te] = np.exp(eta) if fam == "poisson" else 1 / (1 + np.exp(-eta))
        oof[sname] = pred
    pd.DataFrame(fits).to_csv(R / "fits.csv", index=False)
    null = np.zeros(n)
    for f in range(5):
        tr = fold != f; null[fold == f] = claims[tr].sum() / expo[tr].sum() * expo[fold == f]
    # full-data coefficients with quasi-Poisson standard errors
    full = pd.DataFrame(fits); b = full[(full["spec"] == "poisson all") & (full["fold"] == "all")]["coefficient"].to_numpy(); mu = np.exp(X @ b + off); XtW = X.T.multiply(mu).tocsr(); cov = np.linalg.inv((XtW @ X).toarray())
    phi = float(((claims - mu) ** 2 / mu).sum() / (n - len(b))); se = np.sqrt(phi * np.diag(cov)); bl = full[(full["spec"] == "logistic all") & (full["fold"] == "all")]["coefficient"].to_numpy()
    pd.DataFrame({"term": names, "group": grp, "poisson_coefficient": b, "se": se, "relative_frequency": np.exp(b), "ci_low": np.exp(b - 1.959963984540054 * se), "ci_high": np.exp(b + 1.959963984540054 * se), "logistic_coefficient": bl, "odds_ratio": np.exp(bl)}).to_csv(R / "coefficients.csv", index=False)

    # scores out of fold
    dev = {k: deviance(claims, oof[k]) if k.startswith("poisson") else np.nan for k in oof}; dev_null = deviance(claims, null); dev_all = dev["poisson all"]
    sc = [{"model": "one frequency for everyone", "mean_deviance": dev_null / n, "gini": gini(null / expo, expo, claims), "top_decile_lift": 1.0}]
    def top_lift(rank, e, y):
        order = np.lexsort((np.arange(n), -risk_key(rank))); ce = np.cumsum(e[order]); m = (ce - e[order] / 2) / ce[-1] < 0.1; return float((y[order][m].sum() / e[order][m].sum()) / (y.sum() / e.sum()))
    sc.append({"model": "Poisson model with exposure", "mean_deviance": dev_all / n, "gini": gini(oof["poisson all"] / expo, expo, claims), "top_decile_lift": top_lift(oof["poisson all"] / expo, expo, claims)})
    sc.append({"model": "logistic model of any claim, no exposure", "mean_deviance": np.nan, "gini": gini(oof["logistic all"], expo, claims), "top_decile_lift": top_lift(oof["logistic all"], expo, claims)})
    pd.DataFrame(sc).to_csv(R / "scores.csv", index=False)
    imp = [{"variable_group": g, "columns": int((grp == g).sum()), "mean_deviance_without": dev[f"poisson without {g}"] / n, "deviance_increase": (dev[f"poisson without {g}"] - dev_all) / n, "share_of_gain_lost": (dev[f"poisson without {g}"] - dev_all) / (dev_null - dev_all)} for g in GROUPS]
    pd.DataFrame(imp).to_csv(R / "importance.csv", index=False)

    # calibration: by decile of predicted frequency (equal exposure), and by exposure band for the two models
    mu_p = oof["poisson all"]; fr = mu_p / expo; order = np.lexsort((np.arange(n), risk_key(fr))); ce = np.cumsum(expo[order]); dec = np.empty(n, int); dec[order] = np.minimum(9, ((ce - expo[order] / 2) / ce[-1] * 10).astype(int))
    pd.DataFrame([{"decile": d + 1, "policies": int((dec == d).sum()), "exposure": float(expo[dec == d].sum()), "predicted_frequency": float(mu_p[dec == d].sum() / expo[dec == d].sum()), "actual_frequency": float(claims[dec == d].sum() / expo[dec == d].sum())} for d in range(10)]).to_csv(R / "calibration_deciles.csv", index=False)
    rawe = df["Exposure"].to_numpy(); rows = []
    for nm, lo, hi in EXPOSURE_BANDS:
        m = (rawe > lo) & (rawe <= hi) if lo > 0 else (rawe <= hi)
        rows.append({"exposure_band": nm, "policies": int(m.sum()), "actual_claim_share": float(anyc[m].mean()), "poisson_claim_share": float((1 - np.exp(-mu_p[m])).mean()), "logistic_claim_share": float(oof["logistic all"][m].mean()), "actual_frequency": float(claims[m].sum() / expo[m].sum()), "poisson_frequency": float(mu_p[m].sum() / expo[m].sum())})
    pd.DataFrame(rows).to_csv(R / "calibration_exposure.csv", index=False)
    sm = {"policies": n, "claim_share": float(anyc.mean()), "frequency": float(claims.sum() / expo.sum()), "dispersion": phi, "deviance_gain_pct": 100 * (dev_null - dev_all) / dev_null, "gini_poisson": sc[1]["gini"], "gini_logistic": sc[2]["gini"], "top_decile_lift_poisson": sc[1]["top_decile_lift"]}
    pd.DataFrame(list(sm.items()), columns=["key", "value"]).to_csv(R / "summary.csv", index=False)
    for f in ("audit", "scores", "importance", "calibration_deciles", "calibration_exposure", "summary"):
        print("==", f); print(pd.read_csv(R / f"{f}.csv").round(4).to_string(index=False))
    print("== coefficients"); print(pd.read_csv(R / "coefficients.csv")[["term", "relative_frequency", "ci_low", "ci_high", "odds_ratio"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
