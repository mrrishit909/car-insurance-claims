# What predicts a car insurance claim?

678,013 French motor third-party liability policies, each with its risk features, the number of claims and the part of a year it was in force. Four questions:
- how do you model a claim when policies were in force for different lengths of time;
- which risk features matter, measured on policies the model has not seen, and by how much;
- how well does the model rank drivers by risk, and are its frequencies right;
- what goes wrong with the usual first model, a logistic regression of "claim or not" that ignores how long the policy was in force.

The questions follow DataCamp's "Modeling Car Insurance Claim Outcomes" (logistic regression of whether a customer claims, and the key features), answered on a real insurer's policies instead of a course file. The data is real, not a course file.

Build log: https://mrrishit909.github.io/projects/car-insurance-claims/

## Data

OpenML dataset 41214, `freMTPL2freq` (https://www.openml.org/d/41214), which OpenML lists as CC0. It comes from the CASdatasets R package (Dutang and Charpentier, github.com/dutangc/CASdatasets, a GPL-licensed package) that goes with "Computational Actuarial Science with R" (Charpentier, ed., CRC, 2018); the cleaning conventions follow Noll, Salzmann and Wuthrich (2018), "Case study: French motor third-party liability claims". One row per policy: `ClaimNb` (claims during the exposure), `Exposure` (the part of a year in force), `Area`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Density`, `Region`. About 36 MB, so it is not committed; `results/pull.json` has the URL and checksum. OpenML's description says 677,991 policies; the file has 678,013.

```
./venv/bin/pip install pandas numpy scipy matplotlib certifi      # in a venv
./venv/bin/python download.py    # data/*
./venv/bin/python analyze.py     # results/*
./venv/bin/python check.py
./venv/bin/python charts.py
```

## Steps

1. **Download** the file, with its checksum, licence and the policy count OpenML states.
2. **Audit and clean.** Counts, blanks, duplicates; the claim count is capped at 4 and the exposure at 1 year, as in Noll et al.; the covariate profiles (the nine risk features joined as text) and how many policies share one.
3. **Model.** The claim count is Poisson with the log of the exposure as an offset, so the coefficients are relative claim frequencies per year. 52 columns: driver age, bonus-malus and vehicle age in bands, vehicle power, vehicle brand, fuel, region, and the log of the density (`Area` is a banded density and is left out). The "naive" model is a logistic regression of "any claim" on the same columns with no exposure. Fits are by iteratively reweighted least squares; the score equations hold at every solution.
4. **Score out of sample.** Five-fold cross-validation with folds made of covariate profiles (crc32 of the profile mod 5), because 38% of the policies share their risk features with another policy (a split contract) and such twins must not sit in a training and a test fold. Deviance, Gini coefficient, lift and calibration come from out-of-fold predictions; the importance of a variable is the share of the model's out-of-sample gain in deviance that is lost when it is dropped; standard errors are quasi-Poisson (scaled by the Pearson dispersion, 2.27).
5. **Check.** `check.py` rebuilds the 52 columns from the file in plain Python (the csv and math modules and loops: no pandas, numpy or scipy) and, for each of the 60 stored fits (every model on all policies and on each training set), checks that its coefficients solve the score equations X'(y - mu) = 0 on the rows it was fitted to: the likelihoods are strictly concave, so that proves they are the maximum-likelihood fit, without a solver. It then rebuilds every out-of-fold prediction, deviance, Gini coefficient, lift, calibration, importance, standard error and interval: 657 numbers. Changing a cap, a band edge, the folds, the offset, the dispersion or a ranking rule makes it fail.
6. **Charts.**

## Results

- **The policies and the cleaning.** 678,013 policies with 358,499 years of exposure, 34,060 with a claim (5.0%), 36,102 claims, 0.101 claims per year in force. No blank values or repeated policy ids. 1,224 policies have an exposure above one year (the longest is 2.01), capped at 1; 9 have more than 4 claims (up to 16), capped at 4. 257,911 policies (38%) share every risk feature with another policy: 528,765 distinct profiles for 678,013 policies. The oldest vehicle is 100 years old and 401 drivers are over 90; they are kept.
- **Exposure is not a detail.** The share of policies with a claim rises from 2.6% for policies in force up to 0.1 year to 7.0% for 0.75 to under 1 year. The logistic model, which has no exposure, predicts 5.2% for the shortest and 4.6% for the longest: it cannot see the rise. The Poisson model, which has the offset, follows it.
  But claims are not proportional to exposure in this data: policies in force up to 0.1 year had 0.457 claims per year, 6.5 times the 0.070 of policies in force exactly one year, and the Poisson model predicts 0.136 for them; it also over-predicts the policies with an exposure above 1 (0.111 against 0.044). The data does not say why.
- **How good the ranking is.** Out of fold, the Poisson model's mean deviance is 0.3132 against 0.3303 for one frequency for everyone (5.2% lower). Its Gini coefficient (policies ranked by predicted frequency, exposure on the horizontal axis, claims on the vertical) is 0.288, against 0.251 for the logistic model; the riskiest tenth of exposure claims 2.68 times the average frequency (logistic 2.48). In ten equal slices of exposure ranked by predicted risk, predicted and actual frequencies agree to a few per cent: 0.048 against 0.046 in the lowest slice, 0.266 against 0.270 in the highest.
- **What matters.** Dropping vehicle age loses 40.2% of the model's out-of-sample gain in deviance, bonus-malus 35.2%, driver age 7.3%, vehicle power 2.3%, region 1.3%, density 1.2%, vehicle brand 0.6% and fuel 0.3%.
- **The effects** (the full Poisson fit, relative claim frequency with a 95% interval, the other variables held fixed). Against bonus-malus 50: 51-59 1.34× (1.27 to 1.42), 60-79 1.89× (1.81 to 1.98), 80-99 2.14× (2.02 to 2.27), 100 and over 4.33× (4.05 to 4.64). Against a vehicle aged 1 to 5 years: a vehicle in its first year 3.24× (3.08 to 3.40), 6-10 years 1.06× (1.02 to 1.11), 11-15 0.90× (0.86 to 0.94), 16 or more 0.70× (0.65 to 0.75). Each doubling of the density goes with a 2.9% higher frequency, and diesel with 0.93× (0.90 to 0.96).
- **Driver age, raw and adjusted.** The raw claim frequency of drivers aged 18 to 20 is 2.39 times that of drivers aged 41 to 50; after the other variables are held fixed it is 1.08× (0.95 to 1.23). Ages 21 to 25: 1.55 raw, 0.76× (0.70 to 0.82) adjusted. Bonus-malus, which carries the claims history, takes most of the age signal.

## Not done, and caveats

- This is claim frequency only: claim amounts, and so the cost of a policy, are not modelled; and it is one insurer's portfolio over an unstated period, with no dates in the data.
- The Poisson model assumes claims are proportional to exposure; they are not at the short end (see above), which the offset cannot fix. Dropping or modelling short policies differently was not tried.
- The effects are associations. Bonus-malus is itself a function of past claims; the vehicle in its first year has 3.2 times the frequency, which the data cannot explain; and nothing here is a pricing recommendation (using driver age to price is regulated in many places).
- Folds made of covariate profiles stop twins from leaking, but a driver who holds several contracts with different risk features could still appear in both a training and a test fold.
- The bands were fixed in advance, not tuned; there are no interactions; `Area` is left out because it is a banded density; the brand and region levels are many, with wide intervals. Gradient-boosted and neural models, which Noll et al. show do better than a GLM on this data, were not fitted.
- The logistic model is judged on its ranking only (its probabilities are not frequencies); the standard errors assume the quasi-Poisson form (dispersion 2.27), not a negative binomial.
- OpenML's description says 677,991 policies and the file has 678,013; this project uses the file.
- OpenML lists the licence as CC0; the CASdatasets package is GPL-licensed. Nothing from the data is committed here.
