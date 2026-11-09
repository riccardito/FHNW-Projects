# MachineLearning

**Courses:** Supervised Learning (`sul`, autumn 2020) · Unsupervised Learning (`ule`, spring 2021) · a first-semester challenge (January 2020) · **Author:** Riccard Nef

## What this is

Three Minichallenge notebooks and one tiny Python package. The course rule was to write the algorithms by hand — numpy, pandas, matplotlib, seaborn — and reach for scikit-learn only where the task said so. So PCA, K-Means and ridge regression are implemented from the maths here, not imported.

## Contents

| Path | What it does | Result (as written) |
|---|---|---|
| `ule_minichallenge_1.ipynb` | An own PCA class via SVD (`components_`, `variance_`), a derivation that the components are the covariance eigenvectors, a 95 %-variance cut, then NMF on a sample with a search over k and alpha | none stated; the data is about 700 000 solar spectra × 240 features |
| `ule_minichallenge_2.ipynb` | An own K-Means class (scipy `cdist`), elbow against silhouette to pick k, a PCA-2 view, a Gaussian mixture on 5 000 spectra with the EM steps written out in German | "best k = 4"; GMM with four components |
| `sul_minichallenge_1.ipynb` | Ridge regression as an sklearn-style class (gradient descent and normal equation), OLS assumptions, squared features, Lasso, kernel ridge | unfinished — several answer cells are empty |
| `linearRegression/` (`regX`) | My first packaged module: one function, `calc(df)`, fits a closed-form line of mean insurance charges per age. `setup.py`, MIT licence, its own README | `testing.py` is a demo script, not a test suite |

## How to run

Not runnable as is — none of the data files are committed. The `ule` notebooks read `iris_sun_spectra.npy` from `../../../data/`; the `sul` notebook expects `data/schalentiere_*.csv`; `regX` expects an `insurance.xlsx` with `age` and `charges` columns. Libraries: numpy, pandas, scikit-learn, scipy, seaborn.

## Data

- **IRIS solar spectra** (NASA's IRIS mission), provided by the course for the `ule` challenges. Licence not stated, not included.
- **Schalentiere** (shellfish) training and test CSVs, course-provided. Not included.
- **insurance.xlsx** (medical cost per age), a first-semester dataset. Not included.

## What it shows, and what it doesn't

Implementing PCA, K-Means and ridge from the equations; model selection by elbow, silhouette and R²; sampling to keep compute in bounds (the full NMF took two minutes, so it runs on a sample). The `sul` notebook was never finished. I had no solar-physics background, and the notebooks say so.

## Status

`ule` 1 and 2 finished; `sul` 1 a draft; `regX` a 2020 artefact, kept for the record.
