# HAG Regularized Stacking Boosting META

Local Python implementation of a two-algorithm classifier for two-class datasets
with mixed quantitative and nominal features:

- **Algorithm 1 — HAG**: feature weights (Criterion-1 for quantitative features,
  λ·β for nominal ones), contribution values η, and greedy grouping with a
  regularizer `±α·f(−·)` built on a majorizing function `f`. It selects the
  ordered feature set TUPLAM and produces latent features r1..rp.
- **Algorithm 2 — META**: classifies a new object `Snew` by filtering the
  training objects through the TUPLAM features and the signs of the latent
  features, then comparing class scores.
- **Margin analysis** of the latent features, showing how the regularizer widens
  the margin between the classes step by step.

The project has two ways to run:

- stage scripts: one command per algorithm stage, writing reproducible run folders
- PyQt6 desktop GUI: a local research dashboard that runs every stage and shows all tables and plots

No web frontend, database, or server is used.

## Setup

Clone the repository:

```powershell
git clone https://github.com/ktuhtabayev/hag-regularized-stacking-boosting-meta.git
cd hag-regularized-stacking-boosting-meta
```

Create the project virtual environment (Python 3.10+; developed on 3.12.10) and
install the package from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The code resolves project paths relative to the repository root, but a virtual
environment stores absolute paths. If the project folder is moved or renamed,
delete `.venv` and create it again.

Check the environment:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import numpy, pandas, yaml, PyQt6, hag_regularized_stacking_boosting_meta; print('OK')"
```

## Run

Launch the GUI:

```powershell
.\.venv\Scripts\python.exe scripts\launch_gui.py
```

Run all tests and the linter:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Run the stage scripts from the repository root (paths in `configs/` are relative
to it), in pipeline order:

```powershell
.\.venv\Scripts\python.exe scripts\run_quantitative_weights.py   # Criterion-1, Γc, binary {1,2}, η
.\.venv\Scripts\python.exe scripts\run_nominal_weights.py        # λ, β, ω, η
.\.venv\Scripts\python.exe scripts\run_hag_prep.py               # merged contributions + weight ranking
.\.venv\Scripts\python.exe scripts\run_hag_training.py           # HAG: TUPLAM + latent features dij
.\.venv\Scripts\python.exe scripts\run_meta_prep.py              # META training set (ai*, di*, Class)
.\.venv\Scripts\python.exe scripts\run_meta_new_object.py        # random new object Snew
.\.venv\Scripts\python.exe scripts\run_meta_prediction.py        # META classification of Snew
.\.venv\Scripts\python.exe scripts\run_margin_analysis.py        # margins of the latent features
```

`run_meta_prediction.py` creates any missing upstream run automatically; add
`--reuse-new-object` to classify the latest saved `Snew` instead of a new one.
`run_hag_training.py` prints the Excel-style Step-3 columns (B, D, F, H, I–O) for
every candidate, which is useful for checking small datasets against Excel.

`scripts\convert_dat2csv.py` converts a `.dat` dataset into the equivalent CSV
(set `input_path` / `output_path` inside the script).

## Configuration

Everything is configured in:

```text
configs/default.yaml
```

- `dataset`: active dataset path (`format: auto` picks the loader from the file extension)
- `hag`: α (`alpha`), δ (`delta`), κ (`kappa`), `cr1`, class labels,
  and the majorizing function `f` (`majorizing.name` + `majorizing.params`)
- `seed`: seed of the random new object `Snew`
- `dataset_catalog`: dataset presets shown in the GUI dropdown

The organizer u is not a parameter: HAG always starts from the feature with the highest
weight ω of the chosen dataset. On the default dataset that is index 2 (x3 in Excel),
which reproduces the Excel experiment. The GUI shows the organizer found by the last Run
as a read-only 0-based index (e.g. `x2`).

Supported majorizing functions:

| Name | f(x) | Parameters |
| --- | --- | --- |
| `identity` | x | — |
| `sigmoid`, `logistic` | 1 / (1 + e^(−k(x − x0))) | `k`, `x0` |
| `exponential` | scale · e^(k(x − x0)) | `k`, `x0`, `scale` |
| `quadratic` | a·x² + b·x + c | `a`, `b`, `c` |
| `piecewise_linear` | linear interpolation through (x[i], y[i]) | `x`, `y` |

## Datasets

All datasets use the same extended layout, as CSV or whitespace-separated DAT:

```text
m, n, c                 first row: objects, features, classes
x1 ... xn, Class        m data rows (class labels 1 and 2)
t1 ... tn               last row: feature types (1 = quantitative, 0 = nominal)
```

| Dataset | Objects | Features | Quantitative | Nominal | Files |
| --- | ---: | ---: | ---: | ---: | --- |
| Default (= Heart-Disease 10) | 10 | 13 | 6 | 7 | csv, dat |
| Heart-Disease | 10 | 13 | 6 | 7 | csv, dat |
| Heart-Disease | 270 | 13 | 6 | 7 | csv, dat |
| Cancer | 589 | 44 | 44 | 0 | csv, dat |
| Cancer-N | 589 | 44 | 0 | 44 | dat |
| DDos | 10000 | 80 | 79 | 1 | csv, dat |
| DDos | 10000 | 22 | 22 | 0 | dat |
| DDos-N | 10000 | 80 | 0 | 80 | dat |
| Molecular-Biology | 100 | 10 | 0 | 10 | dat |
| Molecular-Biology | 100 | 50 | 0 | 50 | dat |

The startup dataset is `datasets/raw/default.csv`.

## GUI

The PyQt6 app is a local research dashboard. It provides:

- dataset preset dropdown (from `dataset_catalog`) plus a free path picker
- HAG parameters: α, δ, κ, organizer (a feature index or *Auto*), majorizing function and its parameters
- Run action that executes in a background thread with a busy indicator and stage messages
- Export and Open Output Folder actions
- window geometry persists between sessions; the dataset and parameters reset to `configs/default.yaml`
- grouped result tabs:
  - Dataset: objects and a feature summary (type, ω, rank, Γc, TUPLAM position)
  - Quantitative Weights: Criterion-1 (π1, π2, π3, ω, Γc, η), binary {1,2} table, contributions
  - Nominal Weights: λ, β, ω, gradation counts with η, contributions
  - HAG Prep: merged contribution table, weight ranking
  - HAG Training: iterations with the stop rule, Step 3 θ/γ scan of every candidate,
    latent features dij, criterion plot
  - META Training Set: `ai0..aip`, `di1..dip`, `Class`
  - META Classification: editable `Snew` (initial values are binarized by Γc),
    random `Snew` by seed, B1/B2 filtering history, Step 4 decision
  - Margin Analysis: margin report, per-object margins, and an animated margin plot
    that steps through r1..rp

Objects are labeled `S0, S1, ...` and features `x0, x1, ...` (0-based, as in the
code), so the ids in the B1/B2 sets match the table rows.

## Outputs

Every output is written under `outputs/runs/` (ignored by git).

A GUI run only computes and shows results; nothing is written until you press Export,
which writes `outputs/runs/gui/<run_id>/` with the currently shown `Snew` (pressing it
again rewrites the same folder). Saving a plot puts the PNG in that folder's `figures/`:

```text
run_info.json            parameters, TUPLAM, θ/γ history, Snew and its class
dataset_path.txt
dataset_config.json
dataset/                 dataset.csv, features.csv
quantitative/            criterion1.csv, binary.csv, contributions.csv
nominal/                 weights.csv, gradations.csv, contributions.csv
hag/                     merged_contributions.csv, weight_rank.csv, iterations.csv,
                         step3_scan.csv, dij.csv
meta/                    meta_train.csv, snew.csv, filtering.csv, decision.csv
margin/                  margin_report.csv, latent_r<j>_object_margins.csv
figures/                 PNGs saved from the plot tabs
```

Each stage script writes `outputs/runs/<stage>/<run_id>/` with
`<run_id> = YYYYMMDD_HHMMSS_<8 hex>`; the stage folders are `quantitative_weights`,
`nominal_weights`, `hag_prep`, `train`, `meta_prep`, `meta_new_object`, `predict`,
and `margin_analysis`.

## Algorithm

Let `K1` and `K2` be the two classes, `m` objects and `n` features. All indices
in code and outputs are 0-based.

### Feature weights and contributions

Quantitative feature `c` (Criterion-1): objects are sorted by `x_c`, and every
split into a left and a right part is scored

```text
score = [Σ_parts Σ_classes (u² − u)] / [(|K1|² − |K1|) + (|K2|² − |K2|)]
      · [Σ_parts (u1·(|K2| − u2) + u2·(|K1| − u1))] / (2·|K1|·|K2|)
```

where `u1`, `u2` count class objects in a part. The best score is the weight
`ω_c`, its split value is `π2` (`π1`, `π3` are the minimum and maximum), and the
threshold is

```text
Γc = (π2 + b) / 2,   b = nearest value above π2
binary value = 1 if x ≤ Γc, else 2
```

Nominal feature `c` with gradation counts `g1_j` (in K1) and `g2_j` (in K2):

```text
λ_c = 1 − Σ_j g1_j·g2_j / (2·|K1|·|K2|)
β_c = Σ_j [g1_j(g1_j − 1) + g2_j(g2_j − 1)] / (D1 + D2)
D_d = (|K_d| − l_d + 1)(|K_d| − l_d)  if μ > 2,  else |K_d|(|K_d| − 1)
ω_c = λ_c · β_c
```

Contribution of gradation `j` (both feature kinds):

```text
η_c(j) = ω_c · (g1_j / |K1| − g2_j / |K2|)
```

### HAG (Algorithm 1)

Each object gets the sign `s = +1` in K1 and `s = −1` in K2. Starting from the
organizer `u` (TUPLAM = {u}, `R` = its contribution column):

1. Step 3: for every remaining candidate `q`
   `F = R + η_q`, `H = F + s·α·f(−F)`, and θ/γ is computed from running class
   means of `H` (Excel columns I–O). The candidate with the smallest θ/γ
   (below `cr1`) is selected.
2. Step 4: `R ← H + s·α·f(−H)` becomes the next organizer and the latent
   feature `r_t`.
3. Continue while `|TUPLAM| < κ` and `θ/γ > δ` (and candidates remain).

### META (Algorithm 2)

Training rows are `S_i = (ai0..aip, di1..dip, Class)`: `ai_j` is the value of the
j-th TUPLAM feature (quantitative values binarized by Γc) and `di_j = r_j`.
For a new object `Snew = (a0..ap)`:

```text
Step 1:    B1(a0) = {S_i ∈ K1 | ai0 = a0},  B2(a0) = {S_i ∈ K2 | ai0 = a0}
Step 2:    B1(a_j) = {S_i ∈ B1(a_{j−1}) | ai_j = a_j, di_j > 0}
           B2(a_j) = {S_i ∈ B2(a_{j−1}) | ai_j = a_j, di_j < 0},   j = 1..p
Step 4:    score1 = |B1(a_p)| / |K1|,  score2 = |B2(a_p)| / |K2|
           Snew ∈ K1 if score1 > score2, K2 if score1 < score2, else 0
```

### Margin analysis

For each latent feature `d = r_j`:

```text
left boundary  L = max over K2 of d
right boundary R = min over K1 of d
midpoint = (L + R) / 2,   width = R − L   (negative = the classes overlap)
object margin = ±(d − midpoint)  (+ for K1, − for K2),   ŷ = K1 if d > midpoint
```

## Project Layout

```text
configs/        default.yaml (dataset, HAG parameters, dataset catalog)
datasets/raw/   datasets (CSV and DAT)
outputs/        generated run outputs (ignored by git)
resources/      article/ (PDF, DOCX) and experiments/{hag-algorithm, meta-algorithm,
                model-evaluation}/ (Excel experiment + cheatsheet/ notes and screenshots)
scripts/        GUI launcher, stage scripts, dat -> csv converter
src/            Python package
tests/          pytest suite
```

Core package modules (`src/hag_regularized_stacking_boosting_meta/`):

```text
algorithms/hag/     weights_quantitative, weights_nominal, weights (facade),
                    input_preparation, majorizing_functions, regularization,
                    theta_gamma, candidate_selection, greedy_grouping, latent_features
algorithms/meta/    training_set, new_object, filtering, decision, classifier
evaluation/         margin_analysis
domain/             params (HAGParams, MajorizingConfig)
io/                 configs, loaders (csv/dat), writers
services/           runner (end-to-end pipeline), report_builder (tables)
gui/                app (PyQt6 window), plots, theme
utils/              run_manager (run ids and folders)
```
