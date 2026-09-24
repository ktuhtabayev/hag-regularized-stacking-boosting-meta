hag-regularized-stacking-boosting-meta/
│
├─ README.md                                    # This file: overview + run instructions + project rules
├─ pyproject.toml                               # Dependencies + src-layout packaging config
├─ .gitignore                                   # Ignore venv, caches, outputs, large datasets, etc.
│
├─ configs/                                     # Reproducible experiment settings (NO python code here)
│  ├─ default.yaml                              # Default dataset + α, δ, κ + majorizing function + seed + output dirs
│  ├─ experiments.yaml                          # Grid/ablation configs (multiple runs)
│  └─ gui_last_used.yaml                        # GUI saves last-selected dataset/params for convenience
│
├─ datasets/                                    # Data storage (never import from here directly in algorithms)
│  ├─ raw/                                      # Original datasets (immutable)
│  │  ├─ default.csv                            # DEFAULT INPUT FILE (your extended CSV format)
│  │  ├─ default.dat                            # DEFAULT INPUT FILE (DAT: "m n c" + m rows + feature-sign row)
│  │  └─ ...                                    # Future raw files: .dat, .xlsx, .txt
│  ├─ interim/                                  # Intermediate transformed data (optional)
│  └─ processed/                                # Final cleaned/encoded datasets for training/testing
│
├─ outputs/                                     # All artifacts written here (auto-generated)
│  ├─ runs/                                     # Reproducible run folders (scripts + GUI)
│  │  ├─ quantitative_demo/                     # Quantitative stage runs (Criterion-1 → Γc → binary → η)
│  │  │  └─ <run_id>/                           # run_id = timestamp + short uuid
│  │  │     ├─ dataset_path.txt                 # Dataset path used in this run (quick human check)
│  │  │     ├─ dataset_config.json              # DatasetConfig snapshot (format/delimiter/label_mapping/etc.)
│  │  │     ├─ criterion1_table.json            # Criterion-1 results + π values + Γc + η per quantitative feature
│  │  │     ├─ quantitative_binary.csv          # Quantitative features → nominal {1,2} (keeps original qf indices)
│  │  │     └─ quantitative_contrib.csv         # Nominal {1,2} → contributions η (same indices; for reunification)
│  │  │
│  │  ├─ nominal_demo/                          # Nominal stage runs (λ, β, ω → η contributions)
│  │  │  └─ <run_id>/                           # Each nominal run folder
│  │  │     ├─ dataset_path.txt                 # Dataset path used in this run
│  │  │     ├─ dataset_config.json              # DatasetConfig snapshot
│  │  │     ├─ lambda_beta_weight_table.json    # λ, β, ω and gradations and η per nominal feature (GUI + audit)
│  │  │     └─ nominal_contrib.csv              # Nominal features replaced by η contributions (same indices)
│  │  │
│  │  ├─ hag_prep/                              # Prep stage before HAG: merge contrib + build global weight ranking
│  │  │  └─ <run_id>/                           # Each prep run folder (timestamp + short uuid)
│  │  │     ├─ dataset_path.txt                 # Dataset path used in this run
│  │  │     ├─ dataset_config.json              # DatasetConfig snapshot (reproducibility)
│  │  │     ├─ merged_contrib.csv               # Full dataset (quant+nom) replaced by contribution values (keeps feature order)
│  │  │     ├─ weights.json                     # Per-feature weights (ω) by original feature index (0-based in code)
│  │  │     └─ weight_rank.json                 # Sorted indices (desc by weight), stable tie-break (left-to-right)
│  │  │
│  │  ├─ train/                                 # HAG training stage outputs (TUPLAM, R-history, dij)
│  │  │  └─ <run_id>/                           # Each training run gets its own folder
│  │  │     ├─ dataset_path.txt                 # Dataset path used for training stage
│  │  │     ├─ dataset_config.json              # DatasetConfig snapshot (reproducibility)
│  │  │     ├─ run_config.json                  # Run config snapshot (α, δ, κ, majorizing params, seed, etc.)
│  │  │     ├─ tuplam.json                      # Final selected feature set TUPLAM (0-based original feature indices)
│  │  │     └─ dij.csv                          # Latent feature matrix (output of latent step)
│  │  │
│  │  ├─ meta_prep/                             # Preparation stage for META (build Si table: ai* + di* + Class)
│  │  │  └─ <run_id>/                           # Each META prep run folder
│  │  │     ├─ dataset_path.txt                 # Dataset path used
│  │  │     ├─ dataset_config.json              # DatasetConfig snapshot
│  │  │     ├─ run_config.json                  # HAG config snapshot used to generate TUPLAM/dij
│  │  │     ├─ tuplam.json                      # Copied/linked TUPLAM used for META (0-based)
│  │  │     ├─ dij.csv                          # Copied/linked dij latent matrix (r1..rp)
│  │  │     └─ meta_train.csv                   # META training dataset (ai0..aip, di1..dip, Class)
│  │  │
│  │  ├─ meta_new_object/                       # Build Snew=(a0..ap) for META (initial + binary formats); GUI-ready
│  │  │  └─ <run_id>/                           # Each new-object generation run folder
│  │  │     ├─ source.txt                       # Where TUPLAM came from (auto reuse latest train run / explicit run / rerun HAG)
│  │  │     ├─ config_path.txt                  # Config file path used by the demo script
│  │  │     ├─ dataset_path.txt                 # Dataset path used for this Snew
│  │  │     ├─ train_run_dir.txt                # (optional) reused HAG train run folder path
│  │  │     ├─ tuplam.txt                       # TUPLAM used (0-based)
│  │  │     ├─ snew_headers.txt                 # Headers for Snew (a0(x2), a1(x5), ...)
│  │  │     ├─ snew_init.csv                    # Snew initial-format values (nominal original, quantitative raw in dataset format)
│  │  │     ├─ snew_binary.csv                  # Snew binary-format values (quantitative -> {1,2}; nominal unchanged)
│  │  │     └─ snew.json                        # GUI-friendly JSON snapshot of Snew, headers, gamma_map, source, dataset path
│  │  │
│  │  ├─ margin_analysis/                       # Margin Analysis runs (Excel experiments ported to Python)
│  │  │  └─ <run_id>/                           # Each margin analysis run folder (timestamp + short uuid)
│  │  │     ├─ latent_r1_object_margins.csv     # Per-object margins for r1: Sid,d,Class,ObjectMargin,yhat
│  │  │     ├─ latent_r2_object_margins.csv     # Per-object margins for r2: Sid,d,Class,ObjectMargin,yhat
│  │  │     ├─ latent_r3_object_margins.csv     # Per-object margins for r3: Sid,d,Class,ObjectMargin,yhat
│  │  │     ├─ latent_r4_object_margins.csv     # Per-object margins for r4: Sid,d,Class,ObjectMargin,yhat
│  │  │     ├─ margin_report.csv                # Per-latent-feature report: boundaries, midpoint, width, argmax/argmin indices
│  │  │     └─ margin_report.json               # Same report as JSON (GUI-friendly)
│  │  │
│  │  ├─ predict/                               # META prediction stage outputs (B1/B2 filtering + decision)
│  │  │  └─ <run_id>/                           # Prediction run folder
│  │  │     ├─ artifacts_ref.json               # References to train/meta_prep/meta_new_object artifacts used in prediction
│  │  │     ├─ prediction.json                  # Final predicted label + a_new binary vector
│  │  │     └─ meta_debug.json                  # GUI-friendly full B1/B2 history per j + decision summary
│  │  │
│  │  ├─ evaluate/                              # (future) Evaluation stage outputs (metrics + reports + plots)
│  │  │  └─ <run_id>/                           # Evaluation run folder
│  │  │     ├─ metrics.json                     # accuracy/F1/AUC/etc.
│  │  │     ├─ confusion_matrix.csv             # Confusion matrix raw values
│  │  │     └─ report.md                        # Human-readable summary for thesis/paper (optional)
│  │  │
│  │  └─ gui/                                   # (future) GUI-driven runs (single place for the app to browse)
│  │     └─ <run_id>/                           # GUI session run folder
│  │        ├─ gui_state.json                   # Last UI state: selected dataset/params/tab/etc.
│  │        ├─ logs.txt                         # GUI session logs
│  │        └─ links.json                       # Links to train/predict/eval runs produced by GUI actions (optional)
│  │
│  ├─ figures/                                  # Saved plots for papers/thesis (PNG/PDF)
│  └─ logs/                                     # Global logs (optional; per-run logs can also exist)
│
├─ resources/                                   # Non-code assets
│  ├─ article/                                  # PDF/article copies
│  ├─ experiments/                              # Excel notes + experiment evidence (Heart-Disease [10, 13, 2], etc.)
│  ├─ screenshots/                              # Equation screenshots, UI screenshots
│  └─ notes/                                    # Notes, derivations, TODOs, references
│
├─ src/                                         # Installable Python package root (src-layout; clean imports)
│  └─ hag_regularized_stacking_boosting_meta/
│     ├─ __init__.py                            # Package exports (keep minimal)
│     │
│     ├─ domain/                                # Data contracts (dataclasses), types, parameters, errors
│     │  ├─ __init__.py                         # Expose key dataclasses (Params, Artifacts, etc.)
│     │  ├─ params.py                           # RunConfig + DatasetConfig + HAGParams + MajorizingConfig (single source of truth)
│     │  ├─ schema.py                           # Dataset schema: feature types, label mapping, selected cols
│     │  ├─ artifacts.py                        # HAGArtifacts(TUPLAM, dij, ...), MetaArtifacts, RunResult
│     │  └─ errors.py                           # Custom exceptions (ValidationError, MissingArtifactsError, etc.)
│     │
│     ├─ algorithms/                            # Pure algorithm code (NO IO, NO GUI, minimal logging)
│     │  ├─ __init__.py                         # algorithms package marker (stable imports)
│     │  │
│     │  ├─ hag/                                # Algorithm 1: Greedy HAG + Regularization + Latent Features
│     │  │  ├─ __init__.py                      # Public API exports for HAG (build_tuplam, etc.)
│     │  │  ├─ greedy_grouping.py               # Step 1–5 loop: build TUPLAM using θ/γ + stopping (κ, δ)
│     │  │  ├─ prep.py                          # Merge quant+nom contribution datasets + compute global ω ranking
│     │  │  ├─ weights.py                       # Stable facade routing to weights_quant / weights_nominal
│     │  │  ├─ weights_quant.py                 # Quantitative: Criterion-1, Γc, binary dataset, η contributions
│     │  │  ├─ weights_nominal.py               # Nominal: λ, β, ω, gradations, η contributions
│     │  │  ├─ majorizing.py                    # Majorizing function factory (identity/sigmoid/logistic/...)
│     │  │  ├─ stats.py                         # M1/M2, θ, γ, θ/γ; safe ratio handling
│     │  │  ├─ update.py                        # R(St) update, b_t creation, margin adjustment ±α f(-·)
│     │  │  ├─ selection.py                     # Candidate scan in Step 3: choose best q using θ/γ
│     │  │  └─ latent.py                        # Build dij/additional features; produce latent feature matrix
│     │  │
│     │  └─ meta/                               # Algorithm 2: META classifier (filter + decision)
│     │     ├─ __init__.py                      # Public API: MetaClassifier, MetaPredictResult, debug exports
│     │     ├─ prep.py                          # Preparation for META (build Si = ai* + di* + Class)
│     │     ├─ new_object.py                    # Build Snew=(a0..ap) (initial + binary) for GUI/scripts
│     │     ├─ filtering.py                     # META Step 1–3: construct/filter B1/B2 by matches + sign rules
│     │     ├─ decision.py                      # META Step 4: compare scores; output class (K1/K2/0)
│     │     └─ classifier.py                    # OOP wrapper: fit(A,D,y), predict(a_new), predict_batch
│     │
│     ├─ evaluation/                            # Evaluation utilities (Margin now; later confusion/metrics/reports/plots)
│     │  ├─ __init__.py                         # Export evaluation API (margin/metrics/confusion/report/plot helpers)
│     │  ├─ margin.py                           # Margin analysis on dij: boundaries, midpoint, width, object margins, yhat
│     │  ├─ confusion.py                        # Confusion matrix helpers (planned; can start minimal)
│     │  ├─ metrics.py                          # Accuracy/Precision/Recall/F1/AUC helpers (planned; can start minimal)
│     │  ├─ reports.py                          # JSON/Markdown report builders (planned; for GUI + paper)
│     │  └─ plots.py                            # Matplotlib evaluation plots (planned; margin/CM/perf curves)
│     │
│     ├─ services/                              # Orchestration layer (HAG → META → evaluation), GUI entry point
│     │  ├─ __init__.py                         # Service exports (runner/trainer/predictor/evaluator)
│     │  ├─ trainer.py                          # Runs HAG training; returns artifacts (TUPLAM, dij, history)
│     │  ├─ predictor.py                        # Runs META prediction; returns predictions + debug info
│     │  ├─ evaluator.py                        # Metrics + summaries for GUI and reports (future extension)
│     │  └─ runner.py                           # EndToEndRunner: train→predict→evaluate (GUI-friendly)
│     │
│     ├─ io/                                    # Data & artifact IO (isolated from algorithms)
│     │  ├─ __init__.py                         # IO package marker
│     │  ├─ loaders.py                          # Dataset loaders (csv_extended + csv_simple + dat_matrix)
│     │  ├─ writers.py                          # Artifact writers (json/csv) to outputs/runs/...
│     │  ├─ serialization.py                    # JSON/YAML helpers; versioned formats
│     │  ├─ configs.py                          # YAML → RunConfig/DatasetConfig/HAGParams loader
│     │  │
│     │  └─ formats/                            # Pluggable dataset loaders (future split/extension)
│     │     ├─ __init__.py                      # Registers built-in loaders
│     │     ├─ base.py                          # Loader protocol + registry
│     │     ├─ csv_simple.py                    # Basic CSV format loader
│     │     ├─ csv_extended.py                  # Extended CSV loader: header + feature-type row
│     │     ├─ dat_matrix.py                    # DAT loader placeholder (if later moved out of loaders.py)
│     │     ├─ excel_xlsx.py                    # Future: .xlsx loader
│     │     └─ txt_space.py                     # Future: space-separated .txt loader
│     │
│     ├─ viz/                                   # Visualization utilities (matplotlib only; no GUI widgets)
│     │  ├─ __init__.py                         # Visualization package marker
│     │  └─ plots.py                            # θ/γ vs iteration, R distributions, survival plots, confusion matrix
│     │
│     └─ utils/                                 # General helpers (safe to import anywhere)
│        ├─ __init__.py                         # Utils package marker
│        ├─ logging.py                          # Logger setup, formatting, per-run routing
│        └─ paths.py                            # Path helpers: project root, run-id, safe joins
│
├─ app/                                         # PyQt GUI application (thin layer)
│  ├─ main.py                                   # GUI entry point: QApplication + MainWindow
│  ├─ ui/                                       # UI definitions (.ui or widgets)
│  ├─ controllers/                              # UI ↔ services.runner; manages pipeline state
│  ├─ workers/                                  # QThread/QRunnable: long tasks without freezing UI
│  ├─ widgets/                                  # Custom reusable widgets (tables, plots, parameter panels)
│  └─ assets/                                   # Icons, themes (QSS), images
│
├─ scripts/                                     # CLI helpers (batch experiments + demos)
│  ├─ _quick_load_test.py                       # Loader sanity check (prints shapes + indices)
│  ├─ convert_dat2csv.py                        # Utility: convert .dat → .csv (dataset preparation / locale cleanup)
│  ├─ hag_prep_demo.py                          # Runs HAG prep stage → outputs/runs/hag_prep/<run_id>/
│  ├─ load_from_yaml_test.py                    # Tiny sanity test: load YAML and print resolved params
│  ├─ margin_analysis_demo.py                   # Runs margin analysis on dij.csv → outputs/runs/margin_analysis/<run_id>/
│  ├─ meta_new_object_demo.py                   # Creates new object Snew → outputs/runs/meta_new_object/<run_id>/
│  ├─ meta_predict_demo.py                      # Runs META prediction + saves debug JSON → outputs/runs/predict/<run_id>/
│  ├─ meta_prep_demo.py                         # Builds meta_train.csv → outputs/runs/meta_prep/<run_id>/
│  ├─ nominal_demo.py                           # Runs nominal stage (prints message if no nominal features)
│  ├─ quantitative_demo.py                      # Runs quantitative stage (prints message if no quantitative features)
│  ├─ run_experiments.py                        # Batch runner for experiments.yaml (grid-ready)
│  └─ train_hag_demo.py                         # Runs HAG training → outputs/runs/train/<run_id>/
│
└─ tests/                                       # Unit + integration tests (pytest)
   ├─ test_weights_eta.py                       # Validate ω/η computations on small known examples
   ├─ test_theta_gamma.py                       # Validate θ, γ, θ/γ math and edge cases (gamma=0)
   ├─ test_hag_stop_rule.py                     # Validate stopping rule: (|TUPLAM|<κ and crit>δ)
   ├─ test_meta_step4_decision.py               # Known case: 1/4 vs 2/6 → K2; also tie → 0
   └─ test_end_to_end_small.py                  # Tiny dataset end-to-end: HAG→latent→META stable output

## Setup (Windows, Python 3.12.10)

```powershell
cd D:\PhD\CODING\hag-regularized-stacking-boosting-meta
& "C:\Program Files\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"

.venv\Scripts\python.exe -m pytest          # tests
.venv\Scripts\python.exe -m ruff check .    # lint
.venv\Scripts\python.exe scripts\train_hag_demo.py   # run scripts from the project root (paths in configs/ are relative)
```
