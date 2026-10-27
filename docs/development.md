# Running and developing the project

Run commands from the project root. The server Python environment is `runtime/app-env`; `scripts/deployment/project_python.py` loads this project's configuration.

| Component | Entry point |
|---|---|
| Backend API | `src/pbpk_kb/api/main.py` |
| Community Cloud presentation | `cloud/app.py`; dependencies: `cloud/requirements.txt` |
| Private server workspace | `src/pbpk_kb/web/app.py`, `web/static/` |
| Legacy Streamlit | `src/pbpk_kb/ui/streamlit_app.py`, `ui/pages/` |
| Sources and settings | `src/pbpk_kb/config/` |
| Database schema | `src/pbpk_kb/storage/schema.sql` |
| Collection and processing pipeline | `src/pbpk_kb/acquisition/`, `processing/`, `pipeline_utils.py` |
| Component recommendation | `src/pbpk_kb/recommendation/online_agent.py` |
| Model ranking and training | `src/pbpk_kb/recommendation/model_matching/` |
| Blueprints and simulation | `src/pbpk_kb/blueprints/`, `model_runtime/`, `model_assets/` |

```bash
runtime/app-env/bin/python scripts/deployment/project_python.py -m pbpk_kb.main --help
mkdir -p runtime/deployment/test_tmp
runtime/app-env/bin/python scripts/deployment/project_python.py -m pytest tests -q -o addopts= -p no:cacheprovider --basetemp=runtime/deployment/test_tmp
```

Some GPU, database, and external-network tests require explicit enablement. Temporary test outputs can be removed. Production databases, model weights, registered assets, and original evidence are not caches.

See [deployment](deployment.md) for services, [evidence pipeline](evidence_pipeline.md) and [recommendation strategy](recommendation_strategy.md) for scientific processing contracts, and [research plan](research_plan.md) for subsequent work. API exports are generated on demand.

## Reproducing frozen recommendation experiments

Historical recommendation runs bind exact source bytes to their protocols. Their source revision is `5a17a155a997831f8ce78da0337a6a981e534856`, retained in Git history. Current English reporting files are a newer revision and cannot resume those frozen runs; the original hash guards deliberately reject them.

Create a separate project-owned checkout when reproduction is needed:

```bash
git worktree add --detach runtime/historical-recommendation 5a17a155a997831f8ce78da0337a6a981e534856
```

Copy the required project-owned inputs and provide an isolated environment inside that checkout before execution. Keep its results separate from existing experiments; out-of-scope symlinks fail the original guards. This command does not run experiments. Never rewrite historical fingerprints or bypass a failed guard. Historical fixture tests retrieve exact local Git blobs, verify their original SHA256 values, and exercise preparation and evaluation without GPU inference; missing Git history produces an explicit error without a network download or silent fallback.

<a id="offline-blueprint-design"></a>

## Offline blueprint design

The current core is [source- and purpose-driven automatic construction](recommendation_strategy.md#blueprint-design). When a drug has a development curve and its blood-cell partition is only formula-predicted, an RBC calibration candidate is available. `direct_relaxed` remains the three-parameter control. The branch without a drug-specific curve is unchanged. The manuscript and frozen historical versions are preserved; these offline commands do not start the UI, backend, or external model services.

### Current entry points

The latest 30-drug repair batch is complete: `runtime/app-env/bin/python -B scripts/run_sampling_stability.py status`. Its unified entry point supports `prepare/run/all/status/report`, with at most 14 workers; results are in `experiments/blueprint_design/sampling_stability_20261009`. Independent recalculation and plotting use `scripts/report_sampling_stability.py all`; inspect the PNG before exporting a PDF with `plot --export`. Selection is locked before fitting. Previous results and the manuscript are not overwritten by these commands.

**30-drug expansion (2026-10-09):** this batch followed the request for a larger sample. The preceding ten-drug conclusions remain unchanged. The selector trained and fixed on the original ten drugs was evaluated on 20 additional drugs; sources and observed data were prepared for all 30. The 62 preparation, isolation, NCA, runner, and reporting checks passed. The batch completed 40 selection tasks and 386 additional-point fits.

```bash
runtime/app-env/bin/python -B scripts/run_sampling_expansion.py status
runtime/app-env/bin/python -B scripts/run_sampling_expansion.py all --workers 14
runtime/app-env/bin/python -B scripts/run_sampling_expansion.py report
runtime/app-env/bin/python -B scripts/run_sampling_expansion.py cleanup
runtime/app-env/bin/python -B scripts/report_sampling_expansion.py all
# Inspect comparison.png before exporting the same figure as PDF.
runtime/app-env/bin/python -B scripts/report_sampling_expansion.py plot --export
```

The batch directory is `experiments/blueprint_design/sampling_expansion_20261009`; the independent background process uses `run.pid` and `run.log`. `all` chains native construction, protocol locking, pre-observation selection, actual additional-point fitting, prediction locking, and final scoring under one process lock, and continues after SSH disconnects. Preparation progress is in `preparation.json`, experiment progress in `state.json`; `status` selects the relevant stage. Do not start concurrent instances of the same batch. Errors preserve diagnostics and pause execution without inspecting or operating on other projects. Original ten drugs, additional 20 drugs, cross-study, and within-study cases are reported separately. Current status is `complete`; fitting and scoring with 14 workers took approximately 13.6 minutes, excluding source preparation and native export. No additional-point tasks remain. This batch did not overwrite the manuscript. One endpoint CSV, one PNG/PDF comparison figure, and one README are stored together.

The preceding ten-drug entry point is `scripts/run_learned_sampling_design.py`. It reuses 140 actual additional-point results from the missing-segment 4+1 experiment and learns selection value with a fixed regularized linear model. Ten drugs are held out by nine study groups, with all candidates and layouts isolated together; PBPK is not refitted. See [strategy section 8.11.1](recommendation_strategy.md).

```bash
runtime/app-env/bin/python -B scripts/run_learned_sampling_design.py status
runtime/app-env/bin/python -B scripts/run_learned_sampling_design.py run
runtime/app-env/bin/python -B scripts/run_learned_sampling_design.py report
runtime/app-env/bin/python -B scripts/audit_learned_sampling_design.py audit
runtime/app-env/bin/python -B scripts/run_learned_sampling_design.py cleanup
```

`prepare` is for the initial freeze only and does not overwrite an existing protocol. The ten-drug batch is complete; `run` returns its status without repeating training. Evaluation took 12.6 seconds, excluding existing PBPK fitting, development checks, independent recalculation, and figure preparation. No ten-drug background task, GPU, LLM, or service dependency remains.

To resume an incomplete run under the same protocol:

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 runtime/app-env/bin/python -B scripts/run_learned_sampling_design.py run > experiments/blueprint_design/learned_sampling_20261009/run.log 2>&1 < /dev/null &
```

Source hashes and the 10 GiB limit are checked every 20 minutes and at stage boundaries. This batch took less than 20 minutes and triggered only stage checks. Calculation errors retain their cause and pause; failed outputs must not be appended repeatedly. A batch `PAUSE` file prevents further computation. Never launch two writers for the same directory. Outputs are limited to one endpoint CSV, one comparison figure, protocol/features/selections/results/status/audit records, and one report; no per-drug reports or new fitting directories are generated.

The 38 implementation checks passed. Independent recalculation verified 2,220 source hashes, 27 ridge models, 60 c-opt criteria, 5,940 endpoint rows, and 108 statistics. Primary Cmax GMFE was 1.3116 versus 1.3712 for time coverage: three drugs improved, five tied, and two worsened, missing the predefined majority-benefit threshold. Further tuning on these ten drugs stopped. Expansion is not automatic and the manuscript was unchanged at this stage. [Full results](../experiments/blueprint_design/learned_sampling_20261009/README.md).

The retained predictor and historical experiment entry points are frozen rather than listed as active tasks:

| Purpose | Entry point or result |
|---|---|
| Current four-parameter PBPK predictor | `scripts/run_blood_partition_blueprint.py`; full/five-point GMFE 1.3773/1.4099, with gains concentrated in a few drugs |
| Three-parameter control | `scripts/run_exposure_blueprint.py`; `direct_relaxed` |
| Negative result: parameter-identifiability gate | `selective_partition_20261008` |
| Negative result: adding one point to five | `endpoint_evidence_design_20261008` |
| Adding one point to an explicit observation gap | `evidence_gap_design_20261008`; additional evidence helped, but complex selection did not beat time coverage |

Native physiology, source-hash chains, original clinical observations, and manuscript results remain necessary assets and must not be removed as caches. See the [experiment index](../experiments/README.md).

### Core implementation and reuse boundaries

| Implementation | Role |
|---|---|
| `blueprint_policy/exposure_profile.py` | Development-window area constrains nonrenal CL; outer optimization controls distribution and exchange |
| `blueprint_policy/condition_matched_selection.py` | Retained grouped applicability checks for evidence |
| `blueprints/evidence_blueprint.py` | Validates the evidence ledger and constructs human, drug, and dosing snapshots |
| `blueprint_policy/transport_calibration.py` | Manuscript vascular–tissue equations and fixed calibration bounds; historical files are not edited in place |
| `blueprint_policy/perfusion_export.py` | Exports native physiology and drug partitioning, retaining parameter readback |
| `blueprints/extended_prediction.py`, `clinical_prediction_nca.py` | Endpoints from the same trajectory and NCA of clinical samples |
| `blueprints/source_adaptation.py` | Source adaptation for complete reusable models; part of the model-reuse route |

All paths above are under `src/pbpk_kb/`. Original interfaces and manuscript-bound code remain. Retired experiments are not presented as current execution instructions. Engineering executability does not establish clinical accuracy; study-arm population summaries do not establish individual parameter correlations. Missing quantitative parameters or unsupported mechanisms produce explicit gaps. Complete same-drug models and target observations must not be silently used as construction inputs.

Relevant checks:

```bash
PYTHONDONTWRITEBYTECODE=1 runtime/app-env/bin/python -B -m pytest tests/test_exposure_profile.py tests/test_exposure_blueprint_runner.py -q -p no:cacheprovider
```
