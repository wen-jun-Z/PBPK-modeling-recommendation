# PBPK deployment and verification

## Streamlit Community Cloud

The English presentation app is deployed at [PBPK Modeling Recommendation](https://pbpk-modeling-recommendation.streamlit.app/). It is self-contained and reads a committed research snapshot. The GitHub repository was made public at the owner's request. Deployment settings:

| Setting | Value |
|---|---|
| Repository | `wen-jun-Z/PBPK-modeling-recommendation` |
| Branch | `main` |
| Main file | `cloud/app.py` |
| Python | `3.12` |
| Dependencies | `cloud/requirements.txt`, beside the entry point |
| Secrets | None required |
| Access | Currently public: a fresh browser session can access the app without signing in |

To recreate the deployment in [Streamlit Community Cloud](https://share.streamlit.io/), connect the GitHub account, select **Create app**, and enter the settings above. The GitHub file URL is `https://github.com/wen-jun-Z/PBPK-modeling-recommendation/blob/main/cloud/app.py`. Streamlit account login is separate from GitHub CLI authentication. For owner-only access, set **App settings → Sharing → Who can view this app → Only specific people can view this app**, without inviting other viewers. Repository visibility and app access are separate settings; a public repository does not require a public app. The agent has not changed account-level sharing settings. See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [private app settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

The app contains Overview, Compounds & evidence, Public models, Recommendations, Blueprint design, and Knowledge graph. The snapshot contains 3,266 evidence records covering 100 compounds, 280 versions from 202 model families, 32 saved component scenarios, five saved model recommendations, and 30 executable blueprint cases. The five model-recommendation records all concern midazolam; they are saved results, not new inference for arbitrary requests. There are 2,787 verified direct evidence-source links; missing links are reported rather than inferred from bulk-download URLs.

Blueprints use the existing CPU solver and exported physiology. Dose, infusion duration, and prediction window can be changed; human conditions and the four calibrated parameters remain fixed. The interface separates calibration-study demographics from target-case demographics. New-human construction, fitting against new observations, and new GPU recommendation require the original server workspace. Cloud does not contact that server, need its credentials, or load its database or model weights. Source files and native artifacts are verified during export; the Cloud loader verifies the committed snapshot hashes. Original source files are not included, and source verification at export is not represented as new Cloud-side scientific validation.

To refresh the snapshot on the original server, using existing project configuration:

```bash
runtime/app-env/bin/python -B scripts/deployment/export_cloud_snapshot.py
```

The exporter reads one repeatable-read, read-only database transaction, validates all blueprint source and artifact hashes, removes private paths, and atomically writes `cloud/data/manifest.json` last. Inspect and commit the changed snapshot files before pushing. Do not copy server `.env`, databases, model weights, or authentication files into the repository.

To validate the portable app in its isolated environment:

```bash
runtime/cloud-env/bin/python -m pip install -r cloud/requirements.txt
runtime/cloud-env/bin/python -m pip install pytest
mkdir -p runtime/cloud-preview
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 runtime/cloud-env/bin/python -B -m pytest tests/test_ui/test_cloud_app.py tests/test_ui/test_cloud_blueprints.py -q -p no:cacheprovider --basetemp=runtime/cloud-preview/pytest-tmp
```

Verification on 2026-10-27: all 16 Cloud checks passed, including all 30 saved blueprint curves and dose-response recalculation. Seven real-browser checks covered the six pages, evidence source navigation, model and component filtering, prediction CSV download, graph rendering, and mobile layout, with no JavaScript errors. An isolated release containing only Git-tracked files also passed the Cloud checks; no server data directory or database was required.

Live Community Cloud verification, 2026-10-27: seven browser checks passed across all six pages in a fresh anonymous session. Caffeine filtering returned 134 evidence records; midazolam search returned 25 related model versions. Saved recommendations displayed concrete components, parameters, and verified source URLs. Changing Caffeine from 222 to 266.4 mg recalculated the curve and produced a CSV with 1,101 finite concentration points. The graph and 390-pixel mobile layout passed, with no JavaScript exceptions. The app-frame health endpoint returned `200 ok`. Results and screenshots are retained locally in `runtime/cloud-preview/cloud_live_check.json` and `live_*.png`. Anonymous access confirms that app privacy is not yet enabled.

`pytest` is a development dependency and is not required by the deployed app. Local browser smoke tests use a temporary authenticated loopback preview and stop their own process afterward. Preview access uses a randomly generated code held only in memory; Community Cloud access is controlled by the platform's Sharing settings; the current deployment is public. The private server workspace and evidence-processing jobs are independent of this deployment.

## Current private server workspace (2026-10-22)

English interface update, 2026-10-27: the verified workspace process was restarted after confirming no active UI jobs. All six pages rendered in English with no JavaScript errors; personal authentication and loopback binding remain enabled. Database and background jobs were retained, and legacy Streamlit remains disabled.

**PBPK Workspace** serves its frontend and FastAPI on `127.0.0.1:18620`, independently of Streamlit. Access is through SSH forwarding. It uses the existing project PostgreSQL database and reads the current evidence and model catalog. The initial deployment recorded 3,586 usable evidence records, 100 evidence-covered drugs, 202 model families, 280 versions, and 113 identity registrations. Subsequent quality review changed the serving evidence count; the Cloud snapshot above records its own export counts.

On your computer, use your normal SSH host configuration:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18620:127.0.0.1:18620 xuyiqian@192.168.78.55
```

If your normal SSH address, alias, or port differs, retain it and add only the forwarding option. Open **http://127.0.0.1:18620**. If the local port is occupied, change only the first port.

The workspace requires a personal access code, stored in `runtime/deployment/workspace/access.key` with mode `600` inside a mode-`700` directory. Read and enter it only in your own terminal and browser; do not share or commit it. The code is absent from URLs, startup arguments, and logs. Sessions last one day; logout clears the browser session. Loopback binding blocks direct LAN access, and authentication restricts other ordinary accounts on the server. This is not an isolation guarantee against server administrators.

Run from the project root:

```bash
runtime/app-env/bin/python scripts/deployment/workspace_service.py start
runtime/app-env/bin/python scripts/deployment/workspace_service.py status
runtime/app-env/bin/python scripts/deployment/workspace_service.py restart
runtime/app-env/bin/python scripts/deployment/workspace_service.py stop
```

On port conflict, startup tries another high port by binding only; use the URL reported by `status`. The process survives SSH disconnects, but requires `start` after server reboot. Stop acts only on the workspace process after checking account, working directory, startup identity, and pidfd. It preserves the database and experiments. Startup uses the existing project database directory and can restore its process if needed. Legacy Streamlit remains disabled.

- Entry point: `src/pbpk_kb/web/app.py`; frontend and styles: `src/pbpk_kb/web/static/`.
- Log: `runtime/deployment/workspace/web.log`; PID and port: `pid.json` in the same directory.
- Pages: Overview, Compounds & evidence, all model versions, model/component recommendations, Blueprint design, and Knowledge graph. Downloads remain within their relevant pages.
- Evidence links use stored PMID/PMCID/DOI rather than guessing article identifiers from a batch URL. A registered PDF can be opened after authentication and SHA-256 verification. If no verified PDF is available, the article or publisher link remains with a clear availability note. Each evidence item shows one direct source link instead of technical provenance panels.
- New model requests run the completed PBPK-Match light weights against 280 candidates on GPU. Known drugs use grouped weights excluding that drug from training; new drugs use five-fold percentile-rank fusion. Applicability tiers come first, followed by learned scores within each tier. Unsupported or unconfirmed candidates do not enter the recommendation list; full raw rankings can be downloaded. Research inference does not establish clinical validation. Jobs survive page navigation and refresh, identical requests share an active job, and inference is limited to 240 seconds.
- The request form focuses on human public-model selection: basic PK and existing-model reuse, with AUC, Cmax, and concentration curves. Population, dose, frequency, formulation, enzymes, and transporters are under additional filters; dose and frequency have no invented defaults. DDI, pediatric, hepatic/renal impairment, and tissue exposure are not offered as completed purposes. Animal records and historical results remain. Filters support matching and applicability checks; actual prediction belongs to Blueprint design. Incomplete dosing specifications explicitly prevent a direct-reuse conclusion.
- The initial UI exposed 25 saved component results and their citations. New component generation remains disabled; Qwen3-8B is not downloaded again. Snapshot exports report their actual current number of saved scenarios separately.
- Ordinary pages do not load GPU models. Recommendation runs in an isolated project process using Qwen3-Embedding-0.6B and saved small-network weights, then exits to release GPU memory. Phi-4 does not start automatically.
- Blueprints use the current 30-drug workflow: human template and drug evidence → calibration → feasibility checks and necessary repair → target prediction. Structure, parameters, evidence, saved calibration, changed dose/infusion/window, and development-curve recalibration are available. Default human conditions are the case's target conditions. The actual development study, human conditions, dose, and sample count are shown separately and may differ. New-human reconstruction and four-parameter fitting require new observations and complete corresponding human conditions. The backend rejects unobserved demographic edits and mismatched development/prediction human conditions. Age, sex, weight, and height are known inputs, not fitted parameters; the constructor's 18–80 range is not validated coverage. Dose edits reuse calibrated parameters. Historical exploratory results remain marked as uncalibrated for the changed human. Early/late sampling are experimental conditions, not UI calibration-method options. New cases use the manuscript's selected result; historical jobs retain their original inputs. Supported cases are adult single-drug, single-dose intravenous scenarios for the existing 30 drugs. Unsupported drugs are reported without substitution. Each native reconstruction stage is limited to 180 seconds and two CPU threads; total construction to 600 seconds; subsequent calibration/prediction to 180 seconds, four parameters, four initializations, and 4,096 evaluations. At most two blueprint jobs run concurrently. Human outputs are checked against 384 MiB per job and 2 GiB total; excess stops execution without deletion or retry.
- Prediction shows the actual selected-window curve, calculable PK endpoints, history, and JSON/CSV/ZIP downloads with replay code. Development-curve fit is not independent validation.
- Experiment results and evidence QA are excluded from the UI; their assets and backend interfaces remain. Blueprint curves, endpoints, and downloads remain visible.

Real verification entry point; it creates small project-local exports:

```bash
PYTHONDONTWRITEBYTECODE=1 runtime/app-env/bin/python scripts/deployment/test_workspace.py
```

The 2026-10-19 version passed 42 real API and nine-page browser checks, plus five registered-blueprint, six research-result, 15 recommendation-backend, and seven blueprint-backend checks. All 60 details for 30 drugs × two protocols were read. Real new-dose prediction, development-curve calibration, mass balance, and ZIP replay passed. Pagination, 25 component records, graph, source links, downloads, and mobile layout were checked without JavaScript errors. Records and screenshots are in `runtime/deployment/workspace/`. One real midazolam request used `cuda:0` for 280-candidate inference and completed end-to-end verification, including refresh recovery, detail, and download, in approximately 13.5 seconds; peak GPU memory was approximately 1.12 GiB. These timings describe that request only.

```bash
# Inspect retained research-result APIs without restoring their UI page.
PYTHONDONTWRITEBYTECODE=1 runtime/app-env/bin/python scripts/deployment/test_workspace.py --results-only
PYTHONDONTWRITEBYTECODE=1 runtime/app-env/bin/python scripts/deployment/test_workspace.py --blueprints-only
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 runtime/app-env/bin/python -B -m pytest tests/test_ui/test_web_research.py tests/test_ui/test_web_blueprints.py -q -p no:cacheprovider --basetemp=runtime/deployment/workspace/web-test-tmp
```

Recommendation jobs are stored in `runtime/deployment/workspace/recommendations/`, blueprint jobs in `runtime/deployment/workspace/blueprints/`. Interactive jobs are separate from frozen research; rerunning the UI does not overwrite experiments.

The 2026-10-23 demographic-interaction checks passed 44 backend tests. A real Caffeine scenario for a 40-year-old woman, 60 kg and 165 cm, completed native rebuilding and prediction in about 30 seconds. The 15 organ volumes and curve changed; relative mass-balance error was about `1.3e-12`. Refresh recovery, historical display, reset, mobile layout, provenance, ZIP replay, and rejection of unauthenticated downloads passed. Record: `runtime/deployment/workspace/human_conditions_check.json`. This established that conditions entered the calculation, not predictive accuracy. The interface was subsequently restricted: new requests cannot change human conditions without corresponding observations. Historical exploratory results remain.

The 2026-10-23 observation-bound revision passed 49 backend and 11 real browser/API checks. Caffeine reused the saved Blanchard 1983 development observations and corresponding human conditions to rebuild, fit four parameters, and predict in about 31 seconds. Demographic changes without new observations returned 422 without creating a job. Refresh display, historical labels, downloads, mobile layout, and private access passed. Record: `runtime/deployment/workspace/data_bound_calibration_check.json`. This verifies execution with existing development data, not new independent scientific validation.

## Legacy Streamlit deployment and historical verification

Initial verification: 2026-09-05; historical documentation updated 2026-09-14. The evidence batch completed. Qwen3-8B weights remain removed; online component generation and legacy Streamlit remain stopped. Database, API, and GPU embedding services follow project configuration; inspect actual current state with the service scripts.

```bash
runtime/app-env/bin/python scripts/deployment/ui_service.py stop
# Explicitly enable only when the legacy UI is needed:
runtime/app-env/bin/python scripts/deployment/ui_service.py start
```

Stop writes `runtime/deployment/ui.disabled`, preventing supervisor restarts of Streamlit, including full-stack restarts. A disabled supervisor retains only a lightweight waiting Python runner, without Streamlit or a web listener. Start removes the marker and enables the UI.

The legacy service is accessed through a personal SSH tunnel to server loopback `127.0.0.1:18602`; the previous LAN listener is closed:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18602:127.0.0.1:18602 -p SSH_PORT xuyiqian@SSH_HOST
```

Use your usual SSH host/port, without putting a password in the command. Keep the terminal open and browse to **http://127.0.0.1:18602**. On local conflict, change the first `18602` to `18603` and use the matching browser port. Closing the tunnel disconnects your browser, not the server service. Loopback forwarding blocks other LAN computers, but is not web authentication: other logged-in server accounts could access server loopback unless authentication is added. This historical configuration does not open new public or LAN ports.

### Lifecycle and status

```bash
cd /home/xuyiqian/Research_Projects/Knowledge_Graph_Model_Recommendation
scripts/deployment/start.sh
scripts/deployment/status.sh
scripts/deployment/restart.sh
scripts/deployment/stop.sh
```

`status.sh` checks project processes, three HTTP health endpoints, and database queries, returning zero when healthy. Service ownership and process identity are verified before lifecycle actions. The project `supervisor.py` uses independent sessions, closed stdin, ignored SIGHUP, and project-local logs. After the startup terminal exited, the supervisor was reparented to PID 1 and all four child services remained alive; restart was also tested. Stop uses a private control socket and affects only children created by that supervisor, not arbitrary PIDs from old files. Four earlier project-specific user systemd units were disabled to prevent duplicate startup. No boot autostart was configured; use `start.sh` after reboot.

### Historical service inventory

| Component | Listener | Role |
|---|---|---|
| Legacy Streamlit | Stopped; port 18602 released | Disabled until explicitly enabled for SSH-tunnel use |
| FastAPI | `127.0.0.1:18601` | Evidence, recommendation, QA, export, and blueprint APIs |
| PostgreSQL + pgvector | `127.0.0.1:18532` | Project `pbpk_kb` database and vector index |
| Embedding API | `127.0.0.1:18610` | Local PubMedBERT semantic retrieval, GPU 0 |
| Project supervisor | Private project Unix socket | Lifecycle, state, and child restart |

Earlier component generation used Qwen3-8B BF16, dense retrieval, reranking, an HGT index, and a trained graph projector on **GPU 0, NVIDIA RTX A6000, 48 GB**. The 16 files in `runtime/models/Qwen3-8B-BF16`, approximately 16.4 GB, were removed; retrieval, reranking, and graph artifacts remain. APIs can read saved accepted results, but new component generation is paused. QA retrieval and citation logic remain.

Evidence LLM settings are in `runtime/deployment/app.env` and reference the registered vLLM endpoint. The historical 18-step batch completed rather than remaining in extraction. `PBPK_RECOMMENDATION_ONLINE_ENABLED=false`; the old component-generation model path is empty. A replacement model needs adaptation, and the old graph-projector acceptance cannot be transferred automatically.

The user-designated evidence service was registered from PID `3947213`, with API process `3946952`, endpoint `http://127.0.0.1:18581/v1`, model identifier `copd-primary`, and vLLM 0.10.2. Configuration identified Qwen3, BF16, 40 layers, and hidden size 5120. Requests and results use the actual service identifier, not the former 8B name. `runtime/deployment/qwen_service.json` records configuration hashes and process startup identity. Only the registered API is called; its service is not restarted or modified by this project.

Extraction, compound attribution, and quality review share this endpoint. Extraction and quality review use four concurrent requests; attribution retains its scheduler. Thinking is disabled, output uses `response_format.json_schema`, and JSON Schema is checked again by the client. Truncation, model-ID mismatch, and API errors are failures. Calls use no external keys, redirects, fallbacks, or automatic retries. Account and startup identity are checked before each call; a service restart requires explicit registration verification, without port scanning.

Concurrency update, 2026-09-12: `--request-concurrency` defaults to registered `concurrent_requests`, currently four. A thread pool sends requests; the main thread validates and writes results. Eight fixed public-evidence requests took 145.7 seconds sequentially, 80.9 with four workers, and 80.6 with eight, without API errors. Four workers gave approximately 1.8× throughput; eight provided no clear gain. This small shared-service benchmark is not a universal runtime estimate. All 172 related checks passed and saved-result hashes were unchanged. The batch monitor contains `qwen_concurrency_benchmark.json`.

Integration passed 83 regression checks and 11 real-call checks. Four actual batch requests produced one scientifically accepted candidate and six rejected candidates, alongside positive/negative extraction, attribution, repair, and numeric-review checks. This was interface and gate verification, not an accuracy evaluation; test evidence was not imported. Record: `runtime/deployment/evidence_monitor/run_20260909T083120Z/qwen_server_smoke.json`.

Historical batch reproduction; run only when reprocessing is required:

```bash
runtime/app-env/bin/python scripts/deployment/project_python.py scripts/deployment/run_evidence_pipeline.py --compounds atorvastatin,bupropion,carbamazepine,clarithromycin,clopidogrel,cyclosporine,felodipine,fluconazole,furosemide,itraconazole,metoprolol,moxifloxacin,paracetamol,paroxetine,pioglitazone,quinidine,rifampicin,ritonavir,tacrolimus,voriconazole --full-extraction --request-concurrency 4
```

Background runs require an independent session with stdout appended to `runtime/deployment/logs/evidence_pipeline.log`. Monitoring uses `scripts/deployment/evidence_watchdog.py --manifest data_process/pipeline_runs/run_20260909T083120Z.json --pid BATCH_PID --retry-historical-oom`, with `runtime/deployment/logs/evidence_watchdog.log`. This batch is complete. Its PID record is `runtime/deployment/pids/evidence_pipeline.pid` and status is `runtime/deployment/evidence_monitor/run_20260909T083120Z/status.json`.

Project configuration uses `CUDA_VISIBLE_DEVICES=0`, `PBPK_RECOMMENDATION_DEVICE=cuda:0`, and `PBPK_EMBEDDING_DEVICE=cuda:0`, with dense-retrieval batch 16 and reranking batch eight. Commands launched through `project_python.py` inherit these settings; training and indexing CLI CUDA/auto options can use that GPU. Database queries, rendering, rule gates, and I/O remain on CPU.

The former Qwen3-8B runner supported `--wait-for-gpu`, checking every 60 seconds and resuming at 24 GiB free memory. That waiting task is stopped. The threshold is not a verified requirement for a replacement 16B model. Capacity status is in `runtime/deployment/evidence_monitor/<run_id>/gpu_capacity.json`.

`--retry-historical-oom` is allowed only after confirming a historical failure was CUDA OOM with no output. The monitor recalculates input hashes; only unchanged inputs with empty candidates and no output hash are retried. Original failures remain failures. New failures and evidence-validation anomalies pause execution; capacity is checked every 30 seconds and full audits every 20 minutes.

Before removal, the first uncached GPU component request took **71.51 seconds**, returning five components and eight citations with successful validation. Recorded execution used `cuda:0`, BF16, and 16.82 GB peak PyTorch allocation. This includes loading, first compilation, retrieval, generation, and validation, rather than fixed per-request latency. The earlier CPU result of 778.58 seconds used a different request and is not a controlled performance comparison.

Only one GPU component-generation task runs at a time. The 2026-09-06 concurrency fix replaced the busy 422 behavior: identical requests share generation; different requests wait; saved results take priority. An in-process cache retains at most 16 completed results for 60 seconds to cover the pre-import duplicate window. Database import uses a project transaction lock and verifies existing checksums to prevent insertion races. The UI shows waiting and elapsed time. `GET /recommendation/runtime-status` reports active, queued, and shared waiters and releases their state on completion or error. Scientific validation failures retain their normal gates.

Real concurrent A, A, B requests all returned HTTP 200. The two A requests shared one result ID; B received a separate result. At most one task ran, one distinct request queued, and one identical request shared the active task; all counters returned to zero. Record: `runtime/deployment/component_concurrency_test.json`. Startup reports success only after required services are ready.

The deterministic model-recommendation path remains. Old weak-label catalog training is stopped and its entry point disabled; it does not auto-promote a service or become a manuscript baseline. See the [research plan](research_plan.md) and [model-ranking results](../experiments/README.md).

### Preserved legacy capabilities

Entries are under `src/pbpk_kb/ui/`, with child pages in `pages/`.

| Capability | Entry | Dependencies | Historical verification |
|---|---|---|---|
| Overview statistics and charts | `streamlit_app.py` | Database | Real nonempty statistics |
| Public-source browsing and filters | `02_source_registry.py` | API, source registry | Page passed |
| Compound search | `03_compound_search.py` | Database, API | Midazolam search passed |
| Compound profile and evidence summary | `04_compound_profile.py` | Database, evidence views | Page and report export passed |
| Structured, semantic, hybrid retrieval | `05_evidence_search.py` | PostgreSQL, pgvector, Embedding API | All modes and text filters returned results |
| Public model catalog and details | `06_model_registry.py` | Catalog, registered artifacts | Expanded to 202 families and 280 versions |
| Review queue | `07_review_queue.py` | Evidence, curation API | Existing filters passed without rewriting scientific judgments |
| Evidence details and provenance | `08_evidence_detail.py` | Evidence, source locations, audit history | Real detail and source resolution passed |
| Cited QA | `09_qa_assistant.py` | Structured and semantic retrieval | A real answer returned 17 citations |
| Four export types and downloads | `10_exports.py` | API, project export directory | Evidence, compound report, model catalog, and stage summary passed |
| Ingestion monitor | `11_ingestion_monitor.py` | Import records, project logs | Page passed without a new full external collection |
| Knowledge graph and export | `12_knowledge_graph.py` | Graph API, evidence | Compound/enzyme-transporter/source/model centers and JSON passed |
| Administration | `13_admin_settings.py` | Settings, source API | Existing page and forms passed |
| Components, models, and blueprints | `14_recommendation_agent.py` | Local models, graph index, catalog, blueprint package | Saved acceptance, generation, ranking, blueprint audit, and download passed |

“Load compilation audit / Load simulation audit” reads registered execution records; it does not promise a new PK-Sim simulation. Both interfaces and the blueprint-contract validator were called, returning successful engineering records while preserving original scientific-failure states.

### Configuration and checks

Configuration: `runtime/deployment/app.env`; logs: `runtime/deployment/logs/`; PIDs: `runtime/deployment/pids/`; database: `runtime/postgres-data/`. Weights in `runtime/models/` and `runtime/hf_cache/` are runtime dependencies.

Legacy Word attachments use project-local `runtime/tools/antiword`, invoked through `runtime/app-env/bin/antiword`. It came from Debian antiword 0.37-17; download URL and checksum are in `installation.json`. No system-wide package was installed.

```bash
runtime/app-env/bin/python scripts/deployment/project_python.py scripts/deployment/smoke_test.py --mode all
runtime/app-env/bin/python scripts/deployment/project_python.py scripts/deployment/test_component_concurrency.py
```

`smoke_test.py` checks real APIs, pages, components, models, graph, QA, and downloads without launching a reachable Streamlit service. The concurrency test generates new GPU requests. Historical quality/import verification: `data_process/final_evidence/run_20260907T134918Z/00_run_manifest/database_verification.json`.

## Model execution environment (2026-09-13)

Execution is on demand, without a persistent process or Streamlit startup. Project environments are `runtime/ospsuite-env` (R 4.4.3, OSPSuite 12.4.3, rSharp 1.2.0) and `runtime/ospsuite-dotnet` (.NET 8.0.29). Sources and checksums are in `runtime/deployment/ospsuite-runtime.json`; installation logs are `ospsuite-install.log` and `ospsuite-packages-install.log` in that directory. OSPSuite supports [direct execution from JSON snapshots](https://www.open-systems-pharmacology.org/OSPSuite-R/reference/runSimulationsFromSnapshot.html).

```bash
# A new run requires a nonexistent output directory; defaults: 300 seconds and 2 GiB total output.
runtime/app-env/bin/python scripts/deployment/project_python.py scripts/run_snapshot_model_pilot.py --model-family Moxifloxacin --output experiments/model_recommendation/runs/moxifloxacin_new_probe --timeout-seconds 300
# Verify existing outputs without resimulation or overwriting acceptance records.
runtime/app-env/bin/python scripts/deployment/project_python.py scripts/run_snapshot_model_pilot.py --model-family Moxifloxacin --output experiments/model_recommendation/runs/model_execution_20260913/moxifloxacin_runtime_fixed --verify-existing
```

Each run saves `execution.log`, `pid.json`, `execution_summary.json`, and final `output_validation.json`. A PID record after completion is provenance, not proof of a live process. The parent monitors simulation and temporary output; time or size violations stop only that process group. .NET loading is incompatible with a per-file-size ulimit, so total output is checked instead. Only clearly licensed JSON with matching source hashes is executed. Results without scientific validation do not automatically promote reusable-model flags. GMN graph operations use CUDA; OSPSuite native simulation uses CPU.

Moxifloxacin completed 14 scenarios, 14 result CSVs, and 14 PKML files with engineering acceptance. Independent scientific validation remains incomplete. Legacy Streamlit remains stopped.
