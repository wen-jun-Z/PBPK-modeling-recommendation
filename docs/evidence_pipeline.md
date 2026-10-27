# Evidence Collection, Processing, and Admission

Historical batch `run_20260907T134918Z` added evidence for existing drugs. `run_20260909T083120Z` admitted 1,144 records for 20 new drugs, establishing a 30-drug, 2,872-record expansion baseline. After review on 2026-10-26, the original 30 drugs have 2,267 usable records and the additional 70 have 999, totaling 3,266. Pending-review items are separate and not counted as usable. Repair batch 1 is complete; the other 6 follow the same rules. Retain original acceptance records and current repair receipts separately; old paths/counts below are historical examples only.

Retain source documents, final evidence, review queues, and parsed documents referenced by the database. Chunks, parsed tables, and staging state can be regenerated from sources. Re-extraction requires a new processing batch; a completed manifest does not mean every intermediate file remains present.

Processing order: raw acquisition → parsing/extraction → attribution/deduplication → quality admission → indexing/import. This document preserves the scientific rules and failure conditions of all four stages.

This expansion uses the project's Qwen3.6-27B (native Transformers, NF4, 2-request microbatches) through its registered loopback service. Old Qwen3-8B was deleted; DeepSeek is not called. Automatic acceptance is not expert review. Parameter tables without preserved row/column structure enter later table recovery, not ordinary-text LLM extraction.

Expansion must explicitly supply the compound list; attribution reads the current collection plan rather than a fixed old list. Drugs already registered with models retain their IDs. Conflicting stable identifiers block linkage; do not merge salts or stereoisomers. Import fills only missing identity fields in records without human review, preserving provenance and change audits. Count new coverage from actually searchable admitted evidence, not identity registrations or downloaded files.

## Current Expansion: From 30 to 100 Compounds

Reuse the existing 18-stage code to add 70 drugs in 7 batches of 10. First cover 18 drugs with execution data but no knowledge-base evidence, then other drugs with model data and new drugs. Preserve requests, candidates, labels, and results for the original 30; expansion is not a new blinded test. Targets and execution rules: [targets.json](../data_process/knowledge_expansion_20261009/targets.json) and [protocol.json](../data_process/knowledge_expansion_20261009/protocol.json).

From the project root, view colored progress bars refreshing every 2 seconds for the overall workflow, current batch, evidence coverage, current stage, and download activity (`Ctrl+C` exits only the viewer):

```bash
runtime/app-env/bin/python scripts/deployment/expand_knowledge_base.py watch
```

`status` shows one state snapshot. `start` is only for an unstarted queue and rejects duplicate starts when state exists. `resume-controller` takes over surviving acquisition/monitor processes after controller exit, verifying identity before continuing without duplicate downloads; it does not clear unaudited anomaly pauses. Logs/state reside in `runtime/deployment/evidence_expansion_20261009/`. Each batch's first 10 actual LLM requests must pass source/candidate checks before full processing; embedding and deduplication use local GPU models. Check capacity every 30 seconds and audit fully every 20 minutes. Anomalies pause this project's writers and stop the queue, without infinite retries. Keep Streamlit stopped.

`scripts/deployment/follow_evidence_expansion.py run` checks registered project processes every 60 seconds and records an audit state every 20 minutes. It reconnects an unexpectedly exited controller only when writers and monitors are healthy and the queue remains running, at most 3 times per batch with restart-persistent counts. Anomaly pauses remain paused, with reasons in the progress display and `runtime/deployment/evidence_expansion_20261009/followup/status.json`; automatic chat notification is not promised. Newly started acquisition processes try HTTP 429/500/502/503/504 at most 3 times, then stop strictly. Resumed old processes retain memory and do not hot-load new retry code. openFDA recovery may be registered only after all pages, file checksums, and complete filtering records are verified. Retain original failures; HTTP 500 is not empty data.

Even with HTTP 200, DailyMed labels must be valid HL7 XML with the requested `setId`. After 3 body-validation failures, try the [official fixed-version ZIP](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm) once, capped at 32 MiB. Check member paths, CRC, unique main XML, label identity, and version, then register the actual ZIP URL. Fallback failure still pauses. Existing `None` responses become nonextractable historical records only after recovering an archive of the same identity/version. Preserve original files and parsing failures, then reaccept inventory/parsing. A parsing failure must not serve as a successful deduplication reference for another file.

For official PMC XLS files, if byte size, SHA-256, and official single-part ETag all agree but the OLE directory demonstrably exceeds EOF, record an `unsupported` gap meaning corrupt official source content, unparsed. Retain originals and generate neither tables nor evidence. This exception does not cover incomplete local downloads, uncertain sources, or other parsing errors. Verified gaps may be retried individually, followed by complete batch source/statistic checks; preserve failures and do not redownload accepted material.

October batches initially capped acquisition at 100 PMC full texts and 25 OA packages per drug, but later completion stages also processed deferred items; initial manifests are not final missing-download counts. File-level review on 2026-10-26 found, among 1,369 initially deferred entries, 1,060 with parsed full text, 304 with abstracts only, and 5 with metadata only. Count download success, full-text acquisition, completed parsing, and evidence validation separately; abstracts are not full text. New ordinary acquisition jobs retain code defaults of 250 full texts/100 OA packages without rewriting historical configuration.

Original expansion appended records while checking fingerprints of the existing 2,872. Current review runs in isolation and revises only validated records from the corresponding batch, preserving other batches and frozen experiments. View repairs with `scripts/deployment/repair_expansion_quality.py watch`. `scripts/deployment/follow_evidence_expansion.py quality-start` checks the repair queue each minute and audits every 20 minutes. Only when source text, prompt, and complete service receipt establish generation truncation may each batch recover once by splitting original fragments/parameter categories. Do not concatenate broken JSON or reset failure counts. Recovery failure, other errors, manual pauses, and disk anomalies stop execution and appear in the same progress view.

Increase usable-evidence counts only after all quality stages, import, and retrieval acceptance pass. Retain source gaps, human-review items, and drugs with no admitted evidence separately; identity records cannot inflate coverage. Preserve early anomalies before full discovery and inspect paused processes; arbitrary early interruptions do not support unconditional restart.

## High-Precision Raw-Source Retrieval Strategy

### 0. Overall Order of the Three Strategies

```text
Raw-source retrieval
  -> Raw-source parsing and candidate evidence extraction
  -> Final-evidence compound attribution, domain gating, deduplication, and conflict analysis
  -> Build RAG retrieval indexes only after the final evidence layer is complete
  -> Database, UI, Q&A, and exports
```

This strategy executes only stage one. Outputs are immutable raw files and retrieval manifests, not parsed documents, evidence, embeddings, or database records.

### 1. Objective

Obtain public, traceable raw sources directly relevant to PBPK/PK for each target compound.

Completeness means every discovered candidate has a manifest entry and handling decision, not downloading every full text, label, or shared database file that happens to mention the drug.

This stage downloads/saves raw files and selection metadata only; it extracts no parameters, judges no scientific conclusions, and writes no evidence.

### 2. Principles

1. Discover candidates before deciding what to download.
2. Confirm the exact target by name or stable identifier.
3. Download full text, PMC packages, label XML, and trial details only for highly relevant candidates.
4. Retain undownloaded candidates with metadata, reasons, and retryable status.
5. Separate shared-source snapshots from compound material; shared snapshots are excluded from compound runs by default.
6. Preserve original API/PDF/XML/JSON/ZIP/CSV files unchanged; `assets.jsonl` is only an index.

### 3. Compound Identity Scope

First register each compound in an identity-query manifest and preserve original responses covering:

- Generic name and canonical spelling;
- PubChem CID, InChIKey, RxCUI, ChEMBL ID, and UNII as actually returned;
- Salts, prodrugs, metabolites, and stereoisomers;
- Brand, formulation, and combination-product names.

`identity_coverage_manifest.json` records raw identity-discovery responses and missing sources, not cross-source canonical identity merging. Label each candidate with one relationship:

| Label | Meaning | Full-text download allowed? |
|---|---|---|
| `exact_target` | Directly studies the target drug | Yes |
| `target_component` | Explicitly studies this active ingredient in a combination | Requires component-level confirmation |
| `related_form` | Salt, prodrug, or metabolite | Store separately; do not merge automatically |
| `mentioned_only` | Background mention only | No |
| `other_component` | Another ingredient in a combination | Excluded from target-drug corpus |

### 4. Execution Workflow

```text
Confirm identity
  -> Discover candidate IDs/files
  -> Screen structured fields and titles/abstracts
  -> Route to fulltext_selected / metadata_only / excluded
  -> Download selected raw sources
  -> Write asset, failure, coverage, and selection manifests
```

Each source yielding candidate lists produces:

- `candidate_records.jsonl`: all discovered candidates;
- `fulltext_selected_records.jsonl`: candidates selected for full text/details;
- `metadata_only_records.jsonl`: metadata retained without full-text download;
- `excluded_records.jsonl`: excluded candidates and reasons.

These selection files are retrieval metadata, not scientific evidence.

### 5. PubMed and PMC

#### 5.1 Query Method

Literature queries must satisfy both:

```text
Target drug in Title
AND
Relevant PK / ADME / DDI / special-population / PBPK / formulation topic in Title
```

Topic groups:

| Group | Example query terms |
|---|---|
| Clinical PK | pharmacokinetic, AUC, Cmax, Tmax, clearance, half-life, bioavailability |
| ADME | absorption, distribution, protein binding, metabolism, excretion, CYP, UGT, transporter |
| DDI | drug interaction, coadministration, inhibitor, inducer, victim drug, exposure |
| Special populations | renal impairment, hepatic impairment, pediatric, pregnancy, lactation |
| PBPK | physiologically based pharmacokinetic, PBPK, PK-Sim, MoBi, Simcyp |
| Formulation/food effects | food effect, formulation, dissolution, permeability, modified release |

Special-population and formulation/food-effect queries also require explicit PK terms in Title/Abstract, such as `pharmacokinetic`, `AUC`, `Cmax`, `clearance`, or `bioavailability`. Avoid broad drug-name-only full-text searches. MeSH, supplementary concepts, or incidental abstract mentions alone do not justify downloading full text.

#### 5.2 Literature Routing

Download ESummary/EFetch XML for matched PMIDs in batches of at most 100, retry failures, and apply title/abstract rules:

| Decision | Condition | Next action |
|---|---|---|
| `fulltext_selected` | Target drug and topic both in title, with explicit PK/ADME/DDI/PBPK terminology or abstract PK support for a special-population/formulation topic | Resolve PMID to PMCID; download OA XML, PDF, supplements |
| `metadata_only` | Drug only in abstract, topic absent from title, or role/study target unclear | Retain PMID, DOI, abstract, and reason without full text |
| `excluded` | Background mention, food/supplement/botanical content, drug not the study target, or topic mismatch | Retain exclusion record without full text |

For example, caffeine enemas promoting excretion of toxic products should not trigger a caffeine `excretion` full-text download, whereas caffeine clearance in healthy volunteers should enter the queue.

Borderline records may remain `metadata_only`, with later LLM-assisted classification based only on title, abstract, and retrieval context. The LLM does not read full text to decide scientific facts or generate evidence here.

#### 5.3 PMC, PDFs, and Attachments

Query `pubmed_pmc` links only for `fulltext_selected` PMIDs; each PMID–PMCID mapping must be verifiable.

For open-access articles, download:

- PMC EFetch XML;
- OA-package metadata;
- Official NXML/XML, PDFs, supplements, tables, and figures;
- License and download status.

Request PMC ELink/XML in batches of at most 20 with retries. Retain failed batches in `failures.jsonl`; never skip silently.

Without a legal open route, retain only PMID, DOI, abstract, access status, and reason. Failed PMC-link batches must be labeled link-query failure/pending retry, not confirmed absence of open full text. Do not scrape paywalled publisher full text.

`--max-pmc-fulltext-per-compound` and `--max-pmc-oa-packages-per-compound` are runtime safety caps, not relevance decisions; `0` disables the count cap. Rank excess candidates by corresponding PubMed title/abstract screening scores and record deferred entries in the gap manifest rather than dropping them.

### 6. Structured-Source Rules

| Source | Discovery and screening | Downloaded material |
|---|---|---|
| PubChem / RxNorm / ChEMBL | Verify exact CID, InChIKey, RxCUI, ChEMBL ID; name search is for identity discovery only | Original JSON, synonyms, structures, property records |
| DailyMed | Fetch SPL list first; download XML only when target drug exactly appears in title and, by default, is not a combination product | SPL list pages, selected SPL XML, versions, set IDs |
| openFDA | Exact generic-name query; retain identity labels for combinations | Original label JSON and version information |
| PK-DB | Exact study list from `studies/?substance=<compound>`; verify object counts with `filter/?studies__substance=<compound>&concise=false` | Study details; official `filter` ZIP with `download=true`; group, individual, intervention, output, timecourse, scatter, metadata |
| ClinicalTrials.gov | Discover via `query.intr=<compound>` plus PK-topic `query.term`, safe 100-record pages requesting only NCT ID/title/summary/interventions; select studies where target is intervention or PK probe and PK topic is present | Original candidate-page JSON; complete study JSON only for `fulltext_selected` NCTs |
| OSP model library | Separate shared repository cache; select target models by filenames, descriptions, compound metadata | PKML, parameters, scripts, reports, version information |

DailyMed combination products default to `metadata_only`. Download combination XML only with explicit `--include-combination-labels`; later still verify that text concerns the target ingredient.

Collect PK-DB in this order:

1. Paginate `studies/?substance=<compound>` for study SIDs and save every study detail.
2. Save all discoverable-object counts from `filter/?format=json&studies__substance=<compound>&concise=false`; additionally request `studies__licence__in=open` for publicly reusable counts/export UUID.
3. Add `download=true` to the same filter URL and save the official original ZIP with URL, time, headers, size, and SHA-256.
4. Extract safely into an immutable directory, rejecting absolute paths, `..`, symlinks, duplicate members, and oversized archives.
5. Record each member path, size, and SHA-256; count unique study/group/individual/intervention/output/timecourse/scatter objects by stable CSV primary keys and reconcile against the same filter request.
6. Empty/404 list or individual-object APIs must not overwrite existing nonzero filter counts as no data. Use official ZIPs as the authoritative public output/timecourse download route and record list-API status.
7. If the open-license ZIP lacks observation tables, use reference PMIDs in study details to verify PMCIDs through NCBI ID Converter/ELink and download original-paper PMC OA XML, PDF, figures, tables, and supplements. Re-extract observations from paper originals only; PK-DB is a study-discovery source, not permission to copy closed-license curated values.

PK-DB concentration-time data enter processing only after successful open-license ZIP download, unique-object-count verification, and exact study/compound identity matching. HTTP 200 with empty `outputs.csv` or `timecourses.csv` is `observation_tables_missing`, not complete. If ZIP is unavailable, retain study metadata with `official_bulk_export_unavailable`.

After monitoring verifies official URL, target drug, ZIP/audit SHA-256, and that the sole discrepancy is empty observation tables, classify `source_gap_metadata_only` and continue other drugs/sources. Preserve original failures, nonzero index counts, and gaps; empty exports do not enter extraction. Network failures, corruption, other count anomalies, and disk overruns still stop execution and cannot be ignored automatically.

For an openFDA exact-generic-name first-page 404, reverify the official response explicitly states `NOT_FOUND / No matches found!`, then preserve response, checksum, and gap before continuing. This means only that the queried name missed, not absence of labels under other names. Ordinary 404 pages, server errors, and missing later pages still require stopping.

For ChEMBL name-query HTTP 500/502/503/504, allow one size-/timeout-bounded recheck only after the batch's exact-name PubChem/CID/InChIKey and RxNorm originals are complete and verified. Persistent server errors remain preserved raw responses, with ChEMBL marked a retryable source gap and excluded from evidence; do not infer compound absence. Other sources continue. This does not weaken identity, attribution, or quality gates or exempt corruption, ambiguous identity, or other-source network failures.

If recheck returns HTTP 200, verify a unique exact name, ChEMBL ID, and InChIKey agreement with PubChem before registering recovered identity material for normal parsing. Do not keep it classified as a source gap or equate source recovery with validated evidence.

### 7. Shared-Source Snapshots

DrugCentral, HPA, GTEx, NHANES, AACT, EMA, OSP, and similar resources are source-level databases, not direct corpora for individual drugs.

```text
source_cache/<source>/<snapshot_id>/
  Official snapshots, repositories, versions, checksums

compound_runs/<run_id>/
  Only screened compound-related material and selection manifests for that run
```

Compound runs do not download shared snapshots by default. Require both `--include-shared-sources` and at least one `--shared-source <source_id>`, writing a separate cache and `snapshot_manifest.json`. This avoids copying/scanning entire HPA/NHANES/OSP databases for every 10-compound run.

### 8. Outputs and Traceability

A formal run directory contains at least:

```text
RUN_STATUS.json
acquisition_plan.json
download_plan.json
requests.jsonl
assets.jsonl
failures.jsonl
deferred.jsonl
summary.json
compounds/<compound>/selection/<source>/
```

Each selection record includes at least:

```text
compound
source_id
source_record_id
decision
selection_reason
selection_score
query_family
identity_match_type
source URL / request_id
timestamp
```

Each downloaded asset includes at least:

```text
asset_id
source_id
source_record_id
storage_path
checksum
content_type
downloaded_at
request_id
```

`deferred.jsonl` records only deferrals caused by configured download caps; network, access, and file errors belong only in `failures.jsonl`.

The only handoff is a formal run marked completed/accepted in `RUN_STATUS.json`, plus explicitly referenced shared snapshots. Partial/debug/superseded runs must not enter parsing.

### 9. Acceptance Criteria

A completed high-precision retrieval run satisfies:

1. Every compound has identity, label, structured-PK, trial, and literature coverage manifests.
2. Every downloaded full text has an explicit target-drug/topic match rationale.
3. Every PMC full text traces to a selected PMID.
4. Combinations, background mentions, and weakly related material stay out of full-text queues.
5. Shared and compound-specific material have separate directories, manifests, and refresh cycles.
6. Every undownloaded, restricted, over-cap, or failed item has an explicit reason.
7. This stage performs no numerical extraction, evidence generation, embedding, or KG construction.

### 10. Execution

Formal runs use only the unified entry point. First check fixed order and output paths:

```bash
python -m pbpk_kb.main --plan
```

Complete run:

```bash
PBPK_LLM_API_KEY="<api-key>" \
python -m pbpk_kb.main \
  --raw-output-root "/Volumes/SU700/pbpk retrieval resul" \
  --processing-output-root "/Volumes/SU700/pbpk processing result" \
  --shared-source osp_pbpk_model_library \
  --embedding-device cuda
```

The main function must execute `raw_acquisition -> raw_finalization -> pmc_static_transfer -> raw_reconciliation`; proceed to processing only when `FINAL_ACQUISITION_RESULT.json.accepted_as_complete_raw_content=true`.

Resume existing runs through the same entry:

```bash
python -m pbpk_kb.main \
  --raw-run-root "/Volumes/SU700/pbpk retrieval resul/compound_runs/<run_id>" \
  --processing-output-root "/Volumes/SU700/pbpk processing result" \
  --embedding-device cuda
```

By default, the main function finds the latest incomplete unified manifest for the same compound list and skips accepted stages. Only explicit `--new-run` creates a new run. Manual commands must not skip or reorder stages.


## Raw-Source Processing Strategy

### Position in the Overall Workflow

```text
Completed formal raw run
  -> This strategy: inventory / parse / deterministic_extract / llm_extract / table_recovery / completeness
  -> Final evidence: attribution / domain gating / deterministic deduplication / semantic evidence deduplication / conflict analysis
  -> After final evidence: RAG retrieval embeddings, database, UI, Q&A, exports
```

This strategy produces traceable candidate evidence, not final deduplication or a RAG vector index.

### 0. Inputs

Formal raw run:

```text
/Volumes/SU700/pbpk retrieval resul/high_precision_full_20260713_v2/compound_runs/run_20260713T163840Z/
```

Final shared-source snapshots:

```text
/Volumes/SU700/pbpk retrieval resul/high_precision_full_20260713_v2/source_cache/osp_pbpk_model_library/snapshot_20260714T084230Z/
```

Before processing, verify:

1. Read `RUN_STATUS.json` and require:
   - `status = completed`
   - `accepted_as_final_result = true`
2. Read and verify the following files exist:
   - `summary.json`
   - `assets.jsonl`
   - `failures.jsonl`
   - `download_plan.json`
   - `finalization/FINAL_ACQUISITION_RESULT.json`
   - `finalization/FINAL_INTEGRITY_AUDIT.json`
   - `finalization/failure_resolutions.jsonl`
3. Require `accepted_as_complete_raw_content = true` in final acquisition results.
4. Require completeness audit `status = passed` and `issue_count = 0`.
5. Process only this completed run and its referenced final shared snapshots, not partial/debug/superseded runs.

### 1. Outputs

Write processing results to:

```text
/Volumes/SU700/pbpk processing result/raw_processing/run_20260713T163840Z/
```

Do not default to storing processing results on the system disk or in the project directory.

Generate:

```text
00_run_manifest/processing_summary.json
00_run_manifest/inventory_summary.json
00_run_manifest/parse_summary.json
00_run_manifest/parse_progress.json
00_run_manifest/deterministic_extract_summary.json
00_run_manifest/deterministic_extract_progress.json
00_run_manifest/pipeline_state.json
00_run_manifest/acquisition_failure_status.jsonl
00_run_manifest/asset_parse_status.jsonl
00_run_manifest/completeness_matrix.csv
00_run_manifest/completeness_provenance.jsonl
00_run_manifest/completeness_summary.json
00_run_manifest/completeness_progress.json
00_run_manifest/processing_failures.jsonl
01_raw_inventory/raw_inventory.jsonl
01_raw_inventory/raw_asset_aliases.jsonl
02_parsed_outputs/parsed_documents.jsonl
02_parsed_outputs/parsed_records.jsonl
02_parsed_outputs/parsed_tables.jsonl
02_parsed_outputs/structured_record_processing_status.jsonl
02_parsed_outputs/structured_table_processing_status.jsonl
02_parsed_outputs/archive_members.jsonl
02_parsed_outputs/asset_links.jsonl
03_chunks_and_embeddings/text_chunks.jsonl
03_chunks_and_embeddings/chunk_processing_status.jsonl
04_evidence_candidates/evidence_candidates.jsonl
04_evidence_candidates/evidence_candidates_deterministic.jsonl
04_evidence_candidates/evidence_candidates_llm.jsonl
04_evidence_candidates/evidence_pointers.jsonl
05_llm_extraction/llm_agent_prompt.md
05_llm_extraction/llm_extraction_requests.jsonl
05_llm_extraction/llm_extraction_responses.jsonl
05_llm_extraction/llm_extraction_requests_live.jsonl
05_llm_extraction/llm_extraction_responses_live.jsonl
06_table_recovery/recovered_tables.jsonl
06_table_recovery/recovered_table_rows.jsonl
06_table_recovery/table_recovery_status.jsonl
06_table_recovery/table_evidence_candidates.jsonl
06_table_recovery/table_evidence_duplicates.jsonl
06_table_recovery/table_review_queue.jsonl
06_table_recovery/table_recovery_summary.json
06_table_recovery/table_recovery_progress.json
```

`03_chunks_and_embeddings` retains an existing directory name; this strategy does not create `embedding_index.jsonl`. Generate RAG retrieval indexes separately only after final evidence is complete.

Raw/parsed documents may enter a staging database at this stage, but candidate evidence is not the UI's default trusted layer. Update final evidence tables, UI, Q&A, and exports only after the final-evidence strategy completes:

```text
raw_document
parsed_document
document_chunk
physicochemical_property_evidence
clinical_pk_evidence
adme_parameter_evidence
ddi_evidence
special_population_evidence
tissue_expression_evidence
public_model_registry
```

All machine-generated evidence:

```text
extraction_method = machine_extracted
curation_status = needs_review
```

### 2. Current Raw Data

Current formal corpus:

| Metric | Count |
|---|---:|
| compounds | 10 |
| primary run assets | 16482 |
| final OSP snapshot ledger rows | 15 |
| expected inventory rows | 16497 |
| source IDs | 11 |
| historical failure records | 647 |
| unique historical failure requests | 645 |
| unresolved acquisition failures | 0 |

All sources:

| source_id | inventory assets |
|---|---:|
| `pmc_oa` | 10338 |
| `dailymed` | 2709 |
| `pkdb` | 1455 |
| `clinicaltrials` | 1194 |
| `pubmed` | 672 |
| `openfda_drug_label` | 43 |
| `pubchem` | 30 |
| `rxnorm` | 20 |
| `osp_pbpk_model_library` | 16 |
| `chembl` | 10 |
| `identity_discovery` | 10 |

The 16 `osp_pbpk_model_library` rows comprise 1 failed-snapshot audit manifest in the primary run, plus 14 source assets and 1 final snapshot manifest in the final ledger. `identity_discovery` is internal acquisition-stage identity discovery, not an independent scientific source. Step 1 must recompute file-family counts from extensions, content types, and magic bytes rather than reuse an old run's statistics.

### 3. Overall Workflow

```text
1. Read manifests
2. Build raw inventory
3. Verify checksums and file presence
4. Parse raw files by family
5. Generate parsed_documents / parsed_tables / text_chunks
6. Extract structured fields through direct mappings
7. Determine LLM need for unstructured text/complex tables by chunk
8. Write chunk_processing_status.jsonl for every chunk
9. Call the LLM only for llm_required chunks not already processed successfully
10. Validate schema, units, numbers, and source_location
11. Recover flattened tables lacking reliable row/column boundaries
12. Extract complete rows deterministically; send complex rows to table review
13. Merge evidence_candidates
14. Write completeness_matrix
15. Write processing_summary
```

Do not redownload raw sources during processing.

### 3.1 Stage Execution Contract

Each command advances only one explicit stage:

```text
inventory -> parse -> deterministic_extract -> llm_extract -> table_recovery -> completeness
```

Rules:

1. `inventory` writes inventory and terminal acquisition-failure states only; no content parsing.
2. `parse` creates only parsed documents, parsed tables, and text chunks.
3. `deterministic_extract` runs only structured mappers and rule extraction.
4. `llm_extract` handles only chunks classified `llm_required`.
5. `table_recovery` handles only `deferred_unparsed_flattened_table`, first reusing `parsed_tables`, then recovering missing tables from raw files; it must not recall the body-text LLM.
6. Run `completeness` after both text and table evidence processing finishes.
7. Passing `completeness` completes this strategy; `pipeline_state.embedding = not_started` is expected, not failure.
8. Update `pipeline_state.json` after every stage; downstream stages must verify upstream acceptance.
9. A command must not implicitly skip upstream stages that have not passed.
10. Final semantic deduplication belongs to the final-evidence strategy; do not embed all chunks here as a substitute.

### 3.2 Structured and Unstructured Data Rules

Extract structured data directly in code, without an LLM.

Structured data include:

```text
Explicit JSON fields
Explicit XML nodes
Explicit CSV/TSV/XLSX columns
API-returned IDs, statuses, dates, names, and numerical fields
```

Field-mapping workflow:

```text
Read parsed document/table
Identify source_id and field path
Read values with a source-specific mapper
Convert units and numbers
Write evidence_candidates
Record source_id/raw_asset_id/source_location/checksum/source_text
```

Only unstructured data enter LLM-eligibility assessment.

Unstructured data include:

```text
PDF body text
Drug-label narrative paragraphs
PubMed/PMC abstracts or full-text paragraphs
FDA/EMA review narrative
READMEs, model descriptions, report text
Complex table fragments not interpretable from column names alone
```

LLM workflow:

```text
Read text_chunks.jsonl
Classify every chunk
Skip machine JSON, search-result lists, navigation, very short text, and text without evidence terms
Skip machine-text blocks dominated by XML/HTML markup
Compound-retrieved PubMed, ClinicalTrials, OpenFDA, PKDB, PMC, and EMA text requires the target compound's name in the chunk before LLM processing
DailyMed permits only core sections: Clinical Pharmacology, Pharmacokinetics, Absorption, Distribution, Metabolism, Excretion, Drug Interactions, Specific Populations, Renal/Hepatic Impairment, Pediatric/Geriatric, Pregnancy/Lactation
DailyMed intermediate chunks without section headings are excluded unless the chunk itself begins with one of these core headings
Structured/registry/descriptive sources such as AACT, DrugCentral, HPA, NHANES, and OSP tools docs do not enter the LLM
Call the LLM only for filtered natural-language chunks containing PK/ADME/DDI/special-population/PBPK signals
LLM output must include source_text
source_text must occur verbatim in the original chunk
After validation, write evidence_candidates_llm.jsonl
Append to evidence_candidates.jsonl
```

Every chunk writes:

```text
chunk_processing_status.jsonl
```

Status fields:

```text
structured_or_machine_chunk
structured_or_rule_extracted
skipped_too_short
skipped_search_result_list
skipped_no_evidence_terms
skipped_non_natural_language
llm_required
pending_llm_classify_only
llm_processed
llm_processed_existing
llm_failed
pending_llm_request_limit
```

Resume rules:

```text
Do not recall request_ids already marked status=ok in llm_extraction_responses_live.jsonl
Do not duplicate existing evidence_ids
Failed requests may be retried
Regenerate chunk_processing_status.jsonl each time to reflect current status
```

### 4. End-to-End Provenance Contract

Every processing artifact must trace back to its raw file.

Every parsed document, parsed table, text chunk, LLM request/response, evidence candidate, and completeness row retains:

```text
run_id
source_id
raw_asset_id
storage_path
source_url
checksum
source_location
```

When upstream parsed outputs exist, also retain:

```text
parsed_record_id
parsed_document_id
parsed_table_id
chunk_id
```

File-type-specific `source_location` formats:

```text
JSON: <source_id>:<file>.json#json_path=<path>
XML: <source_id>:<file>.xml#xml_path=<path>
PDF page: <source_id>:<file>.pdf#page=<page>
PDF table: <source_id>:<file>.pdf#page=<page>#table=<table>
HTML: <source_id>:<file>.html#css=<selector>
Table cell: <source_id>:<file>.<ext>#sheet=<sheet>#row=<row>#col=<col>
Archive inner file: <source_id>:<archive>!<inner_path>
Repository file: <source_id>:<repo_path>#file=<path>
```

At evidence level, additionally retain:

```text
source_text
source_text_hash
extraction_method
extractor_name
extractor_version
extraction_confidence
curation_status
```

Rules:

1. Records without `raw_asset_id` must not enter parsed/evidence outputs.
2. Records without `source_location` must not enter parsed/evidence outputs.
3. Every evidence candidate requires `source_text`.
4. Locate `source_text` within a parsed document, parsed table, or text chunk.
5. `checksum` comes from raw inventory and must not be fabricated.
6. LLM output cannot replace `source_text`; it may only quote existing original passages/table cells.
7. Aggregated conclusions record their constituent evidence IDs.
8. Conflict decisions record all compared evidence IDs.

### 5. Step 1: raw inventory

Merge the primary run's and final OSP snapshot's `assets.jsonl` into `raw_inventory.jsonl`, deduplicating by `asset_id` without reading other shared snapshots.

Every row includes:

```text
run_id
raw_asset_id
source_id
compound
group
storage_path
source_url
downloaded_at
checksum
content_type
file_extension
file_family
file_size
request_id
asset_metadata_json
```

Processing rules:

1. Check that `storage_path` exists.
2. Recompute the file checksum.
3. On mismatch, write `processing_failures.jsonl` and do not parse the file.
4. Assign `file_family` by extension, content type, and magic bytes; `.bin`, `.dat`, and `.unk` are not unknown merely because of their suffix:
   - `json`
   - `xml`
   - `pdf`
   - `table`
   - `html`
   - `archive`
   - `repo`
   - `text`
   - `document`
   - `presentation`
   - `image`
   - `media`
   - `database`
   - `specialized_binary`
   - `unknown`
5. Magic bytes override unknown/incorrect extensions. This batch's `.unk` SQLite headers mean `database`; `.pzf` means `specialized_binary`; JSON headers in 424 `.fcgi` files mean `json`, not XML by suffix.
6. Business-format extensions override underlying ZIP/OLE container magic: `doc/docx` → `document`, `ppt/pptx` → `presentation`, `xls/xlsx` → `table`. DOCX/PPTX must not be classified as `archive`, and DOC/PPT must not be classified as `table`. Read OOXML `[Content_Types].xml`; if the package's main type disagrees with its extension, dispatch by actual internal type and record `declared_file_family`, `detected_file_family`, and `ooxml_type_mismatch` in terminal asset status rather than discard the file.
7. `jpg/jpeg/png/tif/tiff/eps` → `image`; `mp3/mp4/avi` → `media`; `rar` → `archive`.
8. Deduplicate `failures.jsonl` by `request_id` and join `finalization/failure_resolutions.jsonl`.
9. Write all historical failures and final states to `acquisition_failure_status.jsonl`.
10. `resolved` does not enter `processing_failures.jsonl`.
11. `not_available_from_current_api` enters completeness status, not parser failures.
12. Only requests without terminal resolution create current acquisition gaps; expected count for this run is 0.
13. Failure records do not enter evidence.

Step 1 acceptance:

```text
raw_inventory rows = 16497
unique raw_asset_id = 16497
missing storage_path = 0
file_size mismatch = 0
checksum mismatch = 0
unique historical failure requests = 645
terminal failure statuses = 645
unresolved acquisition failures = 0
unknown file_family = 0
```

This batch's Step 1 file-family counts must be:

```text
archive = 272
database = 6
document = 213
html = 4
image = 3592
json = 4125
media = 9
pdf = 1002
presentation = 5
repo = 4
specialized_binary = 2
table = 73
text = 703
xml = 6487
unknown = 0
```

### 6. Step 2: parser registry

Implement a parser registry:

```text
file_family -> parser
source_id + file_family -> source-specific parser
```

All parsers emit a common structure:

```text
run_id
parsed_document_id
raw_asset_id
source_id
compound
storage_path
source_url
checksum
parser_name
parser_version
content_kind
title
section_title
source_location
text
text_hash
metadata_json
```

Write structured JSON/XML/API records to `parsed_records.jsonl`:

```text
run_id
parsed_record_id
raw_asset_id
source_id
compound
storage_path
source_url
checksum
parser_name
parser_version
record_type
source_location
record_json
record_hash
metadata_json
```

`parsed_tables.jsonl` must include:

```text
run_id
parsed_table_id
raw_asset_id
source_id
compound
storage_path
source_url
checksum
parser_name
parser_version
source_location
sheet_name
page_number
table_index
row_count
column_count
columns_json
cells_json
table_text
table_hash
metadata_json
```

Write archive members to `archive_members.jsonl`, including archive raw asset, internal path, member checksum, compressed/uncompressed sizes, family, parse status, and duplicate target. Write cross-file references to `asset_links.jsonl`, including JATS links from figures/tables to images, supplements, and archive members.

Cross-file linking: link directly when PMCID plus normalized filename identifies exactly one asset. If multiple candidates share an identical checksum, link to the canonical asset designated by `raw_asset_aliases.jsonl` and retain all candidate IDs. Mark `ambiguous` only when candidate checksums differ. Without a raw asset, mark `unresolved` for completeness handling; do not fabricate a target. `parse_summary.json` separately counts `resolved`, `unresolved`, and `ambiguous` links rather than conflating conflicts and missing targets.

`text_chunks.jsonl` must include:

```text
run_id
chunk_id
raw_asset_id
parsed_document_id
parsed_table_id
source_id
compound
storage_path
source_url
checksum
source_location
chunk_index
chunk_text
chunk_text_hash
token_count
metadata_json
```

Write every parser failure to:

```text
processing_failures.jsonl
```

Every raw asset has exactly one terminal state in `asset_parse_status.jsonl`:

```text
parsed
metadata_only
unsupported
duplicate_linked
failed
```

Rules:

1. Select a canonical raw asset by `(checksum, file_family, parser_version)` and parse identical content once.
2. Record other identical assets in `raw_asset_aliases.jsonl`, retaining alias `compound`, `source_id`, `request_id`, URL, and canonical asset ID; preserve compound-retrieval context.
3. `duplicate_linked` is a successful terminal state, not failure; subsequent attribution must also read aliases.
4. Rerunning `parse` clears/rebuilds only parse outputs, never accepted inventory outputs.
5. `parse` must not write evidence, LLM requests/responses, or embeddings.
6. `parse_summary.json` reports terminal asset counts, canonical/alias counts, parser counts, parsed records/documents/tables/chunks, archive members, unsupported items, and failures.
7. During full runs, update `parse_progress.json` at least every 100 raw assets with totals, processed count, current asset, terminal counts, and timestamp; set `completed` when finished.
8. Dispatch parsers by verifiable actual format: package content type for OOXML, magic bytes for archives. Extension disagreement yields an audit warning, not automatic failure.

A `.docx` beginning with an OLE header is parsed as legacy Word, not by mistaking its embedded theme ZIP for the document. HTML under an Office extension is parsed as HTML. Identify AppleDouble resource metadata and Office lock files by both names and headers; retain member paths/checksums with `metadata_only`, not as body text or corruption.

Step 2 acceptance:

```text
asset_parse_status rows = 16497
terminal status per raw_asset_id = 1
unhandled raw assets = 0
evidence rows created by parse = 0
LLM requests created by parse = 0
embeddings created by parse = 0
all parsed outputs have raw_asset_id + checksum + source_location
pipeline_state.inventory = completed/accepted
pipeline_state.parse = completed/accepted
```

Parser rules for nontext attachments:

1. `image`: read format, dimensions, color mode, and checksum; recognize WebP by `.webp` or `RIFF…WEBP` and reuse image metadata parsing. Link captions/body references through PMCID, filename, and JATS `graphic/xlink:href`. Use OCR/visual extraction only for text/table/plot images lacking equivalent XML/HTML content. Retain image file, figure number, caption, and region coordinates, not only a model summary.
2. `media`: read format, duration, and encoding; prefer linked public captions/transcripts/supplement descriptions. Without traceable captions/transcripts, write parsed metadata and completeness only, not evidence.
3. `database`: open SQLite read-only, record schema/tables/columns/row counts, and map tables to `parsed_tables.jsonl`. Execute no embedded scripts or writes. Natural-language fields become chunks; structured fields are extracted directly.
4. `repo`: OSP `.pksim5` retains `repo` semantics but uses read-only SQLite parsing for schema, non-BLOB fields, and BLOB sizes/checksums. Do not reproduce models at this stage.
5. `document`: DOCX yields paragraphs, headings, tables, and internal-media lists. DOC prefers an auditable read-only converter; if unavailable, record `unsupported`, not unstructured `strings` output masquerading as body text.
6. `presentation`: extract PPTX text, tables, notes, and media by slide. Mark old PPT `unsupported` without a reliable parser.
7. `specialized_binary`: identify format/version. Use a read-only GraphPad Prism parser for `.pzf`; if unavailable, record `unsupported_specialized_format` without evidence or a corruption claim.
8. `unknown`: Step 1 must first try magic identification. The current formal corpus permits no remaining `unknown` records; new formats need policy/classification tests before parser entry.

### 7. Step 3: JSON/API Processing

Applies to:

```text
pubchem
rxnorm
chembl
clinicaltrials
openfda_drug_label
pkdb
Some pubmed responses
```

Steps:

1. Read JSON.
2. Verify parseability.
   - Start with a strict JSON parser.
   - Only unescaped control characters in source text permit `strict=false` tolerant parsing. Preserve raw files and record `tolerant_json_parse=true` in parsed metadata and asset warnings.
   - Other JSON errors remain `failed`.
3. Record JSON paths.
4. Write the root and source-specific records to `parsed_records.jsonl`; do not chunk the entire JSON as natural-language text.
5. Fixed fields remain parsed records here; create candidates only during `deterministic_extract`.
6. Only allowlisted free-text fields generate parsed documents/chunks.
7. Write `llm_eligible` metadata only; do not call the LLM here.

Fields not requiring an LLM:

```text
CID
RxCUI
ChEMBL ID
MolecularFormula
MolecularWeight
SMILES
InChIKey
XLogP
TPSA
NCT ID
study ID
```

Fields eligible for an LLM:

```text
clinical_pharmacology
drug_interactions
use_in_specific_populations
eligibility
outcome_description
intervention_description
study_description
protocol_text
```

### 8. Step 4: XML Processing

Applies to:

```text
dailymed SPL XML
pubmed EFetch XML
pmc_oa XML/NXML
Other NCBI responses confirmed as XML by magic bytes
```

Steps:

1. Parse XML.
2. Retain XML paths.
3. Extract document identifiers.
4. Extract titles.
5. Extract sections.
6. Extract paragraphs.
7. Extract tables.
8. Extract references.
9. Generate parsed documents per section/paragraph, parsed records for structured metadata, and parsed tables.
10. Generate only chunks and `llm_eligible` metadata; do not call the LLM.
11. PubMed search responses containing JSON must be classified as `json` in Step 1; search records retain retrieval context only, not evidence.

DailyMed must extract:

```text
set_id
title
active_ingredient
labeler
clinical_pharmacology
pharmacokinetics
absorption
distribution
metabolism
excretion
drug_interactions
renal_impairment
hepatic_impairment
pediatric
geriatric
pregnancy
lactation
```

PubMed must extract:

```text
PMID
title
abstract
journal
authors
publication_date
DOI
MeSH
PMCID
query_family
```

PMC must extract:

```text
PMCID
title
abstract
section hierarchy
paragraphs
tables
figure captions
supplement links
references
```

### 9. Step 5: PDF Processing

Applies to:

```text
pmc_oa full-text and supplementary PDFs
osp_pbpk_model_library model-evaluation PDFs
```

Steps:

1. Open the PDF.
2. Determine whether text is selectable.
3. Traverse every page of text PDFs, not only an initial subset:
   - Extract text by page.
   - Retain body text on every page; estimate table likelihood from lines, numerical density, and layout blocks, running costly table detection only on candidate pages.
   - Record page numbers.
   - Record table indices.
   - Record bounding boxes.
4. For scanned PDFs:
   - Mark `ocr_required`.
   - Do not extract evidence directly.
5. Encrypted, corrupted, or unreadable files:
   - Write to `processing_failures.jsonl`.
6. Generate page-level parsed documents.
7. Generate table-level parsed tables.
8. Do not call an LLM at this stage; only mark in chunk/table metadata whether an LLM is needed later.

PDF `source_location` format:

```text
<source_id>:<file_name>.pdf#page=<page>
<source_id>:<file_name>.pdf#page=<page>#table=<table>
```

PDF table processing:

1. Select candidate pages using deterministic page-layout/table-likelihood rules; write the results and reasons to page metadata.
2. Extract tables from candidate pages using `pdfplumber`, `PyMuPDF`, `camelot`, or an equivalent tool; retain the complete page text for noncandidate pages.
3. Convert tables to Markdown or JSON.
4. Preserve original cells, row numbers, column numbers, and page numbers.
5. At the subsequent LLM stage, identify:
   - Parameter columns
   - Units
   - Dose conditions
   - Population conditions
   - PK parameters such as Cmax/AUC/Tmax/half-life/clearance/Vd
6. Subsequent LLM output must preserve the original table-cell text.

### 10. Step 6: Table-file processing

Applies to:

```text
xpt
xlsx
csv
tsv
tsv.gz
PMC supplementary tables
OSP parameter and evaluation tables
```

Processing steps:

1. Read the workbook/table; identify OOXML workbooks by the spreadsheet content type inside the package, not by the file extension.
2. Record sheet/table names.
3. Record column names.
4. Write the entire table to parsed tables in blocks of at most 5000 rows; do not retain only sample rows. Preserve the start and end row numbers of each block. Convert date, datetime, and time cells to ISO 8601 strings before writing JSON; preserve the original types of booleans, numbers, and null values.
5. Record the row count.
6. Infer column data types.
7. Generate `parsed_tables.jsonl`.
8. This stage writes only candidate column semantics and `llm_eligible` metadata; it does not call an LLM or write evidence.
9. Determine column semantics and match entities at subsequent stages.

The LLM only determines:

```text
gene column
tissue column
expression column
dose column
unit column
compound column
population column
PK parameter column
```

The LLM does not perform:

```text
Statistical calculations
Whole-table compound assignment
Missing-value inference
Final unit conversion
```

### 11. Step 7: HTML processing

Applies to:

```text
pmc_oa full-text or supplementary-material HTML
osp_pbpk_model_library repository HTML/Markdown pages
```

Processing steps:

1. Parse HTML.
2. Remove script/style/navigation/footer elements.
3. Extract the title.
4. Extract headings.
5. Extract body paragraphs.
6. Extract tables.
7. Extract links.
8. Determine the page type:
   - landing page
   - download page
   - record page
   - documentation page
9. Landing/download pages contribute only completeness records.
10. Record-page text and tables produce only chunks/tables and `llm_eligible` metadata; this stage does not call an LLM.

### 12. Step 8: Archive processing

Processing steps:

1. Stream archive members by default; do not repeatedly extract entire archives to the system disk. Any necessary temporary files must be written to this run's SU700 staging area and deleted after processing the member.
2. Identify ZIP, TAR/GZ, RAR, or other archive types by file magic first; do not rely solely on extensions such as `.zip` or `.rar`.
3. Use Python read-only interfaces for ZIP/TAR and an auditable read-only libarchive/`bsdtar` interface for RAR. If no reliable reader is available, record `unsupported`; do not present archive contents as body text.
4. Reject absolute paths, `..` path traversal, device files, and symbolic links. Limit member size, total expanded bytes, and member count; record explicit statuses when limits are exceeded.
5. Generate a complete inner-file manifest.
6. For each inner file, record:
   - archive path
   - inner path
   - checksum
   - file family
   - compressed size
   - uncompressed size
   - parse status
7. If a member's checksum matches a standalone raw asset, record `duplicate_linked`, link to the canonical raw asset, and do not parse it again.
8. Process other inner files according to the JSON/XML/PDF/table/document/presentation/HTML/text/image/media/database rules, retaining `<archive>!<inner_path>` source locations. Continue to correct OOXML routing using the package's primary content type.
9. Expand nested archives to at most two levels, applying the same safety limits throughout.
10. Do not submit archives directly to an LLM.

`source_location` format:

```text
<source_id>:<archive_file>!<inner_path>
```

### 13. Step 9: Repository-mirror processing

Applies to:

```text
osp_pbpk_model_library
```

Processing steps:

1. Traverse repository files.
2. Record file path, extension, size, and checksum.
3. Identify README files.
4. Identify model files.
5. Identify parameter files.
6. Identify simulation/report files.
7. This stage only parses READMEs, model descriptions, reports, and `.pksim5` SQLite schemas; it does not call an LLM.
8. At subsequent LLM/deterministic extraction stages, extract:
   - model name
   - compound
   - species
   - population
   - software
   - parameter file path
   - report file path
9. Assign a compound_id only on an exact compound match.
10. Do not reproduce model execution.

### 14. Step 10: Deterministic evidence extraction

Prerequisites:

```text
pipeline_state.stages.parse.status = completed
pipeline_state.stages.parse.accepted = true
parsed_records rows = 117522
parsed_tables rows = 23546
text_chunks rows = 665316
```

This stage reads only:

```text
raw_inventory.jsonl
raw_asset_aliases.jsonl
parsed_records.jsonl
parsed_tables.jsonl
text_chunks.jsonl
parse_summary.json
pipeline_state.json
```

Execution order:

1. Merge alias compounds by `canonical_raw_asset_id` and generate `retrieval_compounds` for each parsed input. Aliases add retrieval context only; they do not duplicate scientific evidence.
2. Run a source-specific mapper on each `parsed_record`:
   - PubChem: read only explicit physicochemical fields in `PropertyTable.Properties`;
   - ChEMBL: read molecular properties only when normalized `pref_name` or a synonym exactly matches the target compound;
   - OSP model library: read only `.pksim5` files whose directory name in the repository manifest exactly matches the target compound;
   - Do not present identity, catalog, trial-context, or retrieval metadata from DailyMed, RxNorm, PubMed, PMC, PK-DB, ClinicalTrials, or OpenFDA as any of the seven evidence categories; write only terminal processing statuses or topic pointers.
3. Check each `parsed_table` for source, column semantics, and evidence signals. Write evidence directly only when an approved source-specific column mapping exists; mark other complex tables `deferred_complex_table` for subsequent selective LLM processing.
4. For each `text_chunk`, first classify machine-generated text, search lists, length, natural language, and evidence signals; then apply high-precision numeric PK rules, DDI relation rules, and topic-pointer rules. When selecting `source_text`, decimal points between digits and periods in common abbreviations such as `e.g.`/`i.e.`/`vs.`/`et al.` must not split sentences; evidence excerpts must preserve complete parameter values and necessary comparison context.
5. Numeric PK rules must validate parameter and unit types. Matches without reliable units, or clearly belonging to numeric arrays, doses, years, or identifiers, must not become evidence. Natural-language PK values must also match an explicit local relation template: `compound → PK parameter → value+unit`, `PK parameter of/for compound → value+unit`, or equivalent tightly connected `was/is/=/of` syntax. Write only a `clinical_pk` pointer if the target merely appears in the same sentence, the value belongs to another drug/metabolite/control group, other column names intervene between parameter and value, parallel parameters and values cannot be aligned individually, or a multidrug comparison leaves the value's subject unclear. Internal `.pksim5`/MoBi/PK-Sim commands, parameter-setting records, `set from/to`, `assumed`, and chart `threshold` values are not clinical observations and must not become clinical PK evidence. `creatinine clearance`/`CrCl`/`CLcr`/`CL cr`/`CL_CR`/`eGFR` describe subjects' renal function and must not map to drug clearance. Dialysis/hemodialysis clearance must not map to conventional systemic clearance. `higher/lower/later/earlier/increased/decreased/reduced by N unit` denotes a change, not an absolute PK parameter; write only pointers for these results. Record qualifiers such as `under/over/about/approximately/≤/≥` in `value_qualifier`; do not treat bounds as unconditional exact values.
6. If ADME, special-population, or PBPK keywords appear without a numeric or relational assertion, write to `evidence_pointers.jsonl`, not evidence.
7. Natural-language rule results become evidence only when the original sentence explicitly identifies a unique target compound and clearly links it locally to the parameter–value pair or DDI relation. DDI also requires an explicit affirmative, nonnegated relational assertion; `not/neither/whether or not/no evidence`, questions, `may/might/could/possibly`, and joint subjects such as `parent drug and/or metabolites` produce pointers only. If no target compound is matched, multiple targets are matched, the subject is a metabolite such as `1′-OH`/`hydroxy`/`desmethyl`/`glucuronide`, the local PK link is unclear, or DDI targets, direction, or polarity are ambiguous, write only a `clinical_pk`/`ddi` pointer containing the original text. Retain `retrieval_compounds` and `compound_assignment_status` for the subsequent LLM stage. Do not assert evidence ownership solely from the retrieval compound or same-sentence co-occurrence.
8. Write all deterministic evidence to `evidence_candidates_deterministic.jsonl` and initialize `evidence_candidates.jsonl` with identical contents. Subsequent LLM results merge only onto this baseline.

Each structured record, structured table, and chunk must receive its own terminal status in:

```text
structured_record_processing_status.jsonl
structured_table_processing_status.jsonl
chunk_processing_status.jsonl
```

Terminal statuses must include the input ID, `raw_asset_id`, `source_id`, `source_location`, mapper/classifier, evidence count, pointer count, status, and reason. A stage rerun atomically replaces only its own outputs; it must not reparse raw assets or overwrite parsed outputs. Reject reruns if downstream stages have started, unless those stages are explicitly reset first.

Deterministic evidence must include:

```text
parsed_record_id / parsed_document_id / parsed_table_id / chunk_id (as applicable to the actual upstream input)
retrieval_compounds
compound_assignment_status
source_text (original structured-field value, original table cell/row, or verbatim chunk excerpt)
extractor_version = deterministic-extract-v1
extraction_method = source_structured_field or rule_based
curation_status = needs_review
```

Stage acceptance criteria:

```text
structured record status rows = 117522, with unique parsed_record_id
structured table status rows = 23546, with unique parsed_table_id
chunk status rows = 665316, with unique chunk_id
Every evidence/pointer has nonempty raw_asset_id, checksum, source_location, and source_text
Every evidence/pointer resolves to the corresponding parsed input or structured raw-manifest field
Unique deterministic evidence_id values
New LLM request/response rows = 0
New embedding rows = 0
pipeline_state.stages.deterministic_extract = completed + accepted
llm_extract/completeness/embedding remain not_started
```

### 15. Step 11: LLM-assisted extraction

#### 15.1 High-precision routing before LLM calls

Apply the following checks in order. If any check fails, do not call the LLM, and record the reason in
`chunk_processing_status.jsonl`:

1. **Source and section eligibility**
   - DailyMed permits only core sections such as Clinical Pharmacology, Pharmacokinetics, Absorption, Distribution, Metabolism, Excretion, Drug Interactions, Use in Specific Populations, Renal/Hepatic Impairment, Pediatric/Geriatric Use, Pregnancy, and Lactation.
   - openFDA permits only free-text fields corresponding to those sections.
   - ClinicalTrials brief summaries, detailed descriptions, eligibility text, and study plans are trial metadata only; only published PK results in results/outcome-results fields may reach the LLM.
   - Exclude nonresult content such as Indications, Dosage/Dosing, Warnings, Adverse Reactions, Safety, Efficacy, Overdosage, Patient Counseling, References/Bibliography, Introduction/Background, Methods/Materials, Study Design, Objectives, and Eligibility.
   - OSP model reports may exceptionally pass model descriptions only when the original text explicitly contains PBPK, physiologically based, whole-body, PK-Sim, or Simcyp.
2. **Publication-level deduplication**
   - Read PMID, PMCID, and DOI from `parsed_documents.metadata_json`.
   - If PMC full text is available for a PMCID, do not send the corresponding PubMed abstract to the LLM; retain the abstract record with `skipped_pubmed_abstract_with_pmc_fulltext` status.
3. **Sentence-level evidence-assertion screening**
   - Extract complete sentences from the chunk without changing original characters; merge at most two adjacent sentences for results spanning sentences.
   - Candidate text must contain exactly one `target_compound` verbatim and include PK, ADME, DDI, special-population, or PBPK parameter/mechanism terms.
   - Numeric PK candidates must link a parameter to a value/unit or an explicit result such as an increase, decrease, prolongation, or shortening.
   - Qualitative ADME/DDI candidates must contain an explicit relation such as metabolized by, substrate of, inhibits, induces, transported by, or excreted unchanged.
   - Special-population candidates must contain the target compound, a population marker, a PK/ADME parameter, and a difference relation.
   - PBPK-model candidates must contain the target compound, an explicit PBPK marker, and a model-development, validation, prediction, or simulation relation.
   - Sentences describing only study objectives, planned measurements, doses, sample collection, efficacy, safety, sedation, or adverse reactions must not reach the LLM. Parameter-estimation methods such as `regression/model/analysis ... to estimate/predict AUC/Cmax` are not observed evidence either; explicit PBPK model-development statements are an exception.
   - Uncertain or predictive assertions containing `may/might/could/hypothesize/expected` must not reach the LLM. When a parameter clearly belongs to another drug, exclude it unless the original text also explicitly states the target drug's DDI role.
   - Unobserved or negative evidence such as `not studied/not expected/unknown`, truncated sentences, reference entries, and sentences containing only pharmacodynamic endpoints must not reach the LLM.
   - Qualitative metabolism/transport relations must explicitly connect the target compound to relation terms such as metabolized, substrate, inhibitor, inducer, transported, or excreted through local syntax. Do not admit another active ingredient's relation in a combination product merely because the target name is nearby.
   - Each candidate sentence must independently pass negation, completeness, and attribution checks. Do not admit a negative sentence, reference title, or truncated sentence by merging it with a subsequent result sentence.
   - If the current sentence only establishes target-drug context, merge it with the next result sentence only when that sentence does not explicitly repeat the target drug. If the next sentence already makes a complete target-drug assertion, retain that sentence alone; do not also retain an overlapping two-sentence version.
   - A local owner in constructions such as `PK/excretion/clearance/absorption ... of <another substance>` takes precedence over the retrieval target. When the target drug is only treatment, dosing, or study background, do not assign parameters of other drugs, endogenous substances, or metabolites to it. Only explicit DDI roles such as co-administration, presence of, or effect of target on permit LLM assessment.
   - `was/has been determined/evaluated/measured` states that measurement occurred, not a result. Without a numeric parameter value, directional change, or explicit mechanistic relation, exclude it from the LLM. Study-design qualifiers such as `different sites/groups/doses` must not be mistaken for PK-difference results.
   - Treat a natural-language candidate as truncated if it has unmatched parentheses, ends in a dangling comma/colon/conjunction/`almost`/`Eq.`, or lacks terminal punctuation and starts with a subordinate fragment such as `Although/suggest that/approximately`.
   - Only complete table rows with `parsed_table_id` may reach the LLM. Table fragments flattened from XML/JSON/label text without reliable row boundaries receive `deferred_unparsed_flattened_table` and await table parsing without an LLM call. Besides explicit markers such as `Table/§/‡`, identify high numeric density with repeated parenthesized ranges as flattened tables.
   - Clinical recommendations such as `should be initiated/used/administered`, dose adjustments, monitoring, and contraindications are not PK/ADME results and must not reach the LLM; other independent result sentences in the same chunk may still be retained.
   - In PBPK candidates, the target drug, PBPK marker, and model relation such as developed/validated/predicted/applies/describes must belong to the same local assertion. Do not combine a cover, URL, version information, or distant `based on` with a title to invent model evidence.
   - Predictive or speculative relations such as `predicted/probably/likely` are excluded from clinical PK/DDI extraction by default; only model predictions in explicit PBPK-model assertions may qualify for an exception.
   - `creatinine clearance/CrCl/CLcr/eGFR` are subjects' renal-function measures, not target-drug clearance. If the sentence has no other explicit drug PK/ADME result, do not send it to the LLM.
   - Treat extracted fragments beginning with `suggest that/indicate that` and lacking the preceding main clause as incomplete, even if PK parameter terms follow.
4. **Sentence-level deduplication across files**
   - Compute a stable fingerprint from `target_compound + normalized candidate sentence`; normalization is for comparison only and must not change the original text sent to the LLM.
   - For identical sentences, prefer DailyMed over openFDA and PMC full text over PubMed abstracts. Within a source, choose the canonical sentence by stable `source_location + chunk_id` ordering.
   - Mark noncanonical sentences `skipped_duplicate_llm_sentence`, recording the canonical chunk_id and sentence fingerprint.
5. **Unit of invocation**
   - A request contains only screened original sentences from one chunk for which that chunk holds canonical ownership.
   - `llm_required` counts only request candidates remaining after high-precision routing, not chunks matching broad keywords alone.
   - Before a full run, execute `classify-only` and output source distributions, skip reasons, sentence-deduplication counts, unique-request counts, and random samples.

The LLM processes only:

```text
Natural-language paragraphs
Publication abstracts
Body-text sections
Label sections
PDF page text
PDF table fragment
HTML record text
README/model description
Table column names and a small number of sample rows
```

The LLM does not process:

```text
Fixed structured fields
checksum
File-existence checks
Failed-download records
Search-result lists
Landing-page links
Complete raw PDFs
Complete large tables
Complete archives
```

LLM request format:

```json
{
  "request_id": "...",
  "run_id": "run_20260713T163840Z",
  "source_id": "pmc_oa",
  "target_compound": "midazolam",
  "raw_asset_id": "...",
  "storage_path": "...",
  "source_url": "...",
  "parsed_document_id": "...",
  "parsed_table_id": "...",
  "chunk_id": "...",
  "content_kind": "pdf_table",
  "source_location": "pmc_oa:file.pdf#page=12#table=1",
  "checksum": "...",
  "input_hash": "...",
  "input_text_or_table": "...",
  "input_items": ["A candidate assertion comprising one complete sentence or at most two adjacent sentences"]
}
```

LLM response format:

```json
{
  "request_id": "...",
  "run_id": "run_20260713T163840Z",
  "target_compound": "midazolam",
  "evidence_type": "clinical_pk",
  "parameter": "Cmax",
  "value": 53,
  "value_low": null,
  "value_high": null,
  "value_qualifier": null,
  "unit": "ng/mL",
  "condition": "10 mg oral dose",
  "source_text": "...",
  "source_text_hash": "...",
  "source_location": "pmc_oa:file.pdf#page=12#table=1",
  "confidence": 0.72,
  "requires_review": true
}
```

LLM-output validation:

1. Before invocation, verify that `inventory`, `parse`, and `deterministic_extract` are all `completed + accepted`, and that `completeness` and `embedding` have not started.
2. Every request must have a unique `target_compound`; do not call the LLM if the target cannot be uniquely resolved from the chunk or original text.
3. The LLM may extract only assertions explicitly belonging to `target_compound`. Do not assign comparator drugs, concomitant drugs, excipients, incidental mentions, or multidrug text with ambiguous relation direction to the target.
4. Every candidate must pass the Pydantic schema and return a `target_compound` identical to the request.
5. Require `source_text` and `source_location`. Accepted `source_text` must exactly equal one complete `input_items` entry in the current request and contain the target compound name. If the LLM returns an exact substring of an entry, regardless of whether it already contains the target name, the validator may expand it only when it uniquely identifies one complete entry; record `validation_repairs=expanded_source_text_to_complete_input_item`. Reject nonunique expansion. Do not truncate complete sentences to fragments such as `from the i.v.` merely because they remain locatable in the chunk.
6. `source_location` must resolve to the raw file; `checksum` must equal the raw-inventory checksum.
7. `raw_asset_id`, `parsed_document_id`, `parsed_table_id`, or `chunk_id` must resolve to upstream records.
8. Use `value` for a single value and `value_low + value_high` for a range; do not encode ranges as strings or calculate their midpoints. Every returned number must occur in `source_text` and have a unit included in the canonical unit map. Record bounds such as `less than/under/<`, `greater than/over/>`, and `up to` in `value_qualifier`. Qualitative candidates must use `value=null, value_low=null, value_high=null, unit=null, value_qualifier=null`.
9. `source_text` must form a complete assertion. Reject natural-language text ending at connectors such as `from`, `to`, `between`, `and`, `by`, `in`, `after`, `before`, `when`, or `while`, containing unmatched parentheses, or ending with an unfinished condition, comparison, or concomitant-treatment clause. Natural language without final punctuation is allowed only when it ends with a complete value/unit, an explicit PK/ADME predicate, or a complete enzyme/transporter relation. Only complete table rows identified by `parsed_table_id` are exempt from final-punctuation requirements; unparsed flattened tables must not become evidence. Precall routing and postcall validation must share the same completeness check.
10. `evidence_type + parameter` must follow plan Sections 6.2–6.7: `clinical_pk` permits only PK parameters; `special_population` only special-population PK/ADME differences; `ddi` only exposure/PK changes or enzyme/transporter mechanisms; `adme`, `physicochemical`, `tissue_expression`, and `public_pbpk_model` each use their own domain parameters.
11. Parameter names must be supported by terms in `source_text`: for example, `Cmax` requires Cmax/maximum/peak concentration, `steady_state_concentration` requires steady-state concentration, and equivalent requirements apply to AUC/Tmax/half-life/clearance/volume.
12. When the text contains a direct percentage change such as `56% reduction`, a unique bounded percentage such as `less than 0.5%`, or an explicit `2.5-fold` change, the candidate must not leave the corresponding `value` and `unit` empty. If omitted by the LLM, the validator may copy the unique number and `%`/`fold` verbatim; bounded expressions also require `value_qualifier`. Record `direct_percent_change_from_source_text`, `bounded_percent_from_source_text`, or `direct_fold_change_from_source_text`, respectively. Do not calculate ratios or treat doses or sample sizes as PK results.
13. For text containing paired values in `x vs y`, `x versus y`, `respectively`, or other direct two-value comparisons, do not automatically assign either scalar to a population/treatment group. Reject numeric candidates; qualitative candidates may retain the complete sentence for dedicated LLM review, with unresolved cases quarantined.
14. Extract expressions with change terms preceding a value, such as `increased/decreased/reduced by 60%`, using the unique original percentage just as for `56% reduction`; reject a candidate percentage that differs from this direct change value.
15. Parameter and value-unit dimensions must agree: Cmax/concentration uses concentration units, AUC uses AUC units, Tmax/half-life uses time units, clearance uses clearance units, and volume uses volume units. Percentage changes and ratios may represent relative changes in the corresponding parameters.
16. Do not record `dialysis/hemodialysis/dialyzable` clearance or `creatinine clearance/CrCl/eGFR` renal-function measures as systemic clearance of the target drug; retain only the original-text audit trail.
17. Receptor/protein staining changes in compound publications must not become `tissue_expression`. Stage 1 tissue expression comes from structured expression sources such as HPA; the LLM does not generate this type.
18. `public_pbpk_model` text must explicitly contain a PBPK marker such as PBPK, physiologically based, whole-body, PK-Sim, or Simcyp; reject ordinary population PK/PD, empirical models, and text containing only `model`.
19. Population markers for `special_population`, such as renal/hepatic impairment, pediatric, elderly, pregnancy, obesity, pharmacogenomic, or alcoholic/chronic alcohol use, must appear verbatim in `source_text`. An LLM-generated `condition` alone cannot establish this classification.
20. Sedation, respiratory depression, efficacy, safety outcomes, indications, and simple dose/administration recommendations must not be presented as PBPK evidence; record audit rejections only.
21. Do not assign parameters with joint subjects such as `target_compound and its metabolite` or `target_compound and hydroxy/desmethyl/glucuronide metabolite` to the parent drug; reject them and preserve the audit trail. The joint-subject check applies only to the sentence containing the current candidate parameter. Other independent sentences in the same continuous quotation may pass separately if they explicitly describe parent-drug parameters alone.
22. Compute text-reuse keys jointly from `target_compound + input_text_or_table`; never reuse LLM results across different compounds.
23. Write validation failures to `processing_failures.jsonl`, recording stable failure_id, request_id, candidate_index, and failure reason.
24. Write accepted candidates to `evidence_candidates_llm.jsonl`, then merge them into `evidence_candidates.jsonl` by stable evidence_id.
25. All LLM results remain `needs_review`; they must not automatically enter the human-approved layer.
26. First run a smoke batch of 10 real requests and inspect every accepted and rejected candidate. Start the full batch only after API, schema, compound attribution, cross-sentence quotations, text completeness, parent/metabolite boundaries, parameter domains, numeric qualifiers, and units pass validation. During the full run, continuously sample newly accepted candidates and scan them with deterministic completeness checks. If an accepted candidate contains truncated original text, stop immediately, archive, and fix it; do not merge the interrupted batch into official results.
27. Full batches support resumption: do not recall successful request_id values; the same input_hash for the same target compound may be reused. Continuously write request, response, and progress audit records during execution.
28. Mark `pipeline_state.stages.llm_extract` as `completed + accepted` only when every processable chunk has succeeded, been reused, or been explicitly skipped, with no unresolved failures. Request-limited smoke batches remain `in_progress`. Preserve original audit rows for old API failures that become ineligible after a routing upgrade, updating them to `resolved=true, resolution=no_longer_eligible_under_current_routing`; do not delete them or continue blind retries.
29. Accept a unified unit for mixed-time-unit ranges only when the original range is unique, explicitly belongs to the current parameter, and both endpoint conversions can be verified; record a conversion-validation flag. Do not relax numeric attribution for doses, other parameters, or between-group comparisons. Distinguish percentage-range separators from minus signs. Soft-hyphen differences may be resolved only through a unique match that restores the complete original quotation without modifying the source file. Reused responses must match both prompt and validator versions.

LLM audit files:

```text
llm_extraction_requests.jsonl
llm_extraction_responses.jsonl
llm_classification_samples.jsonl
llm_extract_progress.json
llm_extract_summary.json
```

Every LLM call must record:

```text
prompt_template_version
model_name
input_hash
output_hash
run_id
raw_asset_id
parsed_document_id
parsed_table_id
chunk_id
source_location
schema_validation_status
failure_reason
```

### 15A. Step 11A: Deferred-table recovery and evidence extraction

Inputs:

```text
Rows in chunk_processing_status.jsonl with processing_status=deferred_unparsed_flattened_table
parsed_tables.jsonl
raw_inventory.jsonl
Original raw files
```

Execution order:

1. Group deferred chunks by `raw_asset_id`; each chunk may have only one terminal status.
2. Scan `parsed_tables.jsonl`, first reusing tables already present for the same raw asset:
   - For DailyMed, align chunks and tables by `section[n]`;
   - For PDFs, align by page number;
   - For PMC, align by article/table-wrap, PMCID, and source location;
   - For other sources, align by at least `raw_asset_id + table_hash`.
3. Expand complete rows from existing tables into `recovered_table_rows.jsonl`; do not rewrite the same `parsed_table_id + row_index`.
4. Return to raw files only if the raw asset has no usable tables or existing tables do not cover the deferred location:
   - DailyMed SPL XML: read `section/text/table/tr/th/td`, expand `rowspan` and `colspan`, and preserve section titles, captions, and footnotes;
   - PMC JATS XML: read `table-wrap`, caption, label, thead/tbody/tr/th/td, and fn, expanding row and column spans;
   - HTML: read captions, thead/tbody/tfoot, rows, columns, and cells from DOM tables;
   - PDF: recover rows and columns page by page using text coordinates and table detection, retaining page numbers and bounding boxes. Scanned pages receive only `ocr_required`; do not treat OCR guesses as evidence;
   - CSV/TSV/XLS/XLSX/SQLite/PK-DB: read explicit columns and complete rows; do not treat flattened body text as tables;
   - Archives: read registered archive members only, retaining `<archive>!<inner_path>` and avoiding repeated processing of members with identical checksums;
   - OpenFDA: read `set_id`/SPL identifiers from JSON results and link to downloaded DailyMed SPL files. Keep DailyMed as the canonical table source for duplicate labels;
   - If a PubMed abstract has no table body, record `not_table_in_abstract`; do not invent rows and columns from a number-dense abstract.
5. Write each recovered table to `recovered_tables.jsonl` and each row to `recovered_table_rows.jsonl`. Preserve:

```text
run_id
source_id
raw_asset_id
parsed_table_id or recovered_table_id
table_row_id
storage_path
source_url
checksum
source_location (down to table/row/column)
caption
footnotes
columns_json
cells_json
row_text
row_hash
```

6. Deduplicate tables by `canonical raw checksum + table hash + row hash`. Retain only canonical rows for duplicate OpenFDA/DailyMed labels and files appearing both in PMC archives and standalone downloads; record aliases in status records.
7. Deterministic table extraction accepts only the following conditions:
   - An existing `parsed_table` must have the same number of columns as cells in each row, and its column names must not be footnotes or long body-text passages. Otherwise, do not reuse its column semantics; reconstruct from raw data or place it in the review queue;
   - Separate an SPL first-row `colspan` table title from the next row's actual column names. Do not use `tfoot` or footnotes as column names or data rows;
   - Do not extract evidence from repeated explanatory/footnote rows spanning the whole table through `colspan`, even within `tbody`;
   - Parameters, values, and units must map one-to-one through explicit column names or within the same complete row;
   - A numeric cell may contain only one central estimate. `Mean ± standard deviation` counts as one estimate; do not automatically split multiple group means crowded into one cell;
   - Parameter and unit dimensions must be compatible;
   - Determine the compound from the table row, title/header, the unique active ingredient in DailyMed, or the exact OSP directory name;
   - Do not assign DailyMed/OpenFDA combination-product labels solely because the target name appears in the table title. Only single-active-ingredient labels may produce deterministic evidence;
   - Victim drugs, perpetrator drugs, and parameter direction in DDI tables require human or table-specific LLM assessment; the deterministic stage does not directly produce numeric evidence;
   - `creatinine clearance`, `CL cr`, `CrCl`, and `eGFR` describe subjects' renal function, not target-drug clearance;
   - All PDF tables, including existing `parsed_table` entries and tables recovered from coordinates at this stage, may split decimals or confidence intervals across cells. The deterministic stage retains only tables and rows, without directly producing evidence. A table-specific LLM or human review of the page image is required before evidence entry;
   - Preserve comparison groups, doses, administration routes, and populations as conditions; do not mistake them for parameter values.
8. Write complete rows with unresolved column semantics, compound attribution, comparison direction, or units to `table_review_queue.jsonl`; do not directly produce evidence.
9. Only review rows with complete captions, column names, complete rows, footnotes, and reliable `source_location` may receive interpretation assistance in a separate table-LLM task. The LLM must not receive an entire file, large table, or flattened chunk, and must not invent values absent from the original table.
10. Write table evidence to `table_evidence_candidates.jsonl` and merge it into `evidence_candidates.jsonl` after deduplication; retain `machine_extracted + needs_review` for all records.
    - Compute a scientific-content fingerprint from `compound + parameter + canonical value/unit + normalized complete row + normalized condition`;
    - When the same scientific row recurs in multiple label versions or OpenFDA/DailyMed mirrors, prefer DailyMed, then choose the canonical record by stable source location;
    - Write merged records to `table_evidence_duplicates.jsonl`, retaining duplicate evidence ID, canonical evidence ID, and both complete sources. Do not delete provenance;
    - Do not merge equal values from different complete rows or conditions.
11. Write each deferred chunk to `table_recovery_status.jsonl`; permitted terminal statuses are:

```text
resolved_by_existing_table
recovered_from_raw
linked_to_canonical_source
not_table_after_source_inspection
ocr_required
unsupported_table_format
recovery_failed
```

12. Write `table_recovery_summary.json`, reporting deferred chunks, raw assets, reused tables, recovered tables, recovered rows, table-evidence counts before/after deduplication, duplicate evidence, review-queue size, and all unresolved reasons.

Acceptance criteria:

```text
Exactly one table_recovery_status per deferred chunk
recovery_failed = 0 is required for accepted status
Every recovered table/row/evidence passes provenance validation
Unique table-row IDs and table-evidence IDs
Unreliably recovered content does not enter evidence
pipeline_state.stages.table_recovery = completed + accepted
```

### 16. Step 12: Writing evidence candidates

Candidate types:

```text
physicochemical
clinical_pk
adme
ddi
special_population
tissue_expression
public_pbpk_model
```

Every evidence candidate must include:

```text
run_id
evidence_id
evidence_type
compound
source_id
raw_asset_id
parsed_document_id
parsed_record_id
parsed_table_id
chunk_id
storage_path
source_url
checksum
source_location
source_text
source_text_hash
extraction_method
extractor_name
extractor_version
extraction_confidence
curation_status
created_at
```

Numeric evidence additionally includes:

```text
parameter
value
value_text
unit
canonical_value
canonical_unit
condition
population
route
dose
```

Writing rules:

1. Fixed PubChem/ChEMBL fields may directly produce physicochemical-property candidates.
2. Natural-language results from labels, publications, PDFs, HTML, and READMEs must undergo LLM or rule-based extraction before becoming candidates.
3. Results without `source_text` must not become evidence.
4. Results without `source_location` must not become evidence.
5. Results without `raw_asset_id` must not become evidence.
6. Results without `checksum` must not become evidence.
7. Search-result lists alone must not become evidence.
8. Failure records must not become evidence.
9. For structured-field evidence, populate `source_text` with the original field-value text.
10. For table evidence, populate `source_text` with the original cell or row excerpt.
11. Both rule-based and LLM extraction must record `extractor_name` and `extractor_version`.

### 17. Step 13: completeness matrix

Write a status for every compound/source/topic.

topic:

```text
identity
physicochemical
label
literature
open_full_text
clinical_pk
adme
ddi
special_population
tissue_expression
public_pbpk_model
```

status:

```text
acquired
parsed
candidate_extracted
raw_only
missing_from_raw
failed_to_acquire
failed_to_parse
ocr_required
needs_review
```

Every status must include a reason:

```text
run_id
compound
topic
status
status_reason
source_id
raw_asset_count
parsed_count
candidate_count
review_count
failure_count
raw_asset_ids
parsed_document_ids
parsed_record_ids
parsed_table_ids
evidence_ids
review_item_ids
failure_ids
provenance_record_id
```

Execution rules:

1. The matrix must cover every `compound × source_id × topic` combination in the run. Write rows even for combinations without data; do not report only existing evidence.
2. Classify raw/parsed topics by source responsibilities and collection directories: PubChem, RxNorm, ChEMBL, and identity discovery belong to identity/physicochemical; DailyMed and OpenFDA to label; PubMed to literature and additionally the corresponding `esearch/<topic>`; PMC OA to open_full_text; PK-DB and ClinicalTrials.gov to clinical_pk; and the OSP model library to public_pbpk_model.
3. Evidence topics use the validated `evidence_type` directly. Also add their `raw_asset_id`, `parsed_document_id`, and `parsed_table_id` to the corresponding topic-coverage records.
4. Preserve `shared` coverage rows for shared OSP assets. Only model evidence explicitly assigned to a compound enters that compound's row; do not automatically assign a shared repository snapshot to every compound.
5. Status priority is fixed: `candidate_extracted` > `needs_review` > `ocr_required` > `parsed` > `failed_to_parse` > `raw_only` > `failed_to_acquire` > `missing_from_raw`. `candidate_extracted` means only that a candidate was extracted; it still retains `curation_status=needs_review`.
6. Resolved historical download failures do not count toward `failure_count`; retain `not_available_from_current_api` and unresolved acquisition failures. Candidates rejected by rules/LLMs count in reason statistics but must not be presented as acquisition or parse failures.
7. `table_review_queue.jsonl` only places the corresponding topic in `needs_review`; it does not increase `candidate_count`. PDF tables, DDI-relation tables, and tables with uncertain attribution must not receive `candidate_extracted` before review.
8. `completeness_matrix.csv` contains only statuses, counts, reasons, and `provenance_record_id`, so ordinary CSV tools can read it directly. Write all ID arrays without truncation to `completeness_provenance.jsonl`, linked one-to-one by `provenance_record_id`. The summary reports matrix rows, counts by status/topic/source, missing combinations, and missing-provenance counts.
9. Set `pipeline_state.stages.completeness = completed + accepted` only when the matrix covers all combinations, all referenced IDs resolve to original outputs, and `provenance_missing_count=0`.
10. The full-combination matrix is an audit, not a scientific knowledge-gap assessment. For example, PubChem-DDI and RxNorm-Clinical PK fall outside those sources' responsibilities, so their `missing_from_raw` status must not count toward the final Evidence Gap. Reassess final gaps according to source responsibilities and final `reviewable_evidence`.

### 18. Subsequent stages: Two types of embedding

This strategy ends before embedding. Two independent vector tasks follow:

1. **Evidence semantic-deduplication embedding**: in the final-evidence strategy, run this only on remaining natural-language evidence after compound attribution, domain gating, and the first three deterministic deduplication layers, to identify semantic duplicates.
2. **RAG retrieval embedding**: only after the final `reviewable/accepted` evidence layer exists, index the chunks, tables, and final evidence needed for Evidence Search, Q&A, and the Agent.

Do not make RAG retrieval embedding an early completion requirement for this strategy or mix the outputs of the two embedding tasks. Both record types must preserve model name, version, text hash, and complete provenance; neither vector similarity can generate scientific conclusions.

### 19. Processing priorities

P0:

```text
dailymed
pubchem
pubmed
pkdb
openfda_drug_label
```

P1:

```text
clinicaltrials
pmc_oa
chembl
rxnorm
```

P2:

```text
osp_pbpk_model_library
identity_discovery
```

Other registered sources outside this round's official collection results must not produce artificial 0-asset gaps.

### 20. Prohibited operations

Do not:

```text
Redownload raw data
Delete raw data
Overwrite raw data
Write failure records as evidence
Write search-result lists as evidence
Write landing-page links as evidence
Submit an entire PDF directly to an LLM for evidence extraction
Submit an entire large table directly to an LLM for evidence extraction
Accept LLM output without source_text
Accept LLM output without source_location
Write parsed/evidence records without raw_asset_id
Write parsed/evidence records without checksum
Write evidence that cannot be traced to original text
Assign an entire shared dataset table to a compound
Treat embedding search results directly as evidence
Mark machine-extracted results as human-approved
Claim that a public PBPK model has been reproduced
```

### 21. Completion criteria

Processing is complete only when:

1. `raw_inventory.jsonl` has 16497 rows, equal to the number of successful assets after merging and deduplicating the primary-run ledger and final OSP-snapshot ledger.
2. `processing_failures.jsonl` includes every parse/checksum/OCR/LLM-validation failure.
3. Every parsed document traces back to its raw file.
4. Every parsed table traces back to its raw file and page, sheet, row/column, or archive inner path.
5. Every text chunk traces back to a parsed document or parsed table.
6. Every evidence candidate traces back to `source_text`, `source_location`, `raw_asset_id`, and `checksum`.
7. Every LLM output has request/response audit records.
8. Every compound has a completeness matrix.
9. All machine evidence remains `needs_review`.
10. `pipeline_state.stages.completeness = completed + accepted`, and retrieval embedding has not started.
11. This run's `evidence_candidates.jsonl` is the only candidate input to the final-evidence strategy. `table_review_queue.jsonl` remains a separate review input and must not be merged directly into evidence.

### 22. Unified execution entry point

This strategy no longer requires running each stage command individually. `python -m pbpk_kb.main` calls the following fixed sequence:

```text
raw_inventory
  -> parse
  -> deterministic_extract
  -> llm_extract
  -> table_recovery
  -> completeness
```

The main function may advance only after each step writes `completed + accepted` to `pipeline_state.json`. The LLM processes only chunks marked `llm_required` that have not already succeeded; this stage runs neither evidence-deduplication embedding nor RAG embedding.


## Final Evidence Acquisition Strategy

### 0. Stage position and objective

```text
Raw-material retrieval (completed)
  -> Raw parsing and candidate evidence extraction (completed through completeness)
  -> This strategy: attribution, domain gating, deterministic deduplication, semantic deduplication, conflict analysis, evidence layers
  -> After final evidence: RAG retrieval index, database, UI, Q&A, export
```

This strategy organizes machine-extracted candidates into reviewable evidence. It does not redownload raw data, reparse files, delete candidates, or mark machine results as human-approved.

### 1. Current official inputs

The only raw-processing input directory:

```text
/Volumes/SU700/pbpk processing result/raw_processing/run_20260713T163840Z/
```

Prerequisites for execution:

```text
pipeline_state.stages.inventory = completed + accepted
pipeline_state.stages.parse = completed + accepted
pipeline_state.stages.deterministic_extract = completed + accepted
pipeline_state.stages.llm_extract = completed + accepted
pipeline_state.stages.table_recovery = completed + accepted
pipeline_state.stages.completeness = completed + accepted
pipeline_state.stages.embedding = not_started
```

Official candidate inputs:

| File | Count | Purpose |
|---|---:|---|
| `04_evidence_candidates/evidence_candidates.jsonl` | 7,330 | Sole base table of candidate evidence |
| `04_evidence_candidates/evidence_candidates_deterministic.jsonl` | 3,202 | Deterministic-source audit |
| `04_evidence_candidates/evidence_candidates_llm.jsonl` | 3,593 | LLM-source audit |
| `06_table_recovery/table_evidence_candidates.jsonl` | 535 | Table candidates already merged into the 7,330-row base table |
| `06_table_recovery/table_review_queue.jsonl` | 23,956 | Separate table-review queue, not evidence |
| `00_run_manifest/completeness_matrix.csv` | 1,212 | Raw/parsed coverage audit; combinations outside source responsibilities do not count as final knowledge gaps |

The 7,330 candidates by type:

```text
physicochemical      120
adme                 894
clinical_pk         3685
ddi                 1913
special_population   526
public_pbpk_model     192
tissue_expression       0
```

Do not use the old `run_20260629T105513Z`, old `reviewable_evidence`, or old embedding results as inputs to this run.

### 2. Outputs

```text
/Volumes/SU700/pbpk processing result/final_evidence/run_20260713T163840Z/
```

All paths in summaries and `pipeline_state.json` that point to files within this run must be run-relative; do not write absolute paths from GPU servers or other execution hosts.

Required outputs:

```text
00_run_manifest/pipeline_state.json
00_run_manifest/evidence_quality_report.json
00_run_manifest/step1_5_audit.json
00_run_manifest/provenance_validation.jsonl
00_run_manifest/provenance_summary.json
01_compound_attribution/compound_identity.jsonl
01_compound_attribution/compound_attribution.jsonl
01_compound_attribution/llm_attribution_pending.jsonl
01_compound_attribution/llm_attribution_requests.jsonl
01_compound_attribution/llm_attribution_responses.jsonl
01_compound_attribution/attribution_summary.json
02_domain_gates/domain_gate_results.jsonl
02_domain_gates/excluded_by_domain_gate.jsonl
02_domain_gates/domain_review_queue.jsonl
02_domain_gates/domain_gate_summary.json
03_deduplication/deterministic_duplicate_groups.jsonl
03_deduplication/deterministic_duplicate_members.jsonl
03_deduplication/deterministic_representatives.jsonl
03_deduplication/semantic_dedup_input.jsonl
03_deduplication/semantic_duplicate_groups.jsonl
03_deduplication/semantic_duplicate_members.jsonl
03_deduplication/semantic_conflict_candidates.jsonl
03_deduplication/semantic_dedup_state.jsonl
03_deduplication/semantic_dedup_summary.json
04_conflict_analysis/evidence_conflict_report.jsonl
04_conflict_analysis/conflict_members.jsonl
04_conflict_analysis/conflict_summary.json
05_evidence_layers/preselected_evidence.jsonl
05_evidence_layers/reviewable_evidence.jsonl
05_evidence_layers/excluded_from_reviewable.jsonl
05_evidence_layers/accepted_evidence.jsonl
05_evidence_layers/modeling_evidence.jsonl
05_evidence_layers/evidence_layers_summary.json
```

### 3. Required execution order

```text
1. Provenance validation
2. Compound identity table and evidence attribution
3. Domain-quality gating
4. Three deterministic deduplication layers: ID / source / content
5. Biomedical embedding for semantic deduplication of remaining natural-language evidence only
6. Conflict analysis on deduplicated representative evidence
7. Generate preselected evidence, automatically validated reviewable evidence, and exclusion reasons
8. Build the RAG retrieval index separately after final evidence is complete
9. Update the database, UI, Q&A, and exports
```

Do not reorder Steps 2–6. In particular, do not run embedding before compound attribution, domain gating, and deterministic deduplication.

### 4. Step 1: Provenance validation

Every candidate must include:

```text
evidence_id
run_id
source_id
raw_asset_id
storage_path
checksum
source_location
source_text
source_text_hash
extraction_method
extractor_name
extractor_version
curation_status
```

Validation rules:

1. `raw_asset_id` must exist in `raw_inventory.jsonl`.
2. `source_text` must be locatable in a structured field, parsed document, parsed table row, or the specified location in the original file.
3. `source_text_hash` must match the original text.
4. Structured, rule-based, and LLM candidates must all preserve their actual extractor.
5. Write records missing provenance to `excluded_from_reviewable`; they must not proceed to later steps.
6. Keep the candidate base table immutable. Write corrections to derived records; do not overwrite original records.

### 5. Step 2: Compound attribution

#### 5.1 Identity table

The identity table must include at least:

```text
preferred_name
PubChem CID / synonyms / InChIKey
RxNorm names / RxCUI
ChEMBL ID and exact molecule identity
UNII
DailyMed / FDA active ingredient names
Salt-form, prodrug, metabolite, and stereoisomer relationships
```

Related forms record relationships only; do not automatically merge them into the parent compound.

#### 5.2 Rule-based attribution

For every candidate, use:

```text
source_text
source title
section title
source metadata
retrieval compound
source_id
evidence_type
parameter
```

Permitted roles are:

```text
target_compound
victim_drug
perpetrator_drug
co_medication
comparator
excipient
mentioned_only
shared_or_unlinked
```

Execution order:

1. First match synonyms, stable IDs, and active ingredients exactly.
2. Then assess the source and section, such as Clinical Pharmacology, Drug Interactions, or Specific Populations.
3. In `A increased B AUC`, A is usually the perpetrator and B the victim.
4. In `B was inhibited by A`, B is the victim; A may be the perpetrator.
5. For probe/cocktail statements, accept only local drug–enzyme/transporter pairs. Do not assign multidrug, multitarget statements that cannot be locally aligned.
6. comparator, excipient, mentioned_only, and unexplained co-medication roles do not enter the final evidence layer.
7. PubChem structured properties may be assigned under exact retrieval identity; ChEMBL search results additionally require proof of exact molecule identity.
8. Assign OSP models only when filenames, model metadata, or reports explicitly match the target compound. Shared repository snapshots cannot be assigned to all compounds.
9. If DailyMed/OpenFDA table rows or sections do not repeat the drug name, use document-level attribution only when active-ingredient fields establish that the document concerns a single target-drug product. Combination products must not automatically inherit the retrieval compound.
10. `AUC`, `Cmax`, `exposure`, or `clearance` in an ordinary PK sentence does not establish DDI. Assess victim/perpetrator roles only when the evidence is already typed as DDI or the text combines a multidrug relation with an interaction action.
11. `target is a probe/sensitive substrate`, `effect of A on target PK`, and `target AUC/Cmax increased or decreased by A` support high-confidence victim attribution; `target inhibits/induces enzyme or another drug` supports high-confidence perpetrator attribution.

#### 5.3 LLM attribution review

Call the LLM only for:

```text
Two or more drugs in the same source_text
DDI victim/perpetrator roles unresolved by rules
Multidrug relations such as coadministered / concomitant / versus / compared with
Probe/cocktail statements with multiple drugs and targets that cannot be locally paired
Inconsistent section and source_text attribution
Unproven ChEMBL exact identity
Rule confidence < 0.80
Long paragraphs containing multiple doses, populations, drugs, or study groups
```

The LLM may output only:

```text
evidence_id
matched_text
compound_name
compound_role
role_reason
confidence
```

`matched_text` must appear verbatim in `source_text`, and `compound_name` must match the identity table. The LLM must not rewrite the original text, generate new evidence, or change `curation_status`.

If `source_text` contains no drug name or synonym from the identity table and no single-active-ingredient document identity exists, the LLM must not guess attribution; retain the record as `shared_or_unlinked`.

### 6. Step 3: Domain-quality gating

Run domain gating before any embedding. Preserve complete provenance and `exclusion_reason` for excluded records.

#### 6.1 Physicochemical

Allow only pure-compound structures and physicochemical properties:

```text
MolecularFormula, MolecularWeight, InChIKey, CanonicalSMILES,
XLogP/LogP/LogD, TPSA, pKa, solubility, ionization,
fraction_unbound, protein_binding, hydrogen-bond donor/acceptor count
```

Exclude dose, formulation, product strength, food/plant/sample content, and product descriptions.

#### 6.2 ADME

Allow only the target compound's own absorption, distribution, metabolism, excretion, bioavailability, permeability, protein binding, enzyme/transporter mechanisms, and metabolite formation.

Exclude pharmacodynamics, receptor/cellular signaling, behavioral endpoints, plant synthesis, and statements about the target drug affecting another entity's ADME rather than its own.

#### 6.3 Clinical PK

Allow only explicit PK parameters or exposure conclusions, such as AUC, Cmax, Tmax, half-life, clearance, volume of distribution, bioavailability, fraction excreted, and concentration.

Dose, route, and formulation are condition fields only; efficacy, safety, physiological, or exercise endpoints must not become Clinical PK evidence.

#### 6.4 DDI

Victim, perpetrator/interacting drug, effect/mechanism, and direction must be identifiable. Probe-substrate identity, cocktail co-occurrence, combination-label co-occurrence, and multidrug sentences with unclear direction do not establish DDI.

Quarantine tables and text with unresolved direction for review by a dedicated relation-extraction LLM; if still unresolved, exclude them from semantic-deduplication input.

Domain-gating statuses are fixed as `passed`, `needs_manual_review`, and `excluded`. Store `needs_manual_review` in `domain_review_queue.jsonl`; never misrecord it as a passed gate.

#### 6.5 Special Population

Require an explicit population/condition, such as pregnancy, lactation, pediatric, geriatric, renal/hepatic impairment, genotype, or sex difference. Animal studies must explicitly record species and remain separate from human PK.

#### 6.6 Public PBPK Model

Categories:

```text
downloadable_public_model
published_pbpk_model
model_mention_only
```

Only the first two categories may enter reviewable evidence. OSP models must have paths to model, parameter, script, or evaluation-report files. Exclude ordinary population PK/PD and text mentioning only "model."

#### 6.7 Tissue Expression

The current input has 0 records; report this gap accurately. Do not invent tissue-expression evidence from ordinary enzyme/transporter mentions.

### 7. Step 4: Deterministic deduplication

Deduplication creates groups without deleting candidates. Apply these layers in order:

1. Duplicate `evidence_id`.
2. Source duplicates: `raw_asset_id + source_location + source_text_hash`.
3. Content duplicates: `compound + evidence_type + parameter + canonical_value/value + canonical_unit/unit + condition + source_text_hash`.

Retain every group member and choose one deterministic representative. Representative priorities:

```text
Complete provenance
High-confidence compound attribution
Complete value/unit/condition
Complete canonical value/unit
Higher-priority source
More precise source_location
Higher extraction_confidence
```

Group the same SPL in DailyMed and OpenFDA, files downloaded both in PMC packages and separately, and repeatedly chunked copies of the same paragraph at this step.

For `source_structured_table`, `parsed_table_id`, or recovered-table records, treat records as the same table fact when compound, evidence_type, parameter, canonical value/value, canonical unit/unit, and `source_text_hash` match. Different condition formatting or product-name spelling across labels must not prevent deterministic deduplication. Preserve every member's original condition for audit.

After all three layers, write the remaining final representatives to `deterministic_representatives.jsonl`. Separately write natural-language records with valid attribution roles that passed domain gating to `semantic_dedup_input.jsonl`, the only input to Step 5.

### 8. Step 5: Evidence semantic-deduplication embedding

#### 8.1 Input scope

Process only records satisfying all of the following:

```text
Provenance passed
Compound attribution resolved
Domain gating passed
Representative of a deterministic duplicate group
Natural-language source_text
```

Do not embed structured ID/numeric fields, duplicate table cells, excluded records, or the table-review queue.

#### 8.2 Model and input

```text
model = NeuML/pubmedbert-base-embeddings
runtime = sentence-transformers
dimension = 768
```

Construct input text by stably concatenating:

```text
compound
evidence_type
parameter
value_text
condition
population
source_text
```

Compare only within the same compound and evidence_type; if parameters are explicit, further restrict comparisons to the same parameter group.

#### 8.3 Decisions

```text
similarity >= 0.92             duplicate candidate
0.85 <= similarity < 0.92      similar_for_review
similarity < 0.85               no grouping
```

Every pair of evidence records within an automatic `semantic_duplicate` group must have cosine similarity of at least `0.92`. If members reach `0.92` only against a central record but any within-group pair falls below `0.92`, do not merge automatically; downgrade the entire group to `similar_for_review`.

Do not merge highly similar records with conflicting values, units, doses, routes, populations, or conditions; send them directly to conflict analysis. Embedding identifies similar text only; it cannot generate, correct, or adjudicate scientific facts.

`semantic_duplicate_groups.jsonl` must distinguish `semantic_duplicate`, `similar_for_review`, and `semantic_conflict_candidate`. Every row in `semantic_duplicate_members.jsonl` must also retain `group_type`; additionally write conflict candidates to `semantic_conflict_candidates.jsonl`.

Do not use `hashing-embedding-v0` as a fallback for final evidence deduplication. If the GPU or model is unavailable, record an explicit blocker; do not mark the step complete.

### 9. Step 6: Conflict analysis

Inputs are fixed as `deterministic_representatives.jsonl`, semantic-deduplication group/member files, and domain-gating results. First remove nonrepresentative `semantic_duplicate` members; do not automatically remove `similar_for_review` or `semantic_conflict_candidate` members.

Conflict-candidate key:

```text
compound
evidence_type
parameter
canonical_unit
route
population
condition
```

Compare value, unit, dose, route, population, condition, label version, and study design.

Execution rules:

1. Every `semantic_conflict_candidate` must generate a conflict-analysis record. All members must appear in `conflict_members.jsonl`, each mapping to exactly one semantic-conflict analysis group.
2. For representative evidence not already in semantic-conflict groups, scan for conflicting structured values or directional conclusions using the conflict key above.
3. Mark `explainable_variation` only when separately structured route, population, dose, time point, study design, or label version has explicit, nonempty differences within the group. Tags inferred from conditions—fed/fasted, renal/hepatic function, pregnancy/lactation, age, sex, smoking, or genotype—are review hints only. Differences in free-text condition alone require `insufficient_context`.
4. With identical or missing conditions, nonoverlapping canonical intervals, inconsistent values in the same units, or conflicting increase/decrease/no-effect directions receive `unresolved_conflict`.
5. Incomparable units, partially missing conditions, or text containing multiple directions receive `insufficient_context` and automatic exclusion from default reviewable evidence.
6. Overlapping intervals, values agreeing within 1% relative tolerance, or matching directions receive `compatible`. Compatible groups originating from `semantic_conflict_candidate` still require human confirmation of whether they are duplicate statements; do not admit them directly to reviewable evidence.

If different doses, administration routes, populations, time points, study designs, or label versions explain the variation, retain separate evidence for different conditions without marking a true conflict. If conditions are identical or missing and conclusions/values disagree, write a conflict group and automatically exclude it from the default final evidence layer.

Conflict analysis must not average values, select the value that "looks reasonable," or let the LLM determine truth.

### 10. Step 7: Evidence layers and automatic validation

| Layer | Definition | Default use |
|---|---|---|
| `candidate` | 7,330 immutable machine candidates | Audit, debugging |
| `excluded` | Failed provenance, attribution, or domain gating, with reasons retained | Quality audit |
| `preselected` | Deduplicated representative evidence with resolved attribution and passed domain gating | Conflict and automatic-quality validation |
| `reviewable` | Preselected evidence without unresolved attribution, table, similarity, or conflict blockers | Default UI, search, Q&A, Agent |
| `accepted` | Optional human-override layer; not required by automation | Compatibility with legacy curation interfaces |
| `modeling` | Optional human-override layer; not required by automation | Compatibility with legacy modeling-approval interfaces |

File-layer generation rules:

1. `preselected` = deterministic representatives that passed domain gating, minus nonrepresentative `semantic_duplicate` members. Retain conflict and similarity-review members with their blockers.
2. `reviewable` = `preselected` records without `similar_for_review`, semantic-conflict `compatible` duplicate review, `unresolved_conflict`, `insufficient_context`, or other unresolved blockers.
3. `excluded_from_reviewable` and `reviewable` must completely partition the 7,330 candidates by `evidence_id`; preserve `exclusion_reasons` for every excluded record.
4. The automatic pipeline uses `reviewable` as its blocker-free final evidence layer. `accepted_evidence.jsonl` and `modeling_evidence.jsonl` support optional human overrides only and remain empty when there are no curation events.

Legacy `curation_status=needs_review` indicates only the absence of a human event; it no longer blocks the automated knowledge-base and recommendation workflows. Automatic usability is jointly determined by evidence layer, provenance, domain gating, deduplication, and conflict status.

`table_review_queue.jsonl` remains a separate quarantine queue. New candidates produced by table-specific LLM reparsing must reenter this pipeline at Step 1, not append directly to reviewable evidence; records that cannot be resolved automatically remain excluded.

### 11. Step 8: Subsequent RAG retrieval index

Run only after `reviewable_evidence.jsonl` is complete.

The retrieval index uses a high-relevance allowlist, not every chunk and table from raw processing. Generate inputs in the following order:

```text
reviewable_evidence.source_text
accepted_evidence.source_text
text_chunks.chunk_text exactly referenced by reviewable/accepted evidence
Context with adjacent chunk_index +/-1 in the same parsed document and source-location prefix as the referenced chunk
Reliable parsed/recovered table_text exactly referenced by reviewable/accepted evidence
```

Implementation rules:

1. Ordinary raw chunks, ordinary tables, and the table-review queue that are not referenced by final evidence do not enter the Stage 1 default RAG index.
2. Adjacent context must share `parsed_document_id`, the `source_location` prefix after removing `#chunk=`, and adjacent `chunk_index`, preventing contamination across articles or label sections.
3. Link tables exactly by `parsed_table_id`/`recovered_table_id`. Use `parsed_tables.table_text` directly. For structurally reliable recovered tables without `parsed_table_id`, deterministically reconstruct table text from saved captions, `columns_json`, and `recovered_table_rows.row_text` ordered by `row_index`. If rows are missing or structure is unreliable, retain only the corresponding evidence quotation; do not invent table content.
4. Deduplicate retrieval records by `(compound_id, source_location, normalized_text_hash)`; merge record types and linked evidence IDs without losing provenance.
5. Each retrieval record stores `retrieval_record_id`, `compound_id`, `source_id`, `source_location`, `source_text`, `source_text_hash`, `evidence_layer`, linked evidence/chunk/table IDs, and complete provenance.
6. `source_text` always preserves the original text. Store vector input separately as `embedding_text`, deterministically concatenating compound name, evidence type, parameter name, value/unit, and original text. For structured evidence containing only short values such as `194.19`, do not embed an isolated number or use an LLM to rewrite the source.
7. Use `NeuML/pubmedbert-base-embeddings` to generate 768-dimensional normalized vectors. If the GPU/model is unavailable, this step remains failed or incomplete; do not fall back to hash embedding.
8. Output files are fixed as `06_rag_index/retrieval_records.jsonl`, `retrieval_embeddings.npy`, and `rag_index_summary.json`; record rows and vector rows correspond one-to-one through `embedding_row`.

Retrieval embedding and Step 5 semantic-deduplication embedding are independent tasks:

- Deduplication embedding compares evidence only and outputs duplicate/similar groups.
- RAG embedding serves Evidence Search, Q&A, and the Agent, producing a retrieval index.
- Keep separate directories, stages, and summaries; using the same model does not permit shared statuses or outputs.
- Retrieval hits must return `source_id`, `source_location`, `source_text`, and evidence layer.

### 12. Step 9: Database, UI, Q&A, and exports

1. Import only this run's 7,330 candidates and their exactly linked raw/parsed documents. Do not load all ordinary raw-processing documents unreferenced by final evidence/RAG into the Stage 1 operational database.
2. Import candidate, excluded, preselected, reviewable, accepted, and modeling layers through `evidence_layer_membership`. Database views use this run's file-defined layers; do not rederive reviewable evidence with a separate SQL rule set.
3. Import deterministic/semantic duplicate groups and conflict groups/members separately, retaining representatives, statuses, reasons, similarities, and automatic-exclusion reasons.
4. RAG uses a separate `rag_retrieval_record` table with a `vector(768)` field, importing `retrieval_records.jsonl` and `retrieval_embeddings.npy`. Do not write to the old 1536-dimensional hash `document_chunk.embedding`.
5. Compound Profile, Evidence Search, Source Registry, Q&A, KG, exports, and the recommendation Agent
   read only the `auto_validated_evidence` database service layer corresponding to
   `07_llm_quality_validation/final_auto_validated_evidence.jsonl`; reviewable, repair_pending, quarantined,
   duplicate-member, superseded, and unresolved evidence is excluded from default retrieval.
6. Optional human actions must still write `curation_event`, but the unified automatic pipeline does not wait for them.
7. Q&A and the Agent must display evidence layer, automatic-validation results, sources, and uncertainty; do not describe reviewable evidence as human-confirmed.
8. Automatic exports default to auto_validated; every export preserves evidence layer, quality status, and provenance.
9. Calculate Evidence Gap from source responsibilities and final auto_validated coverage; do not directly use all
   `missing_from_raw` rows in raw completeness.

### 13. Acceptance criteria

```text
input_candidate_count = 7330
missing_provenance_count = 0 (records entering preselected/reviewable)
unresolved_compound_attribution_count = 0 (records entering preselected/reviewable)
domain_gate_result_count = 7330
every duplicate member resolves to one duplicate group
semantic dedup uses NeuML/pubmedbert-base-embeddings
every conflict member resolves to one conflict group
every excluded candidate has exclusion_reason
every reviewable evidence has source_text and source_location
every accepted/modeling evidence has curation_event
reviewable_count + excluded_from_reviewable_count = 7330
semantic duplicate non-representatives are absent from preselected/reviewable
similar_for_review and unresolved/insufficient conflict members are absent from reviewable
reviewable contains no repeated structured-table fact with the same compound/type/parameter/value/unit/source_text_hash
table review items are not silently promoted
RAG retrieval indexing starts only after reviewable is complete
```

### 14. Prohibited operations

- Do not mix old runs or old final-evidence results into this run.
- Do not run evidence embedding before attribution, domain gating, and deterministic deduplication.
- Do not treat RAG retrieval embedding as semantic-deduplication results.
- Do not delete candidates to resolve duplicates or conflicts.
- Do not present an LLM summary as original text.
- Do not treat `needs_review` as human confirmation.
- Do not treat comparator, excipient, or mentioned_only records as target evidence.
- Do not treat dose, formulation, or food/sample content as pure-compound physicochemical properties.
- Do not treat pharmacodynamic, behavioral, or safety endpoints as ADME/Clinical PK.
- Do not treat probe/cocktail co-occurrence as DDI.
- Do not automatically merge conflicting values or conditions.
- Do not write the table-review queue directly into final evidence.
- Do not export evidence without provenance.

### 15. Unified execution entry point

After raw-material processing passes completeness acceptance, `python -m pbpk_kb.main` calls this fixed sequence:

```text
provenance_attribution
  -> domain_and_deterministic_dedup
  -> semantic_dedup
  -> conflict_and_evidence_layers
  -> rag_index
  -> deterministic_quality_control
  -> llm_quality_validation
  -> database_import
```

`semantic_dedup` reads only `semantic_dedup_input.jsonl` generated by deterministic deduplication; `rag_index`
may generate candidate retrieval records only after `reviewable_evidence.jsonl` is complete. `database_import` must wait
for `final_auto_validated_evidence.jsonl` and the final quality summary before filtering and writing the service layer.
The unified manifest and each stage's `pipeline_state.json` both record statuses. After failure, rerunning the main function resumes only
unaccepted stages without repeatedly overwriting accepted outputs.


## Automatic Evidence Quality-Control Strategy

### 1. Objective

Classify machine candidates in `reviewable_evidence.jsonl` into:

```text
auto_validated: eligible for the database, KG, retrieval, and recommendation calculations
repair_pending: eligible for revalidation after rereading original text, tables, or adjacent context
quarantined: retained for audit but excluded from KG and recommendation calculations
```

Human review is not a mandatory step. Do not infer values, units, drugs, comparison groups, or experimental conditions absent from the original source.

### 2. Inputs and outputs

Inputs:

```text
05_evidence_layers/reviewable_evidence.jsonl
04_conflict_analysis/evidence_conflict_report.jsonl
04_conflict_analysis/conflict_members.jsonl
Original files, parsed documents, parsed tables, adjacent text chunks
```

Outputs:

```text
06_auto_quality_control/quality_decisions.jsonl
06_auto_quality_control/auto_validated_evidence.jsonl
06_auto_quality_control/repair_pending_evidence.jsonl
06_auto_quality_control/quarantined_evidence.jsonl
06_auto_quality_control/context_repair_requests.jsonl
06_auto_quality_control/kg_ready_nodes.jsonl
06_auto_quality_control/kg_ready_edges.jsonl
06_auto_quality_control/quality_summary.json
07_llm_quality_validation/final_auto_validated_evidence.jsonl
07_llm_quality_validation/final_quarantined_evidence.jsonl
07_llm_quality_validation/final_quality_decisions.jsonl
07_llm_quality_validation/quality_summary.json
```

### 3. Fixed execution order

#### Step 1: Source and original-text validation

Every evidence record must include:

```text
evidence_id
compound_id
source_id
raw_asset_id
storage_path
checksum
source_location
source_text
source_text_hash
extraction_method
extractor_version
```

Recompute `source_text_hash`. Immediately mark a record `quarantined` if its original text is empty, its hash mismatches, its file is missing, or its source locator is absent.

#### Step 2: Field-type and domain validation

Check evidence type, compound role, parameter, numeric types, and domain boundaries.

```text
physicochemical: structures or physicochemical properties
adme: absorption, distribution, metabolism, excretion, enzyme and transporter mechanisms
clinical_pk: AUC, Cmax, Tmax, half-life, clearance, volume of distribution, etc.
ddi: exposure changes or explicit mechanisms involving two drugs
special_population: PK/ADME differences in specific populations
tissue_expression: tissue expression of enzymes or transporters
public_pbpk_model: explicit PBPK-model information
```

Comparator drugs, excipients, incidental drug mentions, and unassigned records must not pass.

#### Step 3: Missing values and units

1. Qualitative ADME, mechanistic relations, and model descriptions may use `value=null` and `unit=null`.
2. Numeric quantitative parameters require units, except dimensionless ratios/fold changes.
3. Accept only a unique unit immediately adjacent to the parameter value in `source_text`.
4. Use `unit_normalizer.py` to generate `canonical_value` and `canonical_unit`.
5. Unit dimensions must match the parameter: for example, concentration for Cmax, concentration multiplied by time for AUC, and time for half-life.
6. If the original text does not uniquely determine the unit, generate a `context_repair_request`; do not guess.
7. Generate a repair request if the text claims a quantitative result but no value can be extracted.

#### Step 4: Context and reference validation

For quantitative `clinical_pk` records, check dose, route, population, and condition. Not every missing field must block acceptance, but record missing fields in `context_missing_fields`.

The following records require a reference or baseline:

```text
increase/decrease/change/difference
ratio/fold/percent change
DDI exposure changes
Changes in special populations relative to healthy populations
```

References may come from structured fields, condition, source_text, or traceable adjacent context. If unresolved, mark `repair_pending`.

DDI requires explicit:

```text
victim drug
Perpetrator drug or an explicit enzyme/transporter mechanism
Effect direction or quantitative impact
```

Records containing only ordinary PK values for the target drug itself must not become DDI evidence.

#### Step 5: Deterministic repairs

Perform only these demonstrable repairs:

```text
Recover a unique unit from the same original text
Normalize units and values
Recover route, dose, population, and reference from explicit fields
Combine headers, footnotes, and cell context from the same table row
Recover complete truncated sentences from adjacent chunks
```

For each repair, save:

```text
field
old_value
new_value
supporting_text
supporting_location
repair_method
```

#### Step 6: Exact LLM extraction

Call the LLM only for `repair_pending`. Restrict inputs to the original evidence, its source paragraph, table headers/footnotes, and adjacent chunks.

The LLM may return only source-supported fields and verbatim quotations, without external knowledge. Returned values must pass Steps 1–4 again.

#### Step 7: Independent LLM validation

Use a separate request to reread the original text and repair results, returning:

```text
supported
unsupported_fields
compound_role_correct
parameter_value_unit_correct
reference_correct
context_sufficient
verbatim_support
confidence
```

Extraction and validation requests must not share prior explanatory text. If validation fails, allow at most one additional reread-and-repair attempt; mark continued failures `quarantined`.

#### Step 8: Conflict validation

Group by compound, parameter, canonical unit, dose, route, population, condition, study design, and time point.

```text
Different conditions: retain as context-specific evidence
Identical conditions and compatible values: retain source-independent supporting relations
Identical conditions and conflicting values or directions: quarantine all; do not average or automatically choose a true value
Missing conditions prevent comparison: mark repair_pending
```

#### Step 9: Automatic admission

Mark `auto_validated` only if all of the following hold:

```text
Complete source and verbatim original text
Resolved compound attribution and role
Correct domain classification
No type conflicts among parameter, value, and unit
Interpretable units for quantitative evidence
Explicit reference for relative changes
Explicit roles of both DDI parties or mechanism
No unresolved conflicts
No blocking missing fields
Passed rule validation
Passed independent LLM validation when the LLM participated in repair
```

### 4. KG output constraints

The KG reads only `auto_validated_evidence.jsonl`.

Nodes:

```text
Compound, Parameter, Enzyme, Transporter, Population, Tissue,
PublicModel, StudyContext, SourceDocument, Evidence
```

Edges:

```text
HAS_EVIDENCE, HAS_PARAMETER, SUBSTRATE_OF, INHIBITS, INDUCES,
TRANSPORTED_BY, HAS_DDI_WITH, STUDIED_IN, EXPRESSED_IN,
DESCRIBED_BY_MODEL, SUPPORTED_BY, CONFLICTS_WITH
```

Every factual edge must carry:

```text
evidence_id
source_id
source_location
source_text_hash
quality_status=auto_validated
validation_method
confidence
```

`repair_pending` and `quarantined` may appear only as audit nodes; they must not produce scientific-fact edges.

### 5. Final database and UI service layer

Database import must follow completed LLM repair and independent validation, reading only:

```text
07_llm_quality_validation/final_auto_validated_evidence.jsonl
```

Layering rules:

```text
final_auto_validated -> auto_validated_evidence database view
final_quarantined -> files and audit records only
reviewable/repair_pending -> process audit only
candidate/preselected/excluded -> process audit only
```

Compound Profile, Evidence Search, Source Registry, Q&A, KG, exports, and recommendations read only
`auto_validated_evidence`. `reviewable_evidence` must not serve as the final service layer.

The importer must:

1. Verify that final-file row counts equal the final `quality_summary.json` field
   `final_auto_validated_count`.
2. Validate every record's evidence ID, compound attribution, source, original text, checksum, and provenance.
3. Overwrite outdated candidate-table fields with final repaired fields, setting status to `auto_validated`.
4. Rebuild physicochemical, ADME, Clinical PK, DDI, and special-population service tables only for finally accepted evidence.
5. Import only RAG records linked to finally accepted evidence or its traceable chunks.
6. UI, API, KG, Q&A, export, and recommendation queries must not return quarantined or process-layer records.
