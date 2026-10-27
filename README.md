# PBPK knowledge base and model recommendation

Evidence retrieval and provenance, component recommendation, public-model recommendation, and executable PBPK blueprints.

**Streamlit Community Cloud presentation.** The native English interface is in [`cloud/app.py`](cloud/app.py), with a portable, verified research snapshot and CPU-based blueprint prediction. It does not require the server database or external API keys. See [Cloud deployment](docs/deployment.md#streamlit-community-cloud). Model and component recommendations are clearly identified as saved research results; new GPU inference remains in the private server workspace.

**The private workspace uses the current method (2026-10-19).** PBPK-Match light recommendation, 30-drug blueprint design and prediction, development-curve calibration, prediction charts, and downloads are available on the server. The knowledge base contains 100 evidence-covered drugs, 3,266 currently usable evidence records (checked on 2026-10-26), and 280 model versions. The server binds to loopback and uses personal SSH forwarding and authentication. [Startup and use](docs/deployment.md).

**Blueprint core: source- and purpose-driven automatic PBPK construction.** Existing experiments support traceable construction and evidence benefits for some drugs. They do not establish a general accuracy advantage, stability without CL, or reliable clinical risk prediction. Results without a drug-specific curve are reported separately from curve-calibrated results. [Strategy and limits](docs/recommendation_strategy.md#blueprint-design).

**Current workflow: human physiology and drug evidence → calibration of unknown parameters → feasibility checks and necessary repair → prediction under target conditions.** Qualified drug evidence, necessary inputs such as fu, and their sources are retained. Main-text descriptions and figures present this workflow without RF labels. Actual RF input provenance, ablations, and CL-predictor comparisons are reported near Appendix Table E.6. Existing numerical results are unchanged; presentation edits do not trigger reruns. [Parameter handling and reproducibility](docs/recommendation_strategy.md#blueprint-rf-role).

**Latest ten-drug structural improvement: calibration of blood-cell partitioning without direct measurements.** Full/five-point curve GMFE decreased from 1.4884/1.5055 to 1.3773/1.4099. This is retained as a candidate for the development-curve branch. Gains concentrate in three drugs and half-life bias remains substantial. Within twofold refers to ten-drug curve GMFE, not every endpoint. [Results and limitations](experiments/blueprint_design/blood_partition_20261008/README.md).

**Ten-drug sampling-point evaluation completed (2026-10-09).** A simple utility score was learned from 140 existing additional-point results, holding out ten drugs in nine study groups, without new PBPK fits. In the missing-early-point task, Cmax GMFE was 1.3116 versus 1.3712 for the time rule, a mean improvement of 4.35%. Three drugs improved, five tied, and two worsened; the paired interval crossed 1 and the predefined majority-benefit criterion was not met. This negative conclusion is retained. The selector was not further tuned on these ten drugs; the subsequent user-requested 30-drug extension preserved the original predictor and manuscript. [Results and comparison](experiments/blueprint_design/learned_sampling_20261009/README.md) · [Strategy](docs/recommendation_strategy.md#blueprint-design) · [Commands](docs/development.md#offline-blueprint-design).

**Existing manuscript results remain frozen.** The ten-drug development-curve branch and 20-drug branch without a drug-specific curve are reported separately. Their scores are not pooled. Detailed historical exploration notes were removed from the introduction; manuscript data, clinical sources, and required reproduction assets remain.

**30-drug repair evaluation completed.** Training and evaluation definitions were aligned, study-held-out consistency was checked, and an unmeasured blood-partition lower bound was repaired where mass balance demonstrated infeasibility. All-30-drug Cmax/curve GMFE was 1.4417/1.5472 versus 1.7619/1.7071 previously. Most gains came from the Montelukast construction repair. Relative to the earlier method, six drugs improved, eight worsened, and 16 tied; a majority benefit is not claimed. This experimental batch did not overwrite the manuscript. [Results, causes, and comparison](experiments/blueprint_design/sampling_stability_20261009/README.md).

- [Manuscript PDF](results/manuscript/pbpk_paper_zh.pdf) · [LaTeX source](results/manuscript/main.tex) · [Publication outputs](results/README.md)
- [Experiment index](experiments/README.md) · [Research plan](docs/research_plan.md) · [Model recommendation strategy](docs/recommendation_strategy.md)
- [Frozen ten-drug results](experiments/blueprint_design/vascular_tissue_ten_drug_20260930/results.json) · [Current frozen research version](results/data/blueprint_selected.json)

Some research links refer to server-retained results excluded from GitHub. The Cloud application reads only its committed portable snapshot.

![Evidence-grounded PBPK component recommendation, model matching, and blueprint design](results/figures/workflow/01_four_panel_en_preview.png)

English previews: [Four panels](results/figures/workflow/01_four_panel_en_preview.png) · [Four lanes](results/figures/workflow/02_four_lanes_en_preview.png). Comic Neue and illustrated scientific elements explain retrieval fusion, graph soft tokens, typed matching, and mass conservation. Panel C in the lane layout illustrates model-use outcomes; panel D presents executable blueprints and validation reports.

<details>
<summary>English four-lane figure caption</summary>

**Evidence-grounded PBPK recommendation and blueprint design.**
(a) Literature, drug labels, PK studies and public model sources are normalized into source-linked evidence and a versioned model catalog. An evidence subgraph G_i is represented by e_i=f(text(G_i)); its normalized embedding is e-hat_i. The same frozen text encoder f embeds the modeling query q in (b).
(b) Query q retrieves subgraphs G_i using normalized query/subgraph embeddings and fused lexical/dense rankings. Query-conditioned HGT states h_i are pooled into h_0. The shared projector P_phi maps h_0 and h_i into t_0 and t_i, which form T at the green token row; cited text enters the frozen LLM separately. Here norm is L2 normalization and w is the learned attention scoring vector. Attention pooling and the shared projector are trained using alignment followed by token cross-entropy, while the LLM remains frozen. Recommended components and their evidence inform the model requirement graph; matching chain symbols identify this handoff across the central bridge.
(c) G_q and G_M denote the requirement and model-version graphs. The paired Organ nodes illustrate a requirement node v and a model node u. The schematic matrix highlights their entry s_vu; gray dashes exclude cross-type comparisons. In the compact set expression, u ranges over the nodes of G_M, so U_v contains only model nodes with type(u)=type(v). These entries feed the deterministic typed-alignment subscore, which sums over eligible requirement nodes v, excluding root, Objective and Regimen. s_vu is label-token Jaccard similarity, or 1 for an identical semantic ID. Fixed type weights w_v are used. Including zero in the maximum gives m_v=0 for an empty U_v; the score is defined as 0 if there are no eligible requirement nodes. The implemented applicability categories are Reuse, Adapt, Reference and Reject. Applicability constraints take priority over alignment scores; relevance is compared within each tier. Run labels the fine curved link from model reuse to prediction under a dosing scenario. The model-use panel illustrates predicted concentration profiles and pharmacokinetic outputs (AUC and Cmax); these are schematic examples, not measured results or evidence of validation.
(d) Each source E_j supplies a parameter value theta_j, with units, context and provenance retained in theta. The illustrated transfer J_(j→i) runs from kidney j to blood i and contributes to the aggregate flux equations. A_i is compartment drug amount and V_i is its fixed volume; A(t) denotes the full state vector. J denotes intercompartment transfer rates; R_in and R_elim denote external input and local elimination rates, all in amount per unit time. SolveODE denotes solving for the state vector A(t); dividing each amount by volume gives concentration. AUC requires a subsequent time integral. Independent validation is designed to use held-out studies with no calibration overlap. Output illustrations are schematic, not reported validation results.

Heavier solid arrows emphasize the main workflow; fine solid connections show internal relations and physiological flows. Dashed paths denote contextual inputs or node alignment; equations are placed inside or beside their corresponding mechanisms. Curves clarify branching and merging. Line width indicates visual hierarchy, not quantitative flux or confidence.

</details>

[Four-panel design prompt](results/figures/workflow/01_four_panel_en_prompt.txt) · [Four-lane design prompt](results/figures/workflow/02_four_lanes_en_prompt.txt) · [Four-panel renderer](scripts/render_english_panels.py) · [Four-lane renderer](scripts/render_english_lanes.py).

Earlier editable artwork remains available: [PowerPoint, three slides](results/figures/workflow/PBPK_workflow_editable.pptx). The SVG and PowerPoint correspond to the historical source-language artwork; English mechanism previews are linked above.

| Layout | Preview | Vector |
|---|---|---|
| Four-panel overview | [PNG](results/figures/workflow/01_four_panel.png) | [SVG](results/figures/workflow/01_four_panel.svg) |
| Four-lane workflow | [PNG](results/figures/workflow/02_four_lanes.png) | [SVG](results/figures/workflow/02_four_lanes.svg) |

Colors follow the supplied research-figure reference. Image generation was used to explore design, and final figures were reconstructed from vector elements. [Design reference](results/figures/workflow/design_reference.png) · [Generation prompt](results/figures/workflow/imagegen_prompt.txt) · [Editable renderer](scripts/render_editable_workflow.py).

| Resource | Current size |
|---|---:|
| Usable evidence | 3,266 records covering 100 drugs; checked 2026-10-26, further recovery in progress |
| Public model catalog | 202 families, 280 versions |
| Catalog-ranking data | 772 requirement scenarios, 216,160 compatibility-relevance labels |

**Evidence review and recovery in progress (2026-10-26).** The original 30 drugs have 2,267 usable records; 37 duplicates were merged. Of 614 initially quarantined records, source completion restored 39 and contextual review restored seven; 568 remain quarantined. The additional 70 drugs have 999 usable records. Batch 1 increased from 137 to 422, including 45 restored by cell-level verification against original tables; provenance, keyword, and semantic retrieval checks passed. Six truncated requests in batch 2 were recovered and downstream processing is underway. The seven batches are not all complete. Original records, models, and frozen experiments remain; automated review is not expert validation. [Table repair and retrieval checks](runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014/table_cell_repair_20261026/run_20261009T140919Z/retrieval_verification.json) · [Coverage comparison](runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014/coverage_audit_20261026.json). Monitor with `runtime/app-env/bin/python -B scripts/deployment/project_python.py -B scripts/deployment/repair_expansion_quality.py watch`.

**Reasons for the smaller new collection:** extraction prompts missed records, table headers were excluded from numeric attribution checks, and the remaining six batches had not completed repair. Of the same 1,419 inputs, repair changed 558 from empty extraction to nonempty candidates, still requiring individual verification. The earlier description of 1,369 deferred drug–article items as undownloaded was inaccurate: file-level checking found 1,060 with parsed full text, 304 with abstracts only, and five with metadata only. The initial 100-article limit cannot establish the final download gap. [Content and parsing audit](runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014/pmc_content_audit_20261026.json).

The remaining six batches reuse validated extraction and table repair, then undergo source checking, deduplication, and quality validation before import. Background follow-up runs every minute and records an audit every 20 minutes. Only confirmed generation truncation permits one recovery pass per batch, followed by continuation on success; other errors stop the run. `watch` also reports the supervisor state. Failure counters are not reset and existing source files are not downloaded again.

The earlier evidence batch completed all 18 steps. It added 1,144 records for 20 previously uncovered drugs while preserving 1,728 existing records. Model expansion added 231 versions. Those versions support retrieval and reference but have not all passed scientific execution validation. The historical 102 identity registrations do not mean 102 evidence-covered drugs. [Evidence import](data_process/final_evidence/run_20260909T083120Z/00_run_manifest/database_import_summary.json) · [Model expansion checks](experiments/model_recommendation/model_expansion_20260909/final_verification.json).

**Model recommendation quality assessment completed (2026-10-01).** The 40,600 primary judgments for 145 core requests, 2,029 repeat judgments, and statistics for 12 methods are complete. On the judged subset of 40 requests with known positives, REARANK/BM25 NDCG@10 was 0.787/0.769, with the difference interval crossing zero. F2LLM applicability class-Y F1 was 0.519; repeat-judgment coverage and reliability remain limited. Table 7 and a concise analysis were added to the manuscript. [Statistics](results/tables/model_recommendation_quality.csv) · [Results and scope](experiments/model_recommendation/results_summary/README.md) · [Manuscript](results/manuscript/pbpk_paper_zh.pdf).

**PBPK-Match (2026-09-27):** the MLP version is the current main model; typed matching and fusion are structural controls. Implementation, small-scale GPU training, ablations, optimization, and three-seed checking are complete, with 183 checks passed. Only ten development requests with known positives support ranking evaluation: MLP ensemble NDCG@10 was 0.807 versus 0.747 for fusion. Applicability recognition remains weak and model selection needs independent validation. Quality labels for 102 locked test requests were not used for development. [Design, results, inference, and limitations](experiments/model_recommendation/pbpk_match/README.md).

**Current development policy:** external-method analysis is complete; no new in-house training or tuning was performed in that assessment. The MLP development results use an earlier, different label snapshot and cannot be compared directly with the later 40-request ranking subset. Further optimization requires development-set diagnosis.

**LambdaMART's perfect score was audited:** the metric calculation was correct, but labels and input features encode the same matching rules. A shallow tree using only four coverage features reached the same perfect score. New experiments do not treat these labels as scientific answers. Real PK results have been extracted; independent relevance and applicability assessment required separate completion. [Audit](experiments/model_recommendation/runs/expanded_catalog_20260913/lambdamart_audit.json).

Real PBPK execution and blueprint functionality remain. The project-local OSPSuite environment ran 14 Moxifloxacin scenarios and passed engineering checks; independent scientific validation remains incomplete. [Simulation and blueprint results](experiments/README.md).

The legacy server Streamlit service remains disabled. Ranking, evaluation, and simulation do not require continuous LLM calls. Difficult evidence processing and component generation use the registered server Qwen endpoint only when needed.

| Directory | Contents |
|---|---|
| `cloud/` | Portable English Streamlit application and verified research snapshot |
| `src/` | Application and algorithm implementation |
| `docs/` | Core workflows, strategy, development, and deployment |
| `results/` | Manuscript figures, final result tables, and supporting data; retained on the server |
| `experiments/` | Current results, required inputs, and real simulation outputs; retained on the server |
| `data_process/` | Final evidence, model catalog, and batch verification |
| `artifacts/` | Original sources, public model code, and runtime assets |
| `scripts/` | Deployment, processing, and experiment entry points |
| `tests/` | Necessary behavior and scientific-rule checks |
| `runtime/` | Isolated environments, weights, databases, and logs |
| `dependencies/` | Required third-party assets |
