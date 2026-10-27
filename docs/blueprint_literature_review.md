# Blueprint design: a review of parameter inference, model discovery, and prediction methods

Review date: 2026-10-11. Materials came from the user-provided `papers(1).docx` and were considered alongside the project's current strategy and code. This document contains literature analysis and research suggestions; it does not modify the current algorithm, frozen experiments, or manuscript.

**Main assessment: the most useful insight is to frame blueprint research as "which errors cannot be explained by adjusting existing parameters, and which minimal mechanism changes are supported by observations?"** Posterior inference can reveal parameter compensation; knowledge-constrained equation search can study mechanism changes; neural operators mainly accelerate computation. These address different problems, and connecting all three does not by itself establish methodological novelty.

## 1. Reading scope and citation corrections

The document contains 15 deduplicated paper links and additionally names GraphCast and DeepONet, for 17 papers total. Core methods, experiments, and discussion were obtained and read in the full texts or complete author manuscripts of 16 papers. For paper 14, Cohen, only public publisher excerpts were obtained, not the full text. **Finding an abstract was not counted as reading a full paper, and no paper's reproduction experiments were run.** Supplementary-material reading scope is stated for each entry.

| No. | Short title | Actual research problem | Materials read |
|---|---|---|---|
| 1 | Gonçalves, 2020 | Neural-dynamics parameter posteriors | Full text, appendices, and key figures |
| 2 | Boelts / MNLE, 2022 | Neural likelihoods for mixed observations | Full text, appendices, and key figures |
| 3 | OutbreakFlow, 2021 | Epidemic ODE parameter posteriors and prediction | Full text; separate supplementary PDF unavailable |
| 4 | Jang, 2026 | SBI identifiability and reliability comparison | Full text; separate supplementary DOCX unavailable |
| 5 | JUNO, 2026 | Simulator calibration using neural likelihoods | Published 18-page PDF; methods and repeated experiments reviewed |
| 6 | Pangu-Weather, 2023 | Forward prediction of weather fields | Main text, Methods, author manuscript, and code documentation |
| 7 | GraphCast, 2023 | Graph-network weather prediction | Complete author manuscript and key supplements; journal webpage returned 403 |
| 8 | Aurora, 2025 | Pretraining on heterogeneous Earth-system data | Main text, Methods, and relevant sections of the 62-page supplement |
| 9 | FNO, 2021 | Function-to-function forward surrogates | Complete 16-page author manuscript, including inverse-problem appendix |
| 10 | DeepONet, 2021 | Branch–trunk operator learning | Complete published typeset text, not the early abbreviated preprint |
| 11 | Rabeh, 2025 | Evaluating flow models for complex geometries | Main text, Methods, Tables 1–3 |
| 12 | Forster, 2024 | Discovering rate equations from trajectories | Published 12-page text and supplementary DOCX |
| 13 | Rogers, 2024 | Knowledge-constrained symbolic regression and experimental design | 33-page author manuscript including Appendix S1 |
| 14 | Cohen, 2024 | PFR reaction-rate discovery | **Publisher excerpts; full text unavailable** |
| 15 | Dyer, 2024 | Parameter posteriors for economic agent-based models | Published 36-page text including appendices |
| 16 | Liu, 2019 | Surrogate-assisted financial-model calibration | Publisher full text, appendices, and relevant tables |
| 17 | McGreivy & Hakim, 2024 | Evaluation bias in scientific machine learning | Complete author manuscript, focusing on main text and methods |

Two points in the supplied document need correction:

- `s44172-025-00513-3`, labeled as a nuclear-system digital twin, is actually paper 11 on flow-model evaluation. The nuclear-energy paper closest to that description appears separately in Section 4; an incorrect link cannot substantiate a conclusion.
- "Train once and never fit again" applies only to some amortized posterior methods. MNLE, JUNO, and Liu still require MCMC, nested sampling, or numerical optimization for new observations; FNO, DeepONet, and weather models primarily learn forward mappings.

## 2. What the current project actually does

The authoritative strategy is [Recommendation strategy, Section 8](recommendation_strategy.md#blueprint-design). Main manuscript results are bound to the `time_repaired` branch of `sampling_stability_20261009`; see the [publication-data binding](../results/data/blueprint_sampling_publication.json).

Inputs are basic human conditions, drug evidence, dosing conditions, requested endpoints, and the small set of same-drug development concentration observations permitted in this branch. The current method constructs a 15-organ vascular–tissue two-region model with reversible exchange. Native artifacts supply initial volumes, flows, partition fields, and related inputs; the project's linear mass-conserving ODE then generates trajectories. **This does not call the full native PK-Sim solver for every complete trajectory.**

For each partition formula, four multipliers—clearance, tissue partition, exchange, and blood-cell partition—are calibrated, and the formula is selected using visible development data. Qualified net renal input remains fixed; total clearance is calibrated when no supported decomposition exists. RF fills qualified CL and fu gaps. Current capacity diagnostics make bounded repairs when observations conflict with partition limits; there is no general mechanism-structure discovery yet.

The core is therefore **parameter calibration under a fixed structure with restricted formula selection**. "4+1" is an observation-budget protocol, not the model's underlying principle or an independent algorithmic innovation. The 30 drugs have already undergone repeated development, and prior sources for the original ten and additional twenty drugs must not be assumed identical. Plasma fitting alone cannot establish that true organ mechanisms have been identified.

Implementations: [human exchange model](../src/pbpk_kb/blueprint_policy/permeability_model.py), [blood-cell calibration](../src/pbpk_kb/blueprint_policy/blood_partition_calibration.py), [optimizer](../src/pbpk_kb/blueprint_policy/distribution_calibration_optimizer.py), [capacity checks](../src/pbpk_kb/blueprint_policy/visible_partition_feasibility.py), and [current experiment entry](../scripts/run_sampling_stability.py).

## 3. Paper-by-paper review

The methods and results below summarize the papers. Implications for this project are our analysis, not PBPK conclusions already validated by the original authors.

### 1) Gonçalves et al.: represent multiple solutions with parameter distributions

*Training deep neural density estimators to identify mechanistic models of neural dynamics*. **eLife, 2020, 9:e56261**.[Article](https://elifesciences.org/articles/56261); [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7581433/); [Official code](https://github.com/mackelab/IdentifyMechanisticModels_2020).

- **Method:** sample parameters from a prior, generate observations with a simulator, and train the conditional density `q(θ|x)`. Single-round training can be reused for new observations; sequential SNPE refines around current observations and requires proposal-distribution correction. The joint parameter posterior is returned to the simulator for prediction.
- **Experiments:** Figure 3 fits 350 ion-channel models with the same eight-parameter model, using one million simulations and approximately 1000/150 CPU hours for simulation/training. Inference on new observations takes under 10 milliseconds, and over 90% of response correlations exceed 0.98. The 350 tasks are mainly simulated; additional real-neuron and crab neural-circuit data are included.
- **Key evidence:** Figures 5–6 show that very different parameters can produce similar dynamics. Figure 4 shows that different additional summaries constrain different parameters. Small error does not imply unique parameters; misspecification and missed posterior modes remain risks.
- **Implication:** retain the joint compensation among our four multipliers and examine whether equivalent fits yield different half-lives. Using SNPE directly applies existing technology; the deeper question is which parameter combinations observations actually constrain. The old code depends on Theano/delfi; new development should evaluate the current `sbi` implementation.

### 2) Boelts et al.: learning a likelihood does not directly return parameters

*Flexible and efficient simulation-based inference for models of decision-making*. **eLife, 2022, 11:e77220**.[Article](https://elifesciences.org/articles/77220); [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9374439/); [Official code](https://github.com/mackelab/mnle-for-ddms).

- **Method:** MNLE factorizes the joint likelihood of discrete choices and continuous reaction times into classification probabilities and conditional densities, using a classifier and spline flow. New data still require MCMC combining the learned likelihood with a prior.
- **Experiments:** the four-parameter DDM has an analytical-likelihood reference; the five-parameter version uses posterior prediction and SBC. In Figures 2–4, MNLE uses 100,000 simulations and achieves mean C2ST 0.65 across 100 datasets of 100 trials; corresponding LAN comparisons are 0.96, 0.78, and 0.70, with ideal value 0.5. These are simulated validations.
- **Limitations:** many trials can accumulate small likelihood errors; systematic bias appears at 1000 trials. Flexibility to change priors remains limited to the covered parameter domain. Independent-trial assumptions cannot be transferred directly to correlated concentration time series.
- **Implication:** with an explicit concentration-error model, PBPK likelihoods may already be computable and should not be relearned merely to use a neural network. Useful lessons are separating prior from likelihood and checking whether repeated observations amplify approximation error. Code is AGPL-3.0; reproduction dependency versions need separate verification.

### 3) OutbreakFlow: reusable simulation training still depends on model applicability

*OutbreakFlow: Model-based Bayesian inference of disease outbreak dynamics with invertible neural networks and its application to the COVID-19 pandemics in Germany*. **PLOS Computational Biology, 2021, 17:e1009472**.[Article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009472); [Official code](https://github.com/stefanradev93/AIAgainstCorona).

- **Method:** epidemic ODEs, interventions, reporting noise, and delays generate time series; convolution, LSTM, and invertible networks infer parameter posteriors, followed by ODE predictions.
- **Experiments:** a five-parameter SIR model uses early real German case data, with another ten thousand simulated scenarios for SBC. The full 34-parameter model uses observations from March 1 to May 21 and forecasts the next three weeks. The national three-channel model cannot directly accept state data lacking recovery records; the authors retrain a two-channel network for 16 states.
- **Limitations:** the undetected fraction has a 95% interval of [0.07,0.91], and some parameters remain close to their priors. The authors distinguish model-to-reality discrepancy, finite-simulation error, and amortization error. SBC mainly checks inference reliability within simulator assumptions.
- **Implication:** simulated trajectories can train an inference model, but one million simulations of one structure neither establish correct human mechanisms nor represent one million drugs. Priors remain necessary without same-drug curves; having no existing model differs from having no drug information. The repository is marked as a legacy implementation, with BayesFlow recommended by the authors.

### 4) Jang et al.: more simulations cannot fix nonidentifiability in the observations

*A comparative study of simulation-based inference methods for epidemic models with identifiability considerations*. **PLOS Computational Biology, 2026, 22:e1014364**.[Article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014364); [Official code](https://github.com/geunsoojang/EpiSBI).

- **Method:** compare ABC, NPE, NPE-LSTM, and PNPE. PNPE spends half its budget on ABC, fits a proposal distribution, then generates simulations to train a posterior network; it depends on the current observations and is not fully amortized.
- **Experiments:** SEIR, Ebola, and SIRTEM use simulation budgets of 1000, 10,000, and 100,000, varying initial conditions, observation channels, and noise. Figure 3 shows that larger budgets cannot recover missing information. In real 1918 influenza experiments, nominal 95% predictive intervals for both PNPE and NPE-LSTM cover 86.67%; ABC covers 100% with wider intervals.
- **Limitations:** dynamics are mainly deterministic; model misspecification and missing data are not systematically validated. Narrower intervals do not automatically mean greater reliability.
- **Implication:** PBPK must jointly report error, interval coverage and width, parameter compensation, and computational cost. Neural posteriors can be baselines, but posterior concentration should not itself be an optimization target. Porting sequential-inference code requires checking proposal corrections and the historical version corresponding to the paper.

### 5) JUNO: combining experimental conditions can resolve parameter confounding

Gavrikov et al., *Simulation-based inference for precision neutrino physics through neural Monte Carlo tuning*. **Communications Physics, 2026, 9:63**.[Article](https://www.nature.com/articles/s42005-026-02499-6); [Published PDF](https://www.nature.com/articles/s42005-026-02499-6.pdf); [Official code](https://github.com/ArseniiGav/neuromct).

- **Method:** the TEDE Transformer histogram-density model and NFDE continuous-flow model learn photoelectron distributions for different radioactive sources and three simulation parameters. Likelihoods from five sources are combined before nested sampling; a single network does not directly predict the three parameters.
- **Experiments:** roughly a billion simulated events and a training grid of 9261 parameter combinations, plus an offset grid to test interpolation. Repeated simulations examine estimator bias and error versus exposure; each of five exposure levels has 1000 repetitions. With 10,000 events per source, relative precision for the three parameters is approximately 2.1%, 4.9%, and 0.2%.
- **Limitations:** validation mainly concerns calibration within one simulator. Public code does not mean all data are public; simulation materials require collaborator permission. NFDE does not outperform TEDE on every distribution metric.
- **Implication:** jointly constraining one drug's parameters across multiple development dosing conditions is more likely to resolve confounding than adding optimization starts to one curve. Target studies must remain held out; fitting observations from target conditions jointly and then calling the result predictive validation is invalid.

### 6) Pangu-Weather: innovation in structured representation and time stepping

Bi et al., *Accurate medium-range global weather forecasting with 3D neural networks*. **Nature, 2023, 619:533–538**.[Article](https://www.nature.com/articles/s41586-023-06185-3); [Official code and weights](https://github.com/198808xc/Pangu-Weather).

- **Method:** maps current weather fields to future fields; a 3D Earth-specific Transformer explicitly handles pressure levels and geographic-position differences. Separate 1-, 3-, 6-, and 24-hour models combine time steps to reduce recursive error.
- **Experiments:** 39 years of ERA5 training data, with 2018 testing and 2019 validation. Reported five-day Z500 RMSE is 296.7 versus 333.7 for operational IFS; each model trains on approximately 192 V100 GPUs for 16 days.
- **Limitations:** this is forward prediction, not a parameter posterior. It relies mainly on reanalysis data; extremes may be smoothed and multiple time-step models may be temporally inconsistent. The public repository supplies weights, inference, pseudocode, and lightweight training guidance, not a fully public complete training pipeline.
- **Implication:** PBPK learning should reflect organ identity, blood-flow direction, nonnegative drug amounts, and conservation. The current linear ODE already provides precise numerical propagation, so relearning four time-step solvers after weather networks may offer little value. Appropriate structural constraints for unknown mechanisms or observation inversion are more relevant.

### 7) GraphCast: data splits and fair baselines matter equally

Lam et al., *Learning skillful medium-range global weather forecasting*. **Science, 2023, 382:1416–1421**.[Journal page](https://www.science.org/doi/10.1126/science.adi2336); [Author manuscript and supplement](https://arxiv.org/html/2212.12794v2); [Official code entry](https://github.com/google-deepmind/graphcast).

- **Method:** takes two states six hours apart, encodes a multiscale spherical graph with message passing, predicts the next-state residual, and rolls forward for ten days. Training gradually extends from one step to 12 steps.
- **Experiments:** the formal main comparison trains on 1979–2017 and evaluates 2018; 90.3% of 1380 verification targets outperform HRES. Training takes about four weeks on 32 TPU devices; a ten-day forecast takes under one minute on one TPU v4.
- **Useful practice:** the authors control data-assimilation windows and compare both methods after smoothing, avoiding advantages caused by unequal conditions. Long leads still face smoothing and extreme-event limitations, and initial fields depend on traditional assimilation.
- **Implication:** a fixed organ graph plus GNN alone is not our contribution; additional representation must demonstrably improve unknown-mechanism inference. Comparisons must control visible concentrations, drug priors, and target conditions. Random time-point splits within one drug do not prove new-drug generalization. The original repository now redirects to WeatherNext, so pin a historical GraphCast implementation.

### 8) Aurora: shared structure and individual differences need supporting data

Bodnar et al., *A foundation model for the Earth system*. **Nature, 2025, 641:1180–1187**.[Article](https://www.nature.com/articles/s41586-025-09005-y); [Supplementary materials](https://api.repository.cam.ac.uk/server/api/core/bitstreams/66e515f4-f522-4b5c-a7c4-bf1b5ae2622f/content); [Official code](https://github.com/microsoft/aurora).

- **Method:** encodes different variables, pressure levels, and resolutions into a shared representation, predicts with a 3D Transformer, and decodes; pretraining precedes short-step and long-rollout adaptation.
- **Experiments:** 1.3 billion parameters and over one million hours of heterogeneous data; pretraining takes about 2.5 weeks on 32 A100 GPUs. Compared with training from scratch, average gains are approximately 54%, 22%, and 25% for pollution, waves, and 0.1° weather tasks; validation also uses observations from more than thirteen thousand weather stations.
- **Limitations:** inexpensive adaptation rests on expensive pretraining. Supplement I.7 shows that LoRA long-rollout adaptation may increase smoothing and suppress extremes; lower aggregate error does not ensure every important endpoint improves.
- **Implication:** represent irregular sampling, dosing history, observed tissues, and missingness explicitly. Shared physiology and drug differences can be learned separately, but 30 drugs cannot support a foundation-model-scale analogy; simulation counts are not real drug diversity.

### 9) FNO: faster forward prediction does not automatically solve parameter inversion

Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*. **ICLR, 2021**.[Original paper page](https://openreview.net/forum?id=c8P9NQVtmnO); [Complete author manuscript](https://arxiv.org/pdf/2010.08895); [Official maintained implementation](https://github.com/neuraloperator/neuraloperator).

- **Method:** parameterizes integral kernels in Fourier space to map coefficient or initial fields to solution functions. Tests cover Burgers, Darcy, and Navier–Stokes equations, generally with 1000 training instances and 10,000 for harder settings.
- **Experimental detail:** in one NS setting, FNO-3D relative error drops from 0.1918 to 0.0820 as samples increase from 1000→10000. The inverse-problem appendix uses 7×7 noisy observations and MCMC; FNO replaces forward calls only. Thirty thousand calls take about 2.5 minutes versus 18 hours for the baseline, but offline simulation and training require about 12 hours.
- **Limitations:** resolution transfer is not generalization across unknown mechanisms; complex models may not win with fewer samples. Similar posterior means do not establish equivalent posterior tails.
- **Implication:** measure whether repeated-call savings offset offline cost before choosing a surrogate. If our low-dimensional linear ODE is already fast, FNO will not resolve compensating multipliers or missing mechanisms.

### 10) DeepONet: dosing functions and query times can be represented separately

Lu et al., *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*. **Nature Machine Intelligence, 2021, 3:218–229**.[Journal page](https://www.nature.com/articles/s42256-021-00302-5); [Published typeset PDF](https://aiichironakano.github.io/cs653/Lu-DeepONet-NMachineIntell21.pdf); [Official code](https://github.com/lululxvi/deeponet).

- **Method:** the branch encodes input-function values at sensor positions, the trunk encodes query positions, and their dot product gives the output. The original branch generally requires fixed sensor positions, so arbitrary irregular inputs are not supported by default.
- **Experiments:** the published version includes 16 operator tasks. In the Figure 4 reaction–diffusion example, 100 input functions each have 1000 output points; test MSE is approximately 10⁻⁵ versus 10⁻¹ for ResNet. This is not merely 100 scalar measurements; training and test sets do not share solution functions.
- **Limitations:** universal approximation does not guarantee finite-data accuracy. Figure 2 shows substantial error increases after input-distribution changes; training on simulation labels does not automatically fix real-world mechanism discrepancy.
- **Implication:** separating dosing history from arbitrary query time is natural, but current linear PBPK already supports matrix exponentials and input convolution. Mapping observations to parameters/mechanisms addresses current difficulties more directly than relearning known propagation.

### 11) Rabeh et al.: low field error does not imply better physics

*Benchmarking scientific machine-learning approaches for flow prediction around complex geometries*. **Communications Engineering, 2025, 4:182**.[Article](https://www.nature.com/articles/s44172-025-00513-3); [Official code](https://github.com/baskargroup/GeometryMatters); [Data](https://huggingface.co/datasets/BGLab/FlowBench/tree/main/LDC_NS_2D/512x512).

- **Method:** compares 11 SciML models predicting velocity and pressure fields from geometry and Reynolds number. There are 300 geometries × 10 conditions, totaling 3000 simulations, with a random 2400/600 split and additional data-size and Reynolds-extrapolation experiments.
- **Evaluation:** reports global error, near-boundary error, and momentum-equation residuals separately. Rankings change across metrics, input representations, and extrapolation conditions; models strong under random splits can deteriorate substantially.
- **Review finding:** the main text's generalization about DeepONet's physical-metric advantage does not cover all tables; other models score higher in Table 1 SDF extrapolation and Table 2 mask extrapolation. Table 3 timing covers the 600-sample test set and is not per-instance latency.
- **Implication:** PBPK should separate prediction error, conservation, parameter/interval reliability, and transfer across conditions. Numerical stability in unmeasured organs cannot replace real exposure validation. This link does not match the supplied document's nuclear-energy digital-twin description.

### 12) Forster et al.: discovered formulas can contain incorrect dependencies

*Machine learning uncovers analytical kinetic models of bioprocesses*. **Chemical Engineering Science, 2024, 300:120606**.[DOI](https://doi.org/10.1016/j.ces.2024.120606); [Published full text](https://dau.url.edu/bitstream/handle/20.500.14342/4620/Machine%20learning%20uncovers%20analytical%20kinetic%20models%20of%20bioprocesses.pdf?sequence=1); [Supplementary materials](https://ars.els-cdn.com/content/image/1-s2.0-S0009250924009060-mmc1.docx).

- **Method:** smooth and differentiate each state time series, use Bayesian Machine Scientist to search explicit state-to-rate expressions, then substitute them back into the ODE for validation. Expression selection balances error and complexity.
- **Experiments:** two synthetic systems, each with 13 batches: 10 training and 3 held out by initial conditions; each batch has 20 or 40 points. Table 2 reports BMS/ANN test R² of 0.999/0.986 in one setting and 0.900/0.990 in another, so symbolic methods do not win universally.
- **Key limitation:** Equation 18 and Figures 8–9 introduce product dependence absent from the true equation. The method is restricted to measurable species; hidden states cannot be discovered directly this way. The supplementary five-point differentiation experiment uses a sine function, not evidence for complex PBPK.
- **Implication:** retain human balance equations and investigate only unknown rate forms, but do not present simulator-generated organ concentrations as measurements and claim organ-mechanism discovery. The [insidapy](https://github.com/forstertim/insidapy) and [udiff](https://github.com/forstertim/udiff) components were confirmed; a one-command repository reproducing all experiments was not.

### 13) Rogers et al.: the most relevant structural-discovery paper

*Integrating knowledge-guided symbolic regression and model-based design of experiments to automate process flow diagram development*. **Chemical Engineering Science, 2024, 300:120580**.[DOI](https://doi.org/10.1016/j.ces.2024.120580); [33-page author manuscript with appendix](https://arxiv.org/pdf/2405.04592).

- **Method:** estimate rates from observable states and known balances; PySR searches expressions, embeds candidates in mass and energy equations, and designs experiments from candidate disagreement. Equation 2 requires the prescribed forward/reverse reaction structure, assigning infinite penalty otherwise; this is not post hoc explanation of a black box.
- **Key experiment:** in Appendix S1, correlated temperature and concentration with insufficient shear variation yield an incorrect initial rate expression. New experiments break the correlation and recover the correct rate. Case1B also loses the true expression through a preference for simplicity, recovering it only after additional discriminating experiments.
- **Results and limitations:** Case1B test KPI MAPE falls from 3.90% to 0.30% by round four; unconstrained Case1C improves from 25.6% to 1.98% and plateaus. Both are synthetic systems with known grammars compatible with true mechanisms and measurements richer than sparse plasma data. [PySR](https://github.com/astroautomata/PySR) is used; a complete paper-reproduction repository was not found.
- **Implication:** the key lesson is which experiment distinguishes competing mechanisms. PBPK can investigate minimal, mass-conserving mechanism edits, but one additional sample may not separate clearance from partitioning; changing development dose/infusion conditions may be more informative.

### 14) Cohen et al.: known transport can reduce inverse-problem difficulty

*Data-driven Discovery of Reaction Kinetic Models in Dynamic Plug Flow Reactors using Symbolic Regression*. **Computer Aided Chemical Engineering, 2024, 53:2947–2952**, ESCAPE/PSE conference chapter.[Original page](https://www.sciencedirect.com/science/article/pii/B9780443288241504920); [DOI](https://doi.org/10.1016/B978-0-443-28824-1.50492-0).

**The complete six-page paper was not obtained.** Publisher, laboratory, and institutional open records were checked. The following derives only from public introduction, method, and result excerpts; no unverified figure numbers or success rates are provided.

The confirmed approach uses known ideal-PFR transport equations and a characteristic transformation to convert outlet observations into data suitable for reaction-source discovery, followed by symbolic regression, reducing integration per candidate. Public excerpts list nonisothermal irreversible reactions, Hougen–Watson kinetics, and noisy cases, each repeated 30 times. Official code was not confirmed.

**Implication:** use known physics to improve the relationship between observations and unknowns before learning the unknown component. PBPK has no equally simple outlet-to-hidden-state transformation; the method of characteristics cannot be transferred directly to whole-body organ models with plasma-only observations. This paper is not currently a primary basis for definite algorithm or performance conclusions.

### 15) Dyer et al.: predictive fit and posterior correctness need separate checks

*Black-box Bayesian inference for agent-based models*. **Journal of Economic Dynamics and Control, 2024, 161:104827**.[DOI](https://doi.org/10.1016/j.jedc.2024.104827); [Oxford published full text](https://ora.ox.ac.uk/objects/uuid%3A957f169d-f93f-41e5-9ed0-ed68b1ecd0f5/files/r3t945r97c); [Official code](https://github.com/joelnmdyer/sbi4abm).

- **Method:** NPE learns conditional parameter densities; NRE estimates density ratios by classifying matched and mismatched pairs, then samples. Temporal/graph encoders learn summaries. Sequential variants concentrate simulations near a particular observation and must not simultaneously be called fully amortized.
- **Experiments:** in the Brock–Hommes case, learned-summary SNRE uses 10,000 simulations and obtains WASS/MMD 0.241/0.451; KDE-ABC uses 150,000 and obtains 0.690/1.015. The appendix uses another 5000 tasks for SBC, with slight rank bias remaining.
- **Limitations:** the simulator is largely assumed correct. A posterior mean close to truth does not prove the distribution correct, and good posterior prediction does not prove the structure true. The graph network uses actually observed individual states and edges.
- **Implication:** reuse posterior comparisons and calibration checks, without equating a fixed anatomy graph with additional organ information. AGPL-3.0 code contains modified dependencies and needs version-locked reproduction; reported performance applies to specific examples and cannot guarantee identical PBPK speedups.

### 16) Liu et al.: fast surrogate calibration can still recover incorrect parameters

*A neural network-based framework for financial model calibration*. **Journal of Mathematics in Industry, 2019, 9:9**.[Full text](https://link.springer.com/article/10.1186/s13362-019-0066-7).

- **Method:** roughly one million simulated samples train a forward MLP mapping parameters to implied volatility. For new quotes, the network is frozen and differential evolution optimizes input parameters. No inverse quote-to-parameter network is trained directly.
- **Experiments:** Heston forward test MSE is 1.23×10⁻⁷; five-parameter calibration averages 0.48 seconds on GPU. The eight-parameter Bates case has total squared error 4.95×10⁻⁶, yet a parameter with true value 1.0 is estimated as 0.598.
- **Key figures and tables:** Figures 4–5 and Table 9 show flat loss valleys and sensitivity differences; Table 10 shows that tuning a misspecified model can absorb some effects of another mechanism. Code and data are available on request, not confirmed open source.
- **Implication:** low concentration error does not guarantee physiological parameter truth, and surrogate optimization does not fix this. If our exact linear solver is not slow, generating a million samples to train a surrogate may offer no practical gain.

### 17) McGreivy & Hakim: do not mistake weak baselines for innovation

*Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations*. **Nature Machine Intelligence, 2024, 6:1256–1269**.[Journal page](https://www.nature.com/articles/s42256-024-00897-5); [Complete author manuscript](https://www.osti.gov/servlets/purl/2475825); [Research data](https://doi.org/10.17605/OSF.IO/GQ5B3).

- **Study design:** reviews 82 fluid-related PDE papers; 76 claim advantages, and 60 of those, or 79%, are judged to use weak baselines. Table 2 reimplements more appropriate baselines, substantially reducing some claimed speedups.
- **Key principles:** compare cost at matched error tolerance, use efficient numerical algorithms suited to the task, and distinguish training, data-generation, and online-execution costs. The statistics concern the screened fluid-PDE papers, not all machine-learning papers.
- **Implication:** compare our new algorithm against strong numerical calibration and reasonable probabilistic inference under the same PBPK equations. Two-compartment models test predictive competition after structural simplification and cannot alone establish a new inference algorithm's effectiveness. Surrogates must report offline investment and the number of reuses required to break even; deliberately slow solvers must not manufacture speedups.

## 4. Additional corrections and checks on novelty boundaries

These additions do not increase the full-text-read count for the original 17 papers. Their purpose is to avoid claiming existing methods as original.

**Nuclear-energy citation correction.** Kobayashi & Alam, *Deep neural operator-driven real-time inference to enable digital twin solutions for nuclear energy systems*, Scientific Reports 2024, DOI [10.1038/s41598-024-51984-x](https://www.nature.com/articles/s41598-024-51984-x). Methods and results were reviewed: DeepONet maps neutron-source inputs at 190 locations to an 80×80 flux field, using PHITS labels, with approximately 30 seconds versus 0.02 seconds reported. It remains a forward surrogate, not inference of unknown parameters from sensors. Real closed-loop operation and uncertainty are future work; dedicated public code was not confirmed.

**Conditioning on arbitrary observations is not new.** ICML 2024 [Simformer / All-in-one simulation-based inference](https://arxiv.org/html/2404.09636v3) already studies arbitrary conditional distributions and missing or irregular observations. The method overview was reviewed here; mask + Transformer + posterior network is not by itself our new algorithm.

**Embedding shared neural networks in mechanistic ODEs is not new either.** The 2025 [cUDE study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12313987/) by de Rooij et al. retains two-compartment c-peptide dynamics and learns an unknown production term, sharing network weights while fitting individual conditioning parameters, then extracting expressions by symbolic regression. Validation/test subjects still require conditioning-parameter estimation; this is not prediction entirely without individual data. [Official code and data](https://github.com/Computational-Biology-TUe/conditional-ude). Public methods and results were reviewed, not the full supplementary materials.

**Separating model discrepancy from parameter sensitivity also has precedent.** Plumlee's JASA 2017 [Bayesian Calibration of Inexact Computer Models](https://doi.org/10.1080/01621459.2016.1211016) proposes discrepancy priors orthogonal to simulator gradients. The abstract and method definition were checked here, not a detailed full-text review. Projecting residuals onto the orthogonal complement of parameter sensitivities cannot itself be presented as a first proposal.

**Symbolic parameter mapping has direct PBPK precedent.** Tomasoni et al., *Predicting Aberrant Fc-fusion Protein Pharmacokinetics from In Silico Structural Properties and Physiologically Based Pharmacokinetic (PBPK) Modeling*, The AAPS Journal 2026, 28:87, [published full text](https://link.springer.com/article/10.1208/s12248-026-01232-z). The full text, S1, and S2 were obtained and reviewed: fit two multipliers per protein, predict them from protein properties by symbolic regression, then connect them to the existing PBPK model for joint optimization. The study uses 41 proteins, with 34 for training and 7 from a later batch for validation, all in mouse intravenous studies. It changes existing uptake and lymphatic-return parameters rather than discovering new state-dependent flux equations.

The paper's median AAFE 1.18 combines training and validation; Figure 6 likewise combines them and reports only 29% of AUC values within two-fold. Abstract error alone cannot establish strong performance on every endpoint. Parameter-perturbation bands of ±20% are not statistical confidence intervals. Public [SymbolicR](https://github.com/cosbi-research/symbolicr) is a general algorithm, not the complete PBPK reproduction project. **Direct implication: replacing RF with symbolic regression to predict four multipliers has clear precedent; we must explain which different problem a new algorithm actually solves.**

## 5. Which of the three directions suits the current blueprint?

| Direction | Practical problem addressed | Already established? | Assessment for this project |
|---|---|---|---|
| Joint parameter-posterior inference | Compensation, multiple solutions, and predictive uncertainty among four multipliers | Yes; extensive SNPE/NLE/MCMC work exists | Strong reliability foundation and baseline; direct integration alone is insufficient novelty |
| Forward neural surrogate/operator | Cost of repeatedly calling expensive simulators | Yes; FNO/DeepONet/MLP surrogates exist | Measure costs first; relearning the current linear ODE is not the priority |
| Physics- and observation-constrained mechanism modification | Dynamic discrepancy unexplained by a fixed parameter family | General symbolic regression, UDEs, and model selection already exist | **Closest to the research question of automatically designing a usable PBPK model**; requires explicit new criteria and richer observations |

The recommendation is not to combine all three in one system. Instead, test the third as the core question, use the first for necessary diagnostics, and retain the existing exact solver for propagation. The complexity of the unknown mapping and available training materials should determine whether machine learning is needed.

## 6. One proposed core algorithmic research question

### 6.1 In one sentence

**On a basic human model, distinguish incomplete parameter tuning from a genuine need to alter mechanisms, retaining only minimal changes supported by observations and useful for the requested endpoints.**

This is a candidate research direction, not an implemented method or a demonstrated claim of originality or superiority over two-compartment models. Automatic selection, symbolic regression, conservation, and posteriors are not new technologies themselves; any contribution must become a reproducible discrimination-and-update algorithm distinguishable from ordinary model selection.

### 6.2 Inputs, intermediate computation, and outputs

**Inputs:** basic human PBPK, drug information with provenance and uncertainty, development-study dosing and sparse concentrations, and user-requested prediction conditions and endpoints. Target-study concentrations do not enter construction. Without same-drug development observations, use the prior-modeling branch and do not claim same-drug mechanism identification.

**Intermediate computation:**

1. Fit adequately within the current four-multiplier family, retaining parameter sets with approximately equivalent fit quality; protection rules for qualified measurements and renal clearance remain active.
2. Determine whether the existing parameter family explains development residuals. If many parameter combinations explain the observations, do not label one combination the true mechanism.
3. Propose very small mechanism edits only when supported, such as a different rate form for a known exchange/clearance term. Internal transport must subtract from one location while adding to another; an arbitrary plasma correction function is not acceptable.
4. A new mechanism must be distinguishable from retuning old parameters and remain supported under held-out development dose/infusion conditions; then assess whether it improves requested-endpoint stability. Lock structure and parameters after selection, and only then read target observations for scoring.
5. When discrimination is impossible, retain the baseline or equivalent candidates and report that the intended use is not reliably supported. Narrower intervals must not conceal insufficient information.

**Outputs:** an executable model, edits and their evidence, the range of parameter/prediction solutions, requested endpoints, and supported scope. Evaluate organ-mechanism truth only when corresponding observations exist.

### 6.3 A concrete starting point for the algorithm

Write the baseline model as:

\[
\dot A=f_0(A,u;\theta,e),\qquad y=H(A,e).
\]

Permit candidates to edit only a few mass-conserving fluxes:

\[
\dot A=f_0(A,u;\theta,e)+S\,g_{\psi,z}(A,u,e).
\]

Here `A` denotes drug amounts, `u` dosing, and `e` evidence; `S` fixes outflow/inflow directions, and `z` selects a few editable mechanisms. `g` is the increment relative to the original flux and may be positive or negative; nonnegativity constrains the modified actual forward rate. Internal-transport columns sum to zero. If excretion/metabolic sinks are added for clearance, the extended mass system still accounts for conservation. Actual rates must also satisfy boundary conditions when donor drug amount is zero. **Nonnegative rates alone do not automatically ensure a correct complete solution.**

A testable approach is to incorporate whether old parameters can compensate for a candidate change into the update criterion. Let `J` be the Jacobian of log predictions for visible development observations with respect to parameters; a local screen is:

\[
r_{\perp}=(I-JJ^+)r.
\]

Whiten residuals and the Jacobian first when noise is heterogeneous. This projection is diagnostic only: residuals at an ordinary least-squares optimum may already be orthogonal to the Jacobian, and nonzero residuals can reflect noise, source differences, or population mismatch, **not necessarily missing mechanisms**. Full nonlinear refitting, discrimination calibration under noise, and validation across development conditions are still required.

The actual research challenge is to make **explainability by parameter compensation, mechanism distinguishability, and requested-endpoint sensitivity jointly constrain the same mechanism update**, and show fewer erroneous edits and more stable transfer than ordinary complexity penalties, standard UDEs, or selecting the lowest fitting error. Merely adding rules and picking the best model is insufficient for strong methodological novelty.

### 6.4 Practical boundaries of current data

- Under a single dose and low concentrations, saturable clearance can approximate a linear CL multiplier; five plasma points cannot guarantee discovery of saturation.
- Multiple doses and infusion conditions can inform mechanism selection only when available in independent development materials. Target data cannot become training data while retaining test status.
- The 30 drugs can support debugging and development but cannot be repackaged as a blinded test. New-drug prediction entirely without same-drug curves must be reported separately from calibration using sparse observations.
- Discrepancy can arise from units, analytes, study populations, and sources, not only model mechanisms. Adding mechanisms cannot replace data verification.

## 7. Demonstrating a methodological contribution rather than a more flexible fitter

| Question | Key experiment | Required baselines | Main outcomes |
|---|---|---|---|
| Can parameter problems be distinguished from mechanism problems? | Three synthetic tasks: only old parameters change; a mechanism is truly missing; only noise/study differences are present | Current four multipliers, ordinary complexity-based model selection, proposed criterion | Erroneous-edit rate, missed detections, mechanism recovery, prediction error |
| Do restricted mechanism edits help? | Identical inputs and budgets; compare editing with and without the corresponding restrictions | Standard UDE or constrained SR; ablate the discrimination criterion | Target-condition GMFE, failure rate, conservation, nonnegativity |
| Does it resolve endpoint instability from multiple solutions? | Parameter/model ensembles with approximately equal development fits | Point estimation, strong MCMC under the same ODE, or profile analysis | AUC/Cmax/half-life interval widths, coverage, tail failures |
| Does it transfer to new conditions? | Predict independent dose/infusion studies after locking | Two-compartment, fixed-PBPK, and reasonable-prior baselines using the same observations | Paired per-drug errors, confidence intervals, all failures |
| Does it offer cross-drug value? | New drugs after development is frozen; learned components separated by drug | Same structure without cross-drug learning | Changes for new drugs, out-of-distribution deterioration, computational cost |

Recovering true mechanisms in synthetic experiments establishes effectiveness only under simulation assumptions; clinical data require separate validation. Eleven output types do not imply observations for all eleven. Endpoints without references remain empty; an algorithmic output is not a basis for fabricated accuracy.

**Suggested reading order:** Rogers Appendix S1 (why mechanisms are selected incorrectly and which experiments correct them) → Gonçalves Figures 5–6 (why multiple parameter sets fit) → Forster Equation 18 and Figures 8–9 (interpretable formulas are not necessarily true mechanisms) → McGreivy's baseline principles. These four sections best turn continued error tuning into a specific methodological research question.

This round produces only the review and candidate directions above. Existing manuscripts and experiments remain unchanged; candidate algorithms are not described as implemented, and simulation performance from the literature does not replace clinical validation in this project.
