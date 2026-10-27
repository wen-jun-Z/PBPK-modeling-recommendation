# PBPK 知识库与模型推荐

提供证据检索与溯源、组件推荐、公共模型推荐和可执行蓝图。

**个人网页工作台已接入当前方法（2026-10-19）。** PBPK-Match light 新请求推荐、30 药蓝图设计与实际预测、开发曲线校准、预测曲线和下载均可在网页使用；知识库包含 100 种有证据药物、3,266 条当前可用证据（2026-10-26 实查）及 280 个模型版本。仅监听服务器回环地址，通过个人 SSH 转发和认证访问。[启动与使用](docs/部署.md)

**蓝图核心：来源和用途驱动的自动PBPK构建。** 现有实验支持可追溯构建和部分药物的证据收益，尚未支持普遍准确率优势、缺CL稳定或临床风险可靠。无本药曲线路径与有曲线校准路径分开。[策略与结论边界](docs/推荐策略.md#blueprint-design)

**当前蓝图方法：人体模板与药物证据 → 未知参数校准 → 可行性检查与必要修复 → 目标条件预测。** 合格药物证据、fu等必要输入及其来源仍保留。正文描述和图表统一展示当前方法，不再标注RF；RF实际输入来源、消融与CL预测器比较集中在论文附录表E.6附近说明。既有实验数值不变，不因展示调整重新运行。[参数处理与复现说明](docs/推荐策略.md#blueprint-rf-role)

**最新十药结构改进已完成：允许校准缺少实测依据的血细胞分配。** 完整/五点曲线GMFE由1.4884/1.5055降至1.3773/1.4099，作为有开发曲线分支的新候选保留。主要收益集中在3药，半衰期仍有明显偏差；两倍以内指十药的曲线GMFE，不是全部终点。[结果与限制](experiments/blueprint_design/blood_partition_20261008/README.md)

**上一轮十药关键采样点验证已完成（2026-10-09）。** 用已有140份补点结果学习简单收益评分，十药按9个研究整组留出，未新增PBPK拟合。缺早期点主任务Cmax GMFE为1.3116，时间规则1.3712，平均改善4.35%；3药改善、5药相同、2药退步，配对区间跨1，未达到多数药物受益的预设门槛。保留该轮未通过门槛的结论，不继续在这十药上调选点器；按用户新增要求开展下述30药扩展，原PBPK预测器和论文保留。[结果与比较图](experiments/blueprint_design/learned_sampling_20261009/README.md) · [策略](docs/推荐策略.md#blueprint-design) · [终端命令](docs/开发.md#offline-blueprint-design)

**现有论文保持冻结。** 十药有开发曲线版与二十药无本药曲线版分别报告；不混合两类成绩。旧探索的详细迭代前言已撤下，论文数据、临床来源和必要复现依赖保留。

**30药修复复评已完成。** 统一训练/评价口径、增加研究留出一致性检查，并在质量守恒证明旧范围不可行时修复未实测血液分配下界。全部30药Cmax/曲线GMFE为1.4417/1.5472，旧法1.7619/1.7071。主要收益来自Montelukast构建修复；相对旧法仍6药改善、8药退步、16药相同，不声称多数药物获益。本轮不覆盖论文。[结果、原因与比较图](experiments/blueprint_design/sampling_stability_20261009/README.md)。

- [论文PDF](results/manuscript/pbpk_paper_zh.pdf) · [LaTeX源码](results/manuscript/main.tex) · [论文成果](results/README.md)
- [实验索引](experiments/README.md) · [研究计划](docs/研究计划.md) · [模型推荐策略](docs/推荐策略.md#当前公共模型推荐策略)
- [十药冻结结果](experiments/blueprint_design/vascular_tissue_ten_drug_20260930/results.json) · [当前冻结研究版本](results/data/blueprint_selected.json)

![PBPK 英文机制版：知识构建、图条件组件生成、模型匹配与蓝图设计](results/figures/workflow/01_four_panel_en_preview.png)

新版英文预览：[01 四宫格](results/figures/workflow/01_four_panel_en_preview.png) · [02 四行流程](results/figures/workflow/02_four_lanes_en_preview.png)。采用 Comic Neue 和卡通科研图示，以简短计算步骤解释检索融合、图软 token、类型匹配及质量守恒。四行版的 C 区展示推荐模型的使用结果（浓度预测、AUC 与 Cmax 示意），D 区展示可执行蓝图与验证报告。

<details>
<summary>四行版英文图注</summary>

**Evidence-grounded PBPK recommendation and blueprint design.**
(a) Literature, drug labels, PK studies and public model sources are normalized into source-linked evidence and a versioned model catalog. An evidence subgraph G_i is represented by e_i=f(text(G_i)); its normalized embedding is e-hat_i. The same frozen text encoder f embeds the modeling query q in (b).
(b) Query q retrieves subgraphs G_i using normalized query/subgraph embeddings and fused lexical/dense rankings. Query-conditioned HGT states h_i are pooled into h_0. The shared projector P_phi maps h_0 and h_i into t_0 and t_i, which form T at the green token row; cited text enters the frozen LLM separately. Here norm is L2 normalization and w is the learned attention scoring vector. Attention pooling and the shared projector are trained using alignment followed by token cross-entropy, while the LLM remains frozen. Recommended components and their evidence inform the model requirement graph; matching chain symbols identify this handoff across the central bridge.
(c) G_q and G_M denote the requirement and model-version graphs. The paired Organ nodes illustrate a requirement node v and a model node u. The schematic matrix highlights their entry s_vu; gray dashes exclude cross-type comparisons. In the compact set expression, u ranges over the nodes of G_M, so U_v contains only model nodes with type(u)=type(v). These entries feed the deterministic typed-alignment subscore, which sums over eligible requirement nodes v, excluding root, Objective and Regimen. s_vu is label-token Jaccard similarity, or 1 for an identical semantic ID. Fixed type weights w_v are used. Including zero in the maximum gives m_v=0 for an empty U_v; the score is defined as 0 if there are no eligible requirement nodes. The implemented applicability categories are Reuse, Adapt, Reference and Reject. Applicability constraints take priority over alignment scores; relevance is compared within each tier. Run labels the fine curved link from model reuse to prediction under a dosing scenario. The model-use panel illustrates predicted concentration profiles and pharmacokinetic outputs (AUC and Cmax); these are schematic examples, not measured results or evidence of validation.
(d) Each source E_j supplies a parameter value theta_j, with units, context and provenance retained in theta. The illustrated transfer J_(j→i) runs from kidney j to blood i and contributes to the aggregate flux equations. A_i is compartment drug amount and V_i is its fixed volume; A(t) denotes the full state vector. J denotes intercompartment transfer rates; R_in and R_elim denote external input and local elimination rates, all in amount per unit time. SolveODE denotes solving for the state vector A(t); dividing each amount by volume gives concentration. AUC requires a subsequent time integral. Independent validation is designed to use held-out studies with no calibration overlap. Output illustrations are schematic, not reported validation results.

Heavier solid arrows emphasize the main workflow; fine solid connections show internal relations and physiological flows. Dashed paths denote contextual inputs or node alignment; equations are placed inside or beside their corresponding mechanisms. Curves clarify branching and merging. Line width indicates visual hierarchy, not quantitative flux or confidence.

</details>


[四宫格设计提示词](results/figures/workflow/01_four_panel_en_prompt.txt) · [四行版设计提示词](results/figures/workflow/02_four_lanes_en_prompt.txt) · [四宫格源码](scripts/render_english_panels.py) · [四行版源码](scripts/render_english_lanes.py)

此前的中文结构稿保留：[下载可编辑 PPT（3 页）](results/figures/workflow/PBPK_workflow_editable.pptx)。以下 SVG 与 PPT 对应中文稿，英文机制稿目前提供预览。

| 方案 | 预览 | 矢量图 |
|---|---|---|
| 01 四宫格：论文方法总图 | [PNG](results/figures/workflow/01_four_panel.png) | [SVG](results/figures/workflow/01_four_panel.svg) |
| 02 四行流程：完整过程展开 | [PNG](results/figures/workflow/02_four_lanes.png) | [SVG](results/figures/workflow/02_four_lanes.svg) |

配色参考用户提供的科研图，并经内置图像生成工具探索设计；最终图由矢量元素重建。[设计参考](results/figures/workflow/design_reference.png) · [生成提示词](results/figures/workflow/imagegen_prompt.txt) · [绘图源码](scripts/render_editable_workflow.py)

| 内容 | 当前规模 |
|---|---:|
| 可用证据 | 3,266 条，覆盖 100 种药物（2026-10-26 实查；后续补漏进行中） |
| 公共模型目录 | 202 个模型族、280 个版本 |
| 目录排序数据 | 772 个需求场景、216,160 对兼容相关性标签 |

**证据复核与补漏进行中（2026-10-26）。** 原30药现有2,267条可用证据；37条重复项合并。初次隔离的614条中，来源补全恢复39条，上下文复核另恢复7条；剩余568条继续隔离。新增70药现有999条可用证据：第1批由137条增至422条，包括本次按原始表格逐格核验恢复的45条，已通过来源、关键词与语义检索验收。第2批6次截断请求已补完，正在后续处理；7批尚未全部完成。原始记录、模型目录和冻结实验保留，自动复核不等于专家验证。[表格修复与检索验收](runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014/table_cell_repair_20261026/run_20261009T140919Z/retrieval_verification.json) · [新旧批次覆盖差异](runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014/coverage_audit_20261026.json)。终端执行 `runtime/app-env/bin/python -B scripts/deployment/project_python.py -B scripts/deployment/repair_expansion_quality.py watch` 查看进度。

**新库偏少的已查原因：** 抽取提示词存在漏抽，表格列头未参与数值归属校验，后6批尚未完成修复。相同1,419份输入中，修复后558份由空抽取变为非空候选；仍须逐项核验后才能计为可用证据。此前将初次采集暂缓的1,369个药物—论文条目写为“未下载”不准确：逐文件核验后，1,060个已有正文且已解析，304个只有摘要，5个只有元数据；不能由初始100篇上限推断最终下载缺口。[原文与解析核验](runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014/pmc_content_audit_20261026.json)。

剩余6批沿用已验证的抽取与表格修复，完成来源核对、去重、质量验证后入库。后台每分钟跟进、每20分钟记录审计；仅对确认的生成截断，每批最多执行一次分项恢复，成功后续跑，其他异常停止。上述 `watch` 命令同时显示跟进程序状态，不清零失败次数，不重复下载已有原文。

证据批次的 18 个步骤已完成。本批为 20 种此前没有可用证据的药物新增 1,144 条证据，原有 1,728 条保留。模型目录新增 231 个版本；新增版本可用于检索参考，尚未完成科学执行验证。身份目录的 102 条登记不等于证据覆盖了 102 种药物。

[证据入库结果](data_process/final_evidence/run_20260909T083120Z/00_run_manifest/database_import_summary.json) · [模型扩库验收](experiments/model_recommendation/model_expansion_20260909/final_verification.json)

**模型推荐质量评估已完成（2026-10-01）。** 145条核心请求的40,600对主判定、2,029对复判及12方法统计全部完成。40条有已知正例请求的已判子集上，REARANK/BM25的NDCG@10为0.787/0.769，差值区间跨零；F2LLM适用Y类F1为0.519，复判覆盖与可靠性仍不足。已补入论文表7及简洁分析。[全部统计](results/tables/model_recommendation_quality.csv) · [结果与范围](experiments/model_recommendation/results_summary/README.md) · [论文PDF](results/manuscript/pbpk_paper_zh.pdf)

**自建推荐模型 PBPK-Match（2026-09-27）：** 当前采用MLP版作为主模型（即此前的“MLP推荐基线”），类型匹配与融合版作为结构对照。已完成代码、GPU小规模训练、消融、优化和三种子复核，183项检查通过。仅10条有已知正例的开发请求可评排序：MLP集成NDCG@10为0.807，融合版为0.747；适用识别仍弱，主模型选择尚需独立验证。102条锁定测试请求的质量标签不用于开发。[设计、全部结果、推理命令与限制](experiments/model_recommendation/pbpk_match/README.md)。

**自建模型安排：** 本次完成外部方法的结果分析，未新增自建模型训练或调参。上述MLP开发成绩使用较早且不同的标签快照，不能与本批40条可排序请求的结果直接比较；是否优化另据开发集诊断决定。

**LambdaMART 的满分已复核：指标没有算错，但标签与输入特征表达同一套匹配规则。** 仅四个覆盖特征训练的浅树也能取得同样的排序满分。新实验不使用这些标签作为科学答案；已有真实 PK 结果已提取，独立相关性与适用性判断仍需补齐。[审计结果](experiments/model_recommendation/runs/expanded_catalog_20260913/lambdamart_audit.json)

真实 PBPK 执行与蓝图功能保留。项目内 OSPSuite 环境已就绪，Moxifloxacin 的 14 个场景已运行并通过工程验收，独立科学验证尚未完成。[仿真与蓝图结果](experiments/README.md#4-实际仿真与蓝图)

Streamlit 保持关闭。排序训练、评测和仿真不需要持续调用 LLM；疑难证据处理与组件生成按需调用已登记的服务器 Qwen 服务。

| 目录 | 内容 |
|---|---|
| `src/` | 应用和算法代码 |
| `docs/` | 核心流程、策略、开发与部署说明 |
| `results/` | 论文图片、最终结果表和支撑数据 |
| `experiments/` | 当前实验结果、必要输入和实际仿真输出 |
| `data_process/` | 最终证据、模型目录和批次验收 |
| `artifacts/` | 原文、公共模型源码和运行资产 |
| `scripts/` | 部署、数据处理与实验入口 |
| `tests/` | 必要的功能与规则检查 |
| `runtime/` | 项目环境、模型权重、数据库和日志 |
| `dependencies/` | 必需的第三方依赖 |
