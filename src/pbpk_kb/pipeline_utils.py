from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from pbpk_kb.paths import processing_output_root, raw_output_root


DEFAULT_COMPOUNDS = (
    "midazolam",
    "warfarin",
    "caffeine",
    "metformin",
    "digoxin",
    "omeprazole",
    "theophylline",
    "dextromethorphan",
    "rosuvastatin",
    "repaglinide",
)
DEFAULT_RAW_OUTPUT_ROOT = raw_output_root()
DEFAULT_PROCESSING_OUTPUT_ROOT = processing_output_root()
DEFAULT_SOURCE_REGISTRY = Path(__file__).resolve().parent / "config" / "sources.yaml"


@dataclass(frozen=True)
class PipelineStage:
    name: str
    strategy: str
    purpose: str


# This is the only legal orchestration order. The individual stage classes also
# validate their upstream pipeline_state.json before doing any work.
PIPELINE_STAGES = (
    PipelineStage("raw_acquisition", "原始资料检索策略", "发现、筛选并下载不可变原始资料"),
    PipelineStage("raw_finalization", "原始资料检索策略", "补齐 PMC、失败重试和共享来源快照"),
    PipelineStage("pmc_static_transfer", "原始资料检索策略", "下载计划中的 PMC XML/PDF/附件静态对象"),
    PipelineStage("raw_reconciliation", "原始资料检索策略", "执行完整性、checksum 和失败终态验收"),
    PipelineStage("raw_inventory", "原始资料处理策略", "建立 raw inventory 并验证文件与 checksum"),
    PipelineStage("parse", "原始资料处理策略", "解析 raw 为 records、documents、tables 和 chunks"),
    PipelineStage("deterministic_extract", "原始资料处理策略", "结构化映射和高精度规则抽取"),
    PipelineStage("llm_extract", "原始资料处理策略", "只处理分类为 llm_required 的文本"),
    PipelineStage("table_recovery", "原始资料处理策略", "恢复复杂或展平表格并抽取可追溯候选"),
    PipelineStage("completeness", "原始资料处理策略", "生成来源/主题/化合物完整性矩阵"),
    PipelineStage("provenance_attribution", "最终证据获取策略", "校验 provenance 并完成化合物归属"),
    PipelineStage("domain_and_deterministic_dedup", "最终证据获取策略", "领域门控和三层确定性去重"),
    PipelineStage("semantic_dedup", "最终证据获取策略", "对剩余自然语言 evidence 做 biomedical embedding 去重"),
    PipelineStage("conflict_and_evidence_layers", "最终证据获取策略", "冲突分析并生成 preselected/reviewable 层"),
    PipelineStage("rag_index", "最终证据获取策略", "最终证据完成后建立独立 RAG 检索索引"),
    PipelineStage("deterministic_quality_control", "自动证据质量控制策略", "执行来源、字段、单位、上下文和冲突质量门"),
    PipelineStage("llm_quality_validation", "自动证据质量控制策略", "修复待定证据并执行独立 LLM 验证"),
    PipelineStage("database_import", "最终证据获取策略", "只将 final_auto_validated 服务层和关联 RAG 写入运行数据库"),
)


@dataclass
class PipelineConfig:
    compounds: tuple[str, ...] = DEFAULT_COMPOUNDS
    raw_output_root: Path = DEFAULT_RAW_OUTPUT_ROOT
    processing_output_root: Path = DEFAULT_PROCESSING_OUTPUT_ROOT
    source_registry_path: Path = DEFAULT_SOURCE_REGISTRY
    raw_run_root: Path | None = None
    resume_incomplete: bool = True
    shared_sources: tuple[str, ...] = ()
    include_combination_labels: bool = False

    acquisition_timeout_seconds: int = 45
    acquisition_delay_seconds: float = 0.34
    ncbi_batch_size: int = 200
    acquisition_page_size: int = 10_000
    dailymed_workers: int = 8
    clinicaltrials_workers: int = 2
    pkdb_workers: int = 4
    pmc_download_workers: int = 8
    pmc_max_connections: int = 16
    max_pmc_fulltext_per_compound: int = 250
    max_pmc_oa_packages_per_compound: int = 100

    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_concurrency: int = 32
    llm_max_requests: int | None = None
    llm_max_retries: int = 4
    llm_timeout_seconds: int = 60
    llm_preflight: Callable[[], None] | None = field(default=None, repr=False)

    embedding_model: str = "NeuML/pubmedbert-base-embeddings"
    embedding_device: str = "auto"
    embedding_batch_size: int = 64
    allow_cpu_embedding: bool = False

    import_database: bool = True
    replace_database: bool = True
    database_url: str | None = None
    docker_container: str = "pbpk_stage1_postgres"
    db_user: str = "pbpk"
    db_name: str = "pbpk_kb"

    checksum_mode: str = "verify"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.compounds:
            raise ValueError("At least one compound is required.")
        if len(set(name.strip().lower() for name in self.compounds)) != len(self.compounds):
            raise ValueError("Compound names must be unique.")
        if not self.source_registry_path.is_file():
            raise FileNotFoundError(self.source_registry_path)
        if self.embedding_device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("embedding_device must be auto, cuda, mps, or cpu.")
        if self.embedding_device == "cpu" and not self.allow_cpu_embedding:
            raise ValueError("CPU embedding requires allow_cpu_embedding=True.")
        if self.llm_concurrency < 1:
            raise ValueError("llm_concurrency must be positive.")
        if self.max_pmc_fulltext_per_compound < 1 or self.max_pmc_oa_packages_per_compound < 1:
            raise ValueError("PMC acquisition limits must be positive.")
        if self.raw_run_root is not None and not self.raw_run_root.is_dir():
            raise FileNotFoundError(self.raw_run_root)


class UnifiedStage1Pipeline:
    """Run all three strategy documents through one ordered, resumable entrypoint."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        handlers: dict[str, Callable[[], dict[str, Any]]] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._retriever: Any | None = None
        if self.config.raw_run_root is None and self.config.resume_incomplete:
            resumable = self._find_resumable_raw_run()
            if resumable is not None:
                self.config.raw_run_root = resumable
        self.raw_run_root = self._resolve_raw_run_root()
        self.raw_output_root = self._resolve_effective_raw_output_root()
        self.run_id = self.raw_run_root.name
        self.raw_processing_dir = (
            self.config.processing_output_root / "raw_processing" / self.run_id
        ).resolve()
        self.final_evidence_dir = (
            self.config.processing_output_root / "final_evidence" / self.run_id
        ).resolve()
        self.manifest_path = (
            self.config.processing_output_root / "pipeline_runs" / f"{self.run_id}.json"
        ).resolve()
        self.handlers = handlers or self._default_handlers()

    def plan(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "raw_output_root": str(self.raw_output_root),
            "raw_run_root": str(self.raw_run_root),
            "raw_processing_dir": str(self.raw_processing_dir),
            "final_evidence_dir": str(self.final_evidence_dir),
            "pipeline_manifest": str(self.manifest_path),
            "compounds": list(self.config.compounds),
            "shared_sources": list(self.config.shared_sources),
            "database_import_enabled": self.config.import_database,
            "stages": [
                {
                    "position": position,
                    "name": stage.name,
                    "strategy": stage.strategy,
                    "purpose": stage.purpose,
                }
                for position, stage in enumerate(PIPELINE_STAGES, start=1)
                if self.config.import_database or stage.name != "database_import"
            ],
        }

    def run(self) -> dict[str, Any]:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._load_or_create_manifest()
        enabled_stages = [
            stage
            for stage in PIPELINE_STAGES
            if self.config.import_database or stage.name != "database_import"
        ]

        for stage in enabled_stages:
            stage_state = manifest["stages"].get(stage.name) or {}
            if stage_state.get("status") == "completed" and stage_state.get("accepted") is True:
                continue

            manifest["status"] = "running"
            manifest["current_stage"] = stage.name
            manifest["updated_at"] = utc_now()
            manifest["stages"][stage.name] = {
                "status": "running",
                "accepted": False,
                "started_at": utc_now(),
                "strategy": stage.strategy,
                "purpose": stage.purpose,
            }
            atomic_write_json(self.manifest_path, manifest)

            try:
                result = self.handlers[stage.name]()
                accepted = bool(result.get("accepted"))
                if not accepted:
                    raise RuntimeError(f"Stage {stage.name} did not pass acceptance checks.")
            except Exception as exc:
                manifest["status"] = "failed"
                manifest["current_stage"] = stage.name
                manifest["updated_at"] = utc_now()
                manifest["stages"][stage.name].update(
                    {
                        "status": "failed",
                        "accepted": False,
                        "failed_at": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                atomic_write_json(self.manifest_path, manifest)
                raise

            manifest["stages"][stage.name].update(
                {
                    "status": "completed",
                    "accepted": True,
                    "completed_at": utc_now(),
                    "result": compact_result(result),
                }
            )
            manifest["updated_at"] = utc_now()
            atomic_write_json(self.manifest_path, manifest)

        manifest["status"] = "completed"
        manifest["current_stage"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["updated_at"] = manifest["completed_at"]
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _resolve_raw_run_root(self) -> Path:
        if self.config.raw_run_root is not None:
            return self.config.raw_run_root.resolve()

        self._retriever = self._make_retriever()
        return self._retriever.run_root.resolve()

    def _make_retriever(self):

        from pbpk_kb.acquisition.comprehensive_raw_retriever import ComprehensiveRawRetriever

        return ComprehensiveRawRetriever(
            output_root=self.config.raw_output_root.resolve(),
            compounds=list(self.config.compounds),
            source_registry_path=self.config.source_registry_path.resolve(),
            timeout_seconds=self.config.acquisition_timeout_seconds,
            delay_seconds=self.config.acquisition_delay_seconds,
            ncbi_batch_size=self.config.ncbi_batch_size,
            page_size=self.config.acquisition_page_size,
            dailymed_workers=self.config.dailymed_workers,
            pkdb_workers=self.config.pkdb_workers,
            download_oa_packages=True,
            mirror_repositories=True,
            include_shared_sources=bool(self.config.shared_sources),
            include_combination_labels=self.config.include_combination_labels,
            max_pmc_fulltext_per_compound=self.config.max_pmc_fulltext_per_compound,
            max_pmc_oa_packages_per_compound=self.config.max_pmc_oa_packages_per_compound,
            official_run=True,
            clinicaltrials_workers=self.config.clinicaltrials_workers,
            shared_source_ids=list(self.config.shared_sources),
        )

    def _find_resumable_raw_run(self) -> Path | None:
        manifest_root = self.config.processing_output_root / "pipeline_runs"
        if not manifest_root.is_dir():
            return None
        expected_compounds = [name.strip().lower() for name in self.config.compounds]
        for manifest_path in sorted(
            manifest_root.glob("run_*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        ):
            manifest = load_json(manifest_path)
            if manifest.get("status") == "completed":
                continue
            compounds = [str(name).strip().lower() for name in manifest.get("compounds") or []]
            if compounds != expected_compounds:
                continue
            raw_run_root = Path(str(manifest.get("raw_run_root") or ""))
            if raw_run_root.is_dir():
                return raw_run_root.resolve()
        return None

    def _resolve_effective_raw_output_root(self) -> Path:
        if self.config.raw_run_root is not None and self.raw_run_root.parent.name == "compound_runs":
            return self.raw_run_root.parents[1]
        return self.config.raw_output_root.resolve()

    def _default_handlers(self) -> dict[str, Callable[[], dict[str, Any]]]:
        return {
            "raw_acquisition": self._run_raw_acquisition,
            "raw_finalization": self._run_raw_finalization,
            "pmc_static_transfer": self._run_pmc_static_transfer,
            "raw_reconciliation": self._run_raw_reconciliation,
            "raw_inventory": lambda: self._run_raw_processing_stage("inventory"),
            "parse": lambda: self._run_raw_processing_stage("parse"),
            "deterministic_extract": lambda: self._run_raw_processing_stage("deterministic_extract"),
            "llm_extract": self._run_llm_extract,
            "table_recovery": self._run_table_recovery,
            "completeness": self._run_completeness,
            "provenance_attribution": self._run_provenance_attribution,
            "domain_and_deterministic_dedup": self._run_domain_and_deterministic_dedup,
            "semantic_dedup": self._run_semantic_dedup,
            "conflict_and_evidence_layers": self._run_conflict_and_evidence_layers,
            "rag_index": self._run_rag_index,
            "deterministic_quality_control": self._run_deterministic_quality_control,
            "llm_quality_validation": self._run_llm_quality_validation,
            "database_import": self._run_database_import,
        }

    def _run_raw_acquisition(self) -> dict[str, Any]:
        if self._accepted_raw_result():
            return {"accepted": True, "reused": True, "run_root": str(self.raw_run_root)}
        if self._retriever is None:
            # A caller may reserve an explicit empty run directory before starting
            # its watchdog. Existing partially written runs are never overwritten.
            if not any(self.raw_run_root.iterdir()):
                self._retriever = self._make_retriever()
                self._retriever.bind_new_run_root(self.raw_run_root)
                summary = self._retriever.run()
                return {"accepted": True, "run_root": str(self.raw_run_root), "summary": summary}
            required = ("RUN_STATUS.json", "assets.jsonl", "failures.jsonl", "download_plan.json")
            missing = [name for name in required if not (self.raw_run_root / name).exists()]
            if missing:
                raise FileNotFoundError(f"Existing raw run is missing acquisition files: {missing}")
            if load_json(self.raw_run_root / "RUN_STATUS.json").get("status") != "completed":
                self._retriever = self._make_retriever()
                summary = self._retriever.resume_after_discovery(self.raw_run_root)
                return {"accepted": True, "resumed": True, "summary": summary}
            return {"accepted": True, "reused": True, "run_root": str(self.raw_run_root)}
        summary = self._retriever.run()
        return {"accepted": True, "run_root": str(self.raw_run_root), "summary": summary}

    def _run_raw_finalization(self) -> dict[str, Any]:
        if self._accepted_raw_result():
            return {"accepted": True, "reused": True}

        from pbpk_kb.acquisition.finalize_raw_run import RawRunFinalizer

        finalizer = RawRunFinalizer(
            run_root=self.raw_run_root,
            output_root=self.raw_output_root,
            sources_yaml=self.config.source_registry_path.resolve(),
            compounds=list(self.config.compounds),
            timeout_seconds=self.config.acquisition_timeout_seconds,
            delay_seconds=self.config.acquisition_delay_seconds,
            pmc_download_workers=self.config.pmc_download_workers,
            # Build a complete transfer plan, then let the dedicated resumable
            # transfer stage fetch every planned object.
            max_pmc_object_bytes=0,
            finish_shared_sources=bool(self.config.shared_sources),
            shared_source_ids=list(self.config.shared_sources),
        )
        summary = finalizer.finalize()
        return {"accepted": True, "summary": summary}

    def _run_pmc_static_transfer(self) -> dict[str, Any]:
        if self._accepted_raw_result():
            return {"accepted": True, "reused": True}

        from pbpk_kb.acquisition.transfer_pmc_static_objects import PMCObjectTransfer

        plan_path = self.raw_run_root / "finalization" / "pmc_object_download_plan.jsonl"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.touch(exist_ok=True)
        transfer = PMCObjectTransfer(
            run_root=self.raw_run_root,
            workers=self.config.pmc_download_workers,
            connect_timeout_seconds=30,
            low_speed_limit_bytes=8192,
            low_speed_time_seconds=120,
            max_transfer_seconds=600,
            retries=2,
            range_connections=4,
            range_min_bytes=8 * 1024 * 1024,
            max_connections=self.config.pmc_max_connections,
            dry_run=False,
            max_objects=None,
        ).run()
        accepted = (
            transfer.get("status") == "completed"
            and int(transfer.get("failed_this_invocation_count") or 0) == 0
            and int(transfer.get("remaining_after_invocation_count") or 0) == 0
        )
        return {"accepted": accepted, "summary": transfer}

    def _run_raw_reconciliation(self) -> dict[str, Any]:
        if self._accepted_raw_result():
            result = load_json(
                self.raw_run_root / "finalization" / "FINAL_ACQUISITION_RESULT.json"
            )
            return {"accepted": True, "reused": True, "summary": result}

        from pbpk_kb.acquisition.reconcile_final_acquisition import reconcile

        result = reconcile(
            self.raw_run_root,
            self.raw_output_root,
            list(self.config.compounds),
        )
        return {
            "accepted": result.get("accepted_as_complete_raw_content") is True,
            "summary": result,
        }

    def _run_raw_processing_stage(self, stage: str) -> dict[str, Any]:
        if self._raw_stage_accepted(stage):
            return {"accepted": True, "reused": True, "stage": stage}

        from pbpk_kb.processing.raw_processor import RawProcessingPipeline

        summary = RawProcessingPipeline(
            run_root=self.raw_run_root,
            output_root=self.raw_processing_dir.parent,
            checksum_mode=self.config.checksum_mode,
            max_assets=None,
            llm_mode="off",
            llm_max_requests=0,
            embedding_mode="off",
            stage=stage,
        ).process()
        return {"accepted": summary.get("accepted") is True, "summary": summary}

    def _run_llm_extract(self) -> dict[str, Any]:
        if self._raw_stage_accepted("llm_extract"):
            return {"accepted": True, "reused": True}
        if self.config.llm_preflight is not None:
            self.config.llm_preflight()

        from pbpk_kb.processing.run_llm_extraction import run_llm_extraction

        classification = run_llm_extraction(
            processed_dir=self.raw_processing_dir,
            llm_max_requests=0,
            llm_model=self.config.llm_model,
            llm_base_url=self.config.llm_base_url,
            concurrency=self.config.llm_concurrency,
            min_chars=160,
            fail_fast=True,
            classify_only=True,
            reuse_by_text_hash=True,
        )
        atomic_write_json(
            self.raw_processing_dir / "00_run_manifest" / "llm_classification_summary.json",
            classification,
        )
        summary = run_llm_extraction(
            processed_dir=self.raw_processing_dir,
            llm_max_requests=self.config.llm_max_requests,
            llm_model=self.config.llm_model,
            llm_base_url=self.config.llm_base_url,
            llm_max_retries=self.config.llm_max_retries,
            retry_backoff_seconds=1.0,
            concurrency=self.config.llm_concurrency,
            min_chars=160,
            fail_fast=True,
            classify_only=False,
            llm_include_existing=False,
            reuse_by_text_hash=True,
            reset_live_ledger=False,
        )
        return {"accepted": summary.get("accepted") is True, "summary": summary}

    def _run_table_recovery(self) -> dict[str, Any]:
        if self._raw_stage_accepted("table_recovery"):
            return {"accepted": True, "reused": True}

        from pbpk_kb.processing.table_recovery_stage import TableRecoveryStage

        summary = TableRecoveryStage(output_dir=self.raw_processing_dir).run()
        return {"accepted": summary.get("accepted") is True, "summary": summary}

    def _run_completeness(self) -> dict[str, Any]:
        if self._raw_stage_accepted("completeness"):
            return {"accepted": True, "reused": True}

        from pbpk_kb.processing.completeness_stage import CompletenessStage

        summary = CompletenessStage(output_dir=self.raw_processing_dir).run()
        return {"accepted": summary.get("accepted") is True, "summary": summary}

    def _run_provenance_attribution(self) -> dict[str, Any]:
        if self._final_stage_accepted("compound_attribution"):
            return {"accepted": True, "reused": True}
        if self.config.llm_preflight is not None:
            self.config.llm_preflight()

        from pbpk_kb.processing.final_evidence_attribution_stage import ProvenanceAttributionStage

        report = ProvenanceAttributionStage(
            run_dir=self.raw_processing_dir,
            output_dir=self.final_evidence_dir,
            llm_mode="live",
            llm_limit=0,
            llm_model=self.config.llm_model,
            llm_base_url=self.config.llm_base_url,
            llm_concurrency=self.config.llm_concurrency,
            llm_timeout_seconds=self.config.llm_timeout_seconds,
            llm_max_retries=self.config.llm_max_retries,
            llm_reprocess_evidence_ids=set(),
            database_url=self.config.database_url if self.config.import_database else None,
        ).run()
        attribution = report.get("compound_attribution") or {}
        provenance = report.get("provenance") or {}
        return {
            "accepted": attribution.get("accepted") is True and provenance.get("accepted") is True,
            "summary": report,
        }

    def _run_domain_and_deterministic_dedup(self) -> dict[str, Any]:
        if self._final_stage_accepted("domain_quality_gates") and self._final_stage_accepted(
            "deterministic_deduplication"
        ):
            return {"accepted": True, "reused": True}

        from pbpk_kb.processing.final_evidence_quality_stage import FinalEvidenceQualityStage

        report = FinalEvidenceQualityStage(
            run_dir=self.raw_processing_dir,
            output_dir=self.final_evidence_dir,
            reset_downstream=False,
        ).run()
        domain = report.get("domain_quality_gates") or {}
        dedup = report.get("deterministic_deduplication") or {}
        return {
            "accepted": domain.get("accepted") is True and dedup.get("accepted") is True,
            "summary": report,
        }

    def _run_semantic_dedup(self) -> dict[str, Any]:
        if self._final_stage_accepted("semantic_deduplication"):
            return {"accepted": True, "reused": True}

        from pbpk_kb.processing.semantic_dedup_file import SemanticDedupRunner

        report = SemanticDedupRunner(
            input_path=self.final_evidence_dir / "03_deduplication" / "semantic_dedup_input.jsonl",
            output_dir=self.final_evidence_dir,
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            threshold=0.92,
            review_threshold=0.85,
            batch_size=self.config.embedding_batch_size,
            max_source_chars=1500,
            min_source_chars=40,
            min_group_size=2,
            max_group_size=5000,
            reset=False,
        ).run()
        return {"accepted": report.get("accepted") is True, "summary": report}

    def _run_conflict_and_evidence_layers(self) -> dict[str, Any]:
        if self._final_stage_accepted("conflict_analysis") and self._final_stage_accepted(
            "evidence_layers"
        ):
            return {"accepted": True, "reused": True}

        from pbpk_kb.processing.final_evidence_post_stage import FinalEvidencePostProcessor

        report = FinalEvidencePostProcessor(
            raw_processing_dir=self.raw_processing_dir,
            output_dir=self.final_evidence_dir,
            reset=False,
        ).run()
        conflict = report.get("conflict_analysis") or {}
        layers = report.get("evidence_layers") or {}
        return {
            "accepted": conflict.get("accepted") is True and layers.get("accepted") is True,
            "summary": report,
        }

    def _run_rag_index(self) -> dict[str, Any]:
        if self._final_stage_accepted("rag_index"):
            return {"accepted": True, "reused": True}

        from pbpk_kb.processing.rag_index_stage import RagIndexStage

        runner = RagIndexStage(
            raw_processing_dir=self.raw_processing_dir,
            final_evidence_dir=self.final_evidence_dir,
            model_name=self.config.embedding_model,
            requested_device=self.config.embedding_device,
            allow_cpu=self.config.allow_cpu_embedding,
            batch_size=self.config.embedding_batch_size,
            reset=False,
        )
        state = load_json(self.final_evidence_dir / "00_run_manifest" / "pipeline_state.json")
        rag_state = ((state.get("stages") or {}).get("rag_index") or {}).get("status")
        prepared = None
        if rag_state != "prepared":
            prepared = runner.prepare()
        embedded = runner.embed()
        return {
            "accepted": embedded.get("accepted") is True,
            "prepare": prepared,
            "embed": embedded,
        }

    def _run_database_import(self) -> dict[str, Any]:
        if self._final_stage_accepted("database_import"):
            return {"accepted": True, "reused": True}

        from pbpk_kb.storage.import_processed_evidence import EvidenceImporter

        summary = EvidenceImporter(
            run_dir=self.raw_processing_dir,
            final_evidence_dir=self.final_evidence_dir,
            container=self.config.docker_container,
            db_user=self.config.db_user,
            db_name=self.config.db_name,
            database_url=self.config.database_url,
            replace=self.config.replace_database,
            limit=None,
        ).run()
        return {"accepted": True, "summary": summary}

    def _run_deterministic_quality_control(self) -> dict[str, Any]:
        summary_path = (
            self.final_evidence_dir
            / "06_auto_quality_control"
            / "quality_summary.json"
        )
        summary = load_json(summary_path)
        if summary.get("accepted") is True:
            return {"accepted": True, "reused": True, "summary": summary}

        from pbpk_kb.processing.auto_quality_stage import AutoQualityStage

        summary = AutoQualityStage(
            final_run_dir=self.final_evidence_dir,
            reset=False,
        ).run()
        return {"accepted": summary.get("accepted") is True, "summary": summary}

    def _run_llm_quality_validation(self) -> dict[str, Any]:
        summary_path = (
            self.final_evidence_dir
            / "07_llm_quality_validation"
            / "quality_summary.json"
        )
        summary = load_json(summary_path)
        if summary.get("accepted") is True:
            return {"accepted": True, "reused": True, "summary": summary}
        if self.config.llm_preflight is not None:
            self.config.llm_preflight()
        from pbpk_kb.processing.auto_quality_llm_stage import AutoQualityLLMStage

        summary = AutoQualityLLMStage(
            final_run_dir=self.final_evidence_dir,
            raw_run_dir=self.raw_processing_dir,
            base_url=self.config.llm_base_url or "local://qwen3-8b",
            model=self.config.llm_model or "Qwen3-8B-BF16",
            workers=self.config.llm_concurrency,
            reset=False,
        ).run()
        return {"accepted": summary.get("accepted") is True, "summary": summary}

    def _raw_stage_accepted(self, stage_name: str) -> bool:
        state = load_json(self.raw_processing_dir / "00_run_manifest" / "pipeline_state.json")
        stage = (state.get("stages") or {}).get(stage_name) or {}
        return stage.get("status") == "completed" and stage.get("accepted") is True

    def _final_stage_accepted(self, stage_name: str) -> bool:
        state = load_json(self.final_evidence_dir / "00_run_manifest" / "pipeline_state.json")
        stage = (state.get("stages") or {}).get(stage_name) or {}
        return stage.get("status") == "completed" and stage.get("accepted") is True

    def _accepted_raw_result(self) -> bool:
        result = load_json(
            self.raw_run_root / "finalization" / "FINAL_ACQUISITION_RESULT.json"
        )
        status = load_json(self.raw_run_root / "RUN_STATUS.json")
        return (
            result.get("accepted_as_complete_raw_content") is True
            and status.get("status") == "completed"
            and status.get("accepted_as_final_result") is True
        )

    def _load_or_create_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            manifest = load_json(self.manifest_path)
            if manifest.get("run_id") != self.run_id:
                raise RuntimeError("Pipeline manifest run_id does not match the selected raw run.")
            if manifest.get("compounds") != list(self.config.compounds):
                raise RuntimeError("Pipeline manifest compounds do not match the explicit requested targets.")
            for stage in (manifest.get("stages") or {}).values():
                if isinstance(stage, dict) and isinstance(stage.get("result"), dict):
                    stage["result"] = compact_result(stage["result"])
            return manifest
        plan = self.plan()
        return {
            **plan,
            "status": "not_started",
            "current_stage": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "strategy_documents": [
                "docs/证据流程.md",
            ],
            "stages": {},
            "metadata": json_safe(self.config.metadata),
        }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, dict):
            scalar_values = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
            }
            if scalar_values:
                compact[key] = scalar_values
    return json_safe(compact)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
