from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RAW_MANIFEST_DIR = "00_run_manifest"
RAW_INVENTORY_DIR = "01_raw_inventory"
RAW_PARSED_DIR = "02_parsed_outputs"
RAW_CHUNKS_DIR = "03_chunks_and_embeddings"
RAW_EVIDENCE_DIR = "04_evidence_candidates"
RAW_LLM_DIR = "05_llm_extraction"
RAW_TABLE_RECOVERY_DIR = "06_table_recovery"

RAW_PROCESSING_DIRS = {
    RAW_MANIFEST_DIR,
    RAW_INVENTORY_DIR,
    RAW_PARSED_DIR,
    RAW_CHUNKS_DIR,
    RAW_EVIDENCE_DIR,
    RAW_LLM_DIR,
    RAW_TABLE_RECOVERY_DIR,
}

RAW_PROCESSING_FILE_DIRS = {
    "processing_summary.json": RAW_MANIFEST_DIR,
    "inventory_summary.json": RAW_MANIFEST_DIR,
    "parse_summary.json": RAW_MANIFEST_DIR,
    "parse_progress.json": RAW_MANIFEST_DIR,
    "deterministic_extract_summary.json": RAW_MANIFEST_DIR,
    "deterministic_extract_progress.json": RAW_MANIFEST_DIR,
    "llm_extract_summary.json": RAW_MANIFEST_DIR,
    "llm_extract_progress.json": RAW_MANIFEST_DIR,
    "pipeline_state.json": RAW_MANIFEST_DIR,
    "acquisition_failure_status.jsonl": RAW_MANIFEST_DIR,
    "asset_parse_status.jsonl": RAW_MANIFEST_DIR,
    "processing_failures.jsonl": RAW_MANIFEST_DIR,
    "completeness_matrix.csv": RAW_MANIFEST_DIR,
    "completeness_provenance.jsonl": RAW_MANIFEST_DIR,
    "completeness_summary.json": RAW_MANIFEST_DIR,
    "completeness_progress.json": RAW_MANIFEST_DIR,
    "raw_inventory.jsonl": RAW_INVENTORY_DIR,
    "raw_asset_aliases.jsonl": RAW_INVENTORY_DIR,
    "parsed_documents.jsonl": RAW_PARSED_DIR,
    "parsed_records.jsonl": RAW_PARSED_DIR,
    "parsed_tables.jsonl": RAW_PARSED_DIR,
    "structured_record_processing_status.jsonl": RAW_PARSED_DIR,
    "structured_table_processing_status.jsonl": RAW_PARSED_DIR,
    "archive_members.jsonl": RAW_PARSED_DIR,
    "asset_links.jsonl": RAW_PARSED_DIR,
    "text_chunks.jsonl": RAW_CHUNKS_DIR,
    "embedding_index.jsonl": RAW_CHUNKS_DIR,
    "chunk_processing_status.jsonl": RAW_CHUNKS_DIR,
    "evidence_candidates.jsonl": RAW_EVIDENCE_DIR,
    "evidence_candidates_deterministic.jsonl": RAW_EVIDENCE_DIR,
    "evidence_candidates_llm.jsonl": RAW_EVIDENCE_DIR,
    "evidence_pointers.jsonl": RAW_EVIDENCE_DIR,
    "llm_agent_prompt.md": RAW_LLM_DIR,
    "llm_extraction_requests.jsonl": RAW_LLM_DIR,
    "llm_extraction_responses.jsonl": RAW_LLM_DIR,
    "llm_extraction_requests_live.jsonl": RAW_LLM_DIR,
    "llm_extraction_responses_live.jsonl": RAW_LLM_DIR,
    "llm_classification_samples.jsonl": RAW_LLM_DIR,
    "recovered_tables.jsonl": RAW_TABLE_RECOVERY_DIR,
    "recovered_table_rows.jsonl": RAW_TABLE_RECOVERY_DIR,
    "table_recovery_status.jsonl": RAW_TABLE_RECOVERY_DIR,
    "table_evidence_candidates.jsonl": RAW_TABLE_RECOVERY_DIR,
    "table_evidence_duplicates.jsonl": RAW_TABLE_RECOVERY_DIR,
    "table_review_queue.jsonl": RAW_TABLE_RECOVERY_DIR,
    "table_recovery_summary.json": RAW_TABLE_RECOVERY_DIR,
    "table_recovery_progress.json": RAW_TABLE_RECOVERY_DIR,
}


@dataclass(frozen=True)
class RawProcessingLayout:
    root: Path

    @classmethod
    def from_path(cls, path: Path | str) -> "RawProcessingLayout":
        candidate = Path(path)
        if candidate.name in RAW_PROCESSING_DIRS:
            return cls(candidate.parent)
        return cls(candidate)

    def ensure_dirs(self) -> None:
        for dirname in RAW_PROCESSING_DIRS:
            (self.root / dirname).mkdir(parents=True, exist_ok=True)

    def path(self, filename: str | Path, *, for_write: bool = False) -> Path:
        file_path = Path(filename)
        if file_path.parent != Path("."):
            return self.root / file_path

        name = file_path.name
        dirname = RAW_PROCESSING_FILE_DIRS.get(name)
        staged = self.root / dirname / name if dirname else self.root / name
        flat = self.root / name

        if for_write:
            return staged
        if staged.exists():
            return staged
        if flat.exists():
            return flat
        return staged

    def candidate_paths(self, filename: str | Path) -> list[Path]:
        file_path = Path(filename)
        if file_path.parent != Path("."):
            return [self.root / file_path]
        staged = self.path(file_path, for_write=True)
        flat = self.root / file_path.name
        if staged == flat:
            return [staged]
        return [staged, flat]

    def relative_path(self, filename: str | Path, *, for_write: bool = False) -> str:
        path = self.path(filename, for_write=for_write)
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


def raw_processing_layout(path: Path | str) -> RawProcessingLayout:
    return RawProcessingLayout.from_path(path)
