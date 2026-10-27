from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from pbpk_kb.paths import recommendation_model_cache_dir
from pbpk_kb.recommendation.ontology import RELATION_COMPACT_GLOSSES


BENCHMARK_VERSION = "pbpk-retrieval-silver-1.2.0"
MODEL_INPUT_SERIALIZATION_VERSION = "pbpk-model-input-1.0.2"
MINIMUM_QUERY_COUNT = 120
ACCEPTANCE_THRESHOLDS = {
    "minimum_query_count": MINIMUM_QUERY_COUNT,
    "minimum_category_count": 5,
    "minimum_selected_hit_rate_at_10": 0.80,
    "minimum_selected_mrr": 0.45,
    "minimum_selected_ndcg_at_10": 0.50,
    "minimum_composite_improvement_over_best_baseline": 0.005,
    "maximum_category_hit_rate_regression_vs_best_baseline": 0.10,
}

PARAMETER_SPECS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("auc", "auc 0", "auc_", "systemic_exposure", "plasma_exposure", "exposure"),
        (
            "area under the plasma concentration time curve",
            "total systemic drug exposure over time",
            "integrated concentration versus time exposure",
        ),
    ),
    (
        ("cmax", "peak concentration", "peak plasma concentration"),
        (
            "highest observed plasma drug concentration",
            "maximum concentration reached after dosing",
            "peak circulating concentration",
        ),
    ),
    (
        ("tmax",),
        (
            "time required to reach peak concentration",
            "time of the maximum plasma level",
            "when the concentration peak occurs",
        ),
    ),
    (
        ("half-life", "half_life", "halflife"),
        (
            "time for drug concentration to decrease by one half",
            "terminal elimination half life",
            "duration of concentration decay by fifty percent",
        ),
    ),
    (
        ("clearance",),
        (
            "volume of plasma cleared of drug per unit time",
            "drug elimination clearance capacity",
            "systemic removal rate normalized by concentration",
        ),
    ),
    (
        ("volume of distribution", "volume_of_distribution"),
        (
            "apparent volume relating amount in body to plasma concentration",
            "extent of drug distribution volume",
            "central or total apparent distribution space",
        ),
    ),
    (
        ("bioavailability",),
        (
            "fraction of an administered dose reaching systemic circulation",
            "absolute or relative systemic availability after dosing",
            "extent of dose entering the circulation unchanged",
        ),
    ),
    (
        ("fraction_absorbed", "fraction absorbed"),
        (
            "proportion of the administered dose absorbed from the dosing site",
            "fraction entering the body across the absorption barrier",
            "extent of gastrointestinal dose absorption",
        ),
    ),
    (
        ("fraction_unbound", "protein binding", "protein_binding"),
        (
            "fraction of drug not bound to plasma proteins",
            "unbound circulating drug proportion",
            "plasma protein binding and free fraction",
        ),
    ),
    (
        ("renal clearance", "renal_clearance"),
        (
            "kidney mediated drug clearance",
            "plasma clearance attributable to renal elimination",
            "renal contribution to systemic removal",
        ),
    ),
    (
        ("intrinsic clearance", "intrinsic_clearance"),
        (
            "enzyme intrinsic metabolic clearance capacity",
            "unbound intrinsic hepatic clearance",
            "metabolic elimination ability independent of blood flow",
        ),
    ),
    (
        ("absorption rate constant", "absorption_rate_constant"),
        (
            "first order rate constant for drug absorption",
            "speed constant governing entry from the dosing site",
            "ka absorption kinetic constant",
        ),
    ),
    (
        ("molecularweight", "full_mwt"),
        (
            "molar mass of the chemical compound",
            "molecular mass physicochemical property",
            "chemical formula weight",
        ),
    ),
    (
        ("xlogp", "alogp"),
        (
            "octanol water partition coefficient lipophilicity",
            "logarithmic hydrophobic partition property",
            "compound lipophilicity log p",
        ),
    ),
    (
        ("tpsa", "psa"),
        (
            "topological polar surface area",
            "molecular polar surface area descriptor",
            "surface area contributed by polar atoms",
        ),
    ),
)

RELATION_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "ABSORBED_VIA",
        (
            "evidence describing how a drug enters systemic circulation",
            "drug absorption process from its administration site",
            "mechanism or event governing uptake into the body",
        ),
        "scenario_query",
    ),
    (
        "EXCRETED_VIA",
        (
            "evidence describing renal biliary fecal or urinary drug removal",
            "drug excretion pathway out of the body",
            "route by which unchanged drug or metabolites are eliminated",
        ),
        "scenario_query",
    ),
    (
        "SUBSTRATE_OF",
        (
            "drug identified as a substrate of a metabolic enzyme",
            "compound processed by an enzyme or transporter",
            "substrate relationship for disposition machinery",
        ),
        "scenario_query",
    ),
    (
        "INHIBITS",
        (
            "compound suppressing the activity of a metabolic enzyme",
            "drug acting as an enzyme inhibitor",
            "inhibitory effect on drug metabolism machinery",
        ),
        "scenario_query",
    ),
    (
        "TRANSPORTED_BY",
        (
            "drug disposition mediated by a membrane transporter",
            "compound carried by an uptake or efflux transporter",
            "transporter dependent movement of a drug",
        ),
        "scenario_query",
    ),
    (
        "UNDER_REGIMEN",
        (
            "pharmacokinetic result under a defined dose route and schedule",
            "evidence linked to an administration regimen",
            "dose frequency and route context for a measurement",
        ),
        "scenario_query",
    ),
    (
        "STUDIED_IN",
        (
            "pharmacokinetic evidence in a defined patient or volunteer population",
            "study result tied to demographic or disease context",
            "population specific drug disposition evidence",
        ),
        "special_population",
    ),
    (
        "DESCRIBED_BY_MODEL",
        (
            "public physiologically based pharmacokinetic model for a compound",
            "reusable mechanistic whole body drug disposition model",
            "PBPK model implementation associated with a drug",
        ),
        "scenario_query",
    ),
    (
        "HAS_VALUE",
        (
            "quantitative pharmacokinetic measurement with a numerical value",
            "numeric observation for a PK parameter",
            "measured drug disposition result expressed as a number",
        ),
        "scenario_query",
    ),
    (
        "HAS_UNIT",
        (
            "pharmacokinetic value linked to its measurement unit",
            "unit annotated quantitative PK result",
            "measurement carrying dimensional units",
        ),
        "scenario_query",
    ),
)

POPULATION_SPECS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("healthy", "volunteer", "subject"),
        (
            "pharmacokinetics in healthy volunteer participants",
            "drug disposition in adults without the target disease",
            "normal healthy human subjects",
        ),
    ),
    (
        ("preterm", "neonate"),
        (
            "drug disposition in premature newborn infants",
            "pharmacokinetics in preterm neonates",
            "exposure in newborn patients after premature birth",
        ),
    ),
    (
        ("renal failure", "arf", "kidney"),
        (
            "pharmacokinetics in patients with acute kidney failure",
            "drug exposure under severely impaired renal function",
            "critical care patients with acute renal injury",
        ),
    ),
    (
        ("chinese", "caucasian", "ethnic"),
        (
            "pharmacokinetic evidence in a defined ethnic ancestry group",
            "drug disposition comparison by ethnicity",
            "population PK evidence for Chinese or Caucasian participants",
        ),
    ),
    (
        ("prostate cancer", "cancer"),
        (
            "drug exposure in patients with prostate malignancy",
            "pharmacokinetics in oncology patients with prostate cancer",
            "patient population with metastatic prostate disease",
        ),
    ),
)

DDI_ROLE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "VICTIM_IN",
        (
            "drug whose exposure is altered by a coadministered perpetrator",
            "affected victim compound in a drug interaction",
            "substrate drug receiving the pharmacokinetic interaction effect",
            "compound with AUC or Cmax changed by another drug",
            "victim role in an enzyme mediated DDI",
        ),
    ),
    (
        "PERPETRATOR_IN",
        (
            "drug causing a change in the exposure of a coadministered victim",
            "perpetrator compound responsible for a drug interaction",
            "inhibitor or inducer drug affecting another medicine",
            "causal interacting compound modifying victim AUC",
            "perpetrator role in an enzyme mediated DDI",
        ),
    ),
)


class Encoder(Protocol):
    name: str
    revision: str
    dimension: int

    def encode_documents(self, texts: list[str]) -> Any: ...

    def encode_queries(self, texts: list[str]) -> Any: ...

    def document_token_stats(self, texts: list[str]) -> list[dict[str, Any]]: ...

    def release(self) -> None: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(namespace: str, *parts: Any) -> str:
    body = "\x1f".join(str(part) for part in parts)
    return f"{namespace}:{hashlib.sha256(body.encode('utf-8')).hexdigest()[:24]}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def normalize(value: Any) -> str:
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .replace("–", "-")
        .replace("‐", "-")
        .split()
    )


def _hard_negatives(
    corpus: list[dict[str, Any]],
    positives: set[str],
    query_id: str,
    limit: int = 20,
) -> list[str]:
    candidates = [
        str(row["subgraph_id"])
        for row in corpus
        if str(row["subgraph_id"]) not in positives
    ]
    return sorted(
        candidates,
        key=lambda value: hashlib.sha256(f"{query_id}\x1f{value}".encode()).hexdigest(),
    )[:limit]


def _query_row(
    *,
    text: str,
    category: str,
    target_hop: int,
    positives: set[str],
    corpus: list[dict[str, Any]],
    label_basis: dict[str, Any],
) -> dict[str, Any]:
    query_id = stable_id("retrieval-query", category, target_hop, text)
    return {
        "query_id": query_id,
        "query_text": text,
        "category": category,
        "target_hop": target_hop,
        "positive_subgraph_ids": sorted(positives),
        "hard_negative_subgraph_ids": _hard_negatives(corpus, positives, query_id),
        "label_basis": label_basis,
        "benchmark_level": "silver",
    }


def build_locked_benchmark(subgraph_dir: Path, output_dir: Path) -> dict[str, Any]:
    corpora = {
        hop: load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
        for hop in (0, 1, 2)
    }
    queries: list[dict[str, Any]] = []

    # Entity identity tests are stratified and deterministic so large node classes
    # do not dominate the semantic task categories.
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpora[0]:
        by_type[str(row["center_type"])].append(row)
    for center_type, rows in sorted(by_type.items()):
        selected = sorted(
            rows,
            key=lambda row: hashlib.sha256(str(row["center_id"]).encode()).hexdigest(),
        )[:12]
        for row in selected:
            text = f"Find the normalized {center_type} entity for {row['center_name']}."
            positives = {str(row["subgraph_id"])}
            queries.append(
                _query_row(
                    text=text,
                    category="entity_query",
                    target_hop=0,
                    positives=positives,
                    corpus=corpora[0],
                    label_basis={
                        "center_id": row["center_id"],
                        "center_type": center_type,
                        "stable_identifiers": row["nodes"][0].get("standard_identifiers") or {},
                    },
                )
            )

    parameter_rows = [
        row for row in corpora[0] if row.get("center_type") == "PKParameter"
    ]
    for aliases, paraphrases in PARAMETER_SPECS:
        positives = {
            str(row["subgraph_id"])
            for row in parameter_rows
            if any(alias in normalize(row["center_name"]) for alias in aliases)
        }
        if not positives:
            continue
        for text in paraphrases:
            queries.append(
                _query_row(
                    text=text,
                    category="parameter_query",
                    target_hop=0,
                    positives=positives,
                    corpus=corpora[0],
                    label_basis={"parameter_alias_patterns": list(aliases)},
                )
            )

    compound_by_hop = {
        hop: [
            row for row in corpora[hop] if row.get("center_type") == "Compound"
        ]
        for hop in (1, 2)
    }
    compound_two_hop = compound_by_hop[2]
    for predicate, paraphrases, category in RELATION_SPECS:
        for hop in (1, 2):
            positives = {
                str(row["subgraph_id"])
                for row in compound_by_hop[hop]
                if any(edge.get("predicate") == predicate for edge in row.get("edges") or [])
            }
            if not positives:
                continue
            for text in paraphrases:
                queries.append(
                    _query_row(
                        text=text,
                        category=category,
                        target_hop=hop,
                        positives=positives,
                        corpus=corpora[hop],
                        label_basis={
                            "required_predicate": predicate,
                            "center_type": "Compound",
                            "required_within_hop": hop,
                        },
                    )
                )

    population_rows = [
        row for row in corpora[0] if row.get("center_type") == "Population"
    ]
    for patterns, paraphrases in POPULATION_SPECS:
        positives = {
            str(row["subgraph_id"])
            for row in population_rows
            if any(pattern in normalize(row["center_name"]) for pattern in patterns)
        }
        if not positives:
            continue
        for text in paraphrases:
            queries.append(
                _query_row(
                    text=text,
                    category="special_population",
                    target_hop=0,
                    positives=positives,
                    corpus=corpora[0],
                    label_basis={"population_name_patterns": list(patterns)},
                )
            )

    for predicate, paraphrases in DDI_ROLE_SPECS:
        for hop in (1, 2):
            positives = {
                str(row["subgraph_id"])
                for row in compound_by_hop[hop]
                if any(
                    edge.get("predicate") == predicate
                    and edge.get("subject_id") == row.get("center_id")
                    for edge in row.get("edges") or []
                )
            }
            if not positives:
                continue
            for text in paraphrases:
                queries.append(
                    _query_row(
                        text=text,
                        category="ddi_role",
                        target_hop=hop,
                        positives=positives,
                        corpus=corpora[hop],
                        label_basis={
                            "required_direct_subject_predicate": predicate,
                            "center_type": "Compound",
                            "required_within_hop": hop,
                        },
                    )
                )

    multi_hop_patterns = (
        (
            ("VICTIM_IN", "AFFECTS_PARAMETER"),
            (
                "victim drug connected through an interaction event to an affected PK parameter",
                "affected compound with a two edge path to an exposure endpoint",
                "DDI victim linked to the pharmacokinetic outcome changed by the interaction",
            ),
        ),
        (
            ("PERPETRATOR_IN", "MEDIATED_BY"),
            (
                "perpetrator drug connected through an interaction event to an enzyme mediator",
                "causal interacting compound with a two edge path to metabolic machinery",
                "DDI perpetrator linked to the enzyme or transporter mediating the effect",
            ),
        ),
        (
            ("OF_COMPOUND", "HAS_VALUE"),
            (
                "compound connected through a PK measurement to a quantitative value",
                "drug with a two edge path to a numerical pharmacokinetic observation",
                "measured compound linked to its quantitative PK result",
            ),
        ),
    )
    for predicates, paraphrases in multi_hop_patterns:
        positives: set[str] = set()
        for row in compound_two_hop:
            center_id = str(row["center_id"])
            first_events = {
                str(edge["object_id"])
                for edge in row.get("edges") or []
                if edge.get("subject_id") == center_id and edge.get("predicate") == predicates[0]
            }
            if not first_events and predicates[0] == "OF_COMPOUND":
                first_events = {
                    str(edge["subject_id"])
                    for edge in row.get("edges") or []
                    if edge.get("object_id") == center_id and edge.get("predicate") == predicates[0]
                }
            if any(
                edge.get("subject_id") in first_events and edge.get("predicate") == predicates[1]
                for edge in row.get("edges") or []
            ):
                positives.add(str(row["subgraph_id"]))
        if not positives:
            continue
        for text in paraphrases:
            queries.append(
                _query_row(
                    text=text,
                    category="multi_hop_path",
                    target_hop=2,
                    positives=positives,
                    corpus=corpora[2],
                    label_basis={"required_two_edge_predicates": list(predicates)},
                )
            )

    queries = sorted(queries, key=lambda row: str(row["query_id"]))
    category_counts = Counter(str(row["category"]) for row in queries)
    if len(queries) < MINIMUM_QUERY_COUNT:
        raise RuntimeError(
            f"Locked retrieval benchmark has only {len(queries)} queries; "
            f"minimum is {MINIMUM_QUERY_COUNT}."
        )
    if min(category_counts.values(), default=0) < ACCEPTANCE_THRESHOLDS["minimum_category_count"]:
        raise RuntimeError(f"Retrieval category is undersized: {dict(category_counts)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / "locked_retrieval_silver.jsonl"
    _atomic_jsonl(query_path, queries)
    acceptance_path = output_dir / "preregistered_acceptance.json"
    _atomic_json(
        acceptance_path,
        {
            "benchmark_version": BENCHMARK_VERSION,
            "locked_before_model_evaluation": True,
            "thresholds": ACCEPTANCE_THRESHOLDS,
            "selection_metric": "composite_score",
            "selection_rule": (
                "Compare dense-only and fixed RRF(k=60) dense-plus-lexical retrieval for "
                "every real embedding model. Select the accepted method with the highest "
                "composite score; it must exceed lexical_tfidf and hashing_embedding baselines."
            ),
            "created_at": utc_now(),
        },
    )
    manifest = {
        "accepted": True,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_level": "silver",
        "query_count": len(queries),
        "category_counts": dict(sorted(category_counts.items())),
        "hop_counts": dict(sorted(Counter(str(row["target_hop"]) for row in queries).items())),
        "query_sha256": sha256_file(query_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "subgraph_hashes": {
            f"{hop}hop": sha256_file(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
            for hop in (0, 1, 2)
        },
        "limitations": [
            "Labels are deterministic silver labels derived from the audited KG, not expert gold judgments.",
            "Synthetic query paraphrases test semantic retrieval but do not replace prospective user-query evaluation.",
            "The benchmark is locked by checksum before any candidate model score is calculated.",
        ],
        "generated_at": utc_now(),
    }
    _atomic_json(output_dir / "benchmark_manifest.json", manifest)
    return manifest


class SentenceTransformerEncoder:
    def __init__(
        self,
        *,
        name: str,
        cache_dir: Path,
        device: str,
        query_prefix: str = "",
        query_prompt_name: str | None = None,
        truncate_dimension: int | None = None,
        batch_size: int = 32,
    ) -> None:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        snapshot = Path(
            snapshot_download(
                name,
                cache_dir=str(cache_dir),
                ignore_patterns=[
                    "*.bin",
                    "*.onnx",
                    "*.xml",
                    "*.h5",
                    "*.msgpack",
                    "*.tflite",
                    "openvino/*",
                ],
            )
        )
        self.name = name
        self.revision = snapshot.name
        self.model = SentenceTransformer(
            str(snapshot),
            device=device,
            truncate_dim=truncate_dimension,
        )
        self.dimension = int(
            truncate_dimension or self.model.get_sentence_embedding_dimension() or 0
        )
        self.query_prefix = query_prefix
        self.query_prompt_name = query_prompt_name
        self.batch_size = batch_size

    def _encode(self, texts: list[str], *, prompt_name: str | None = None) -> Any:
        options: dict[str, Any] = {}
        if prompt_name:
            options["prompt_name"] = prompt_name
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            **options,
        )

    def encode_documents(self, texts: list[str]) -> Any:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> Any:
        return self._encode(
            [f"{self.query_prefix}{text}" for text in texts],
            prompt_name=self.query_prompt_name,
        )

    def document_token_stats(self, texts: list[str]) -> list[dict[str, Any]]:
        return _token_stats(
            self.model.tokenizer,
            texts,
            max_length=int(self.model.max_seq_length),
            tokenizer_revision=self.revision,
        )

    def release(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        model.to("cpu")
        self.model = None


class MedCPTEncoder:
    def __init__(
        self,
        *,
        cache_dir: Path,
        device: str,
        batch_size: int = 32,
        query_only: bool = False,
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        query_snapshot = Path(
            snapshot_download(
                "ncbi/MedCPT-Query-Encoder",
                cache_dir=str(cache_dir),
                ignore_patterns=["*.bin"],
            )
        )
        article_snapshot = Path(
            snapshot_download(
                "ncbi/MedCPT-Article-Encoder",
                cache_dir=str(cache_dir),
                ignore_patterns=["*.bin"],
            )
        )
        self.name = "ncbi/MedCPT"
        self.revision = f"query:{query_snapshot.name};article:{article_snapshot.name}"
        self.device = torch.device(device)
        self.torch = torch
        self.query_tokenizer = AutoTokenizer.from_pretrained(str(query_snapshot))
        self.query_model = AutoModel.from_pretrained(str(query_snapshot)).to(self.device).eval()
        self.article_tokenizer = None
        self.article_model = None
        if not query_only:
            self.article_tokenizer = AutoTokenizer.from_pretrained(str(article_snapshot))
            self.article_model = AutoModel.from_pretrained(str(article_snapshot)).to(self.device).eval()
        self.dimension = int(self.query_model.config.hidden_size)
        self.batch_size = batch_size

    def _encode(
        self,
        texts: list[str],
        *,
        tokenizer: Any,
        model: Any,
        max_length: int,
    ) -> Any:
        import numpy as np

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                vectors = model(**encoded).last_hidden_state[:, 0, :]
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
            batches.append(vectors.detach().cpu().numpy().astype("float32"))
        return np.concatenate(batches, axis=0)

    def encode_documents(self, texts: list[str]) -> Any:
        if self.article_tokenizer is None or self.article_model is None:
            raise RuntimeError(
                "This MedCPT encoder was initialized for online query encoding only."
            )
        return self._encode(
            texts,
            tokenizer=self.article_tokenizer,
            model=self.article_model,
            max_length=512,
        )

    def encode_queries(self, texts: list[str]) -> Any:
        return self._encode(
            texts,
            tokenizer=self.query_tokenizer,
            model=self.query_model,
            max_length=64,
        )

    def document_token_stats(self, texts: list[str]) -> list[dict[str, Any]]:
        if self.article_tokenizer is None:
            raise RuntimeError(
                "This MedCPT encoder was initialized for online query encoding only."
            )
        return _token_stats(
            self.article_tokenizer,
            texts,
            max_length=512,
            tokenizer_revision=self.revision,
        )

    def release(self) -> None:
        for attribute in ("query_model", "article_model"):
            model = getattr(self, attribute, None)
            if model is not None:
                model.to("cpu")
                setattr(self, attribute, None)
        self.query_tokenizer = None
        self.article_tokenizer = None


def _token_stats(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    tokenizer_revision: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for start in range(0, len(texts), 256):
        encoded = tokenizer(
            texts[start : start + 256],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        lengths = encoded.get("length")
        if lengths is None:
            lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
        for length in lengths:
            count = int(length)
            results.append(
                {
                    "original_token_count": count,
                    "model_input_token_count": min(count, max_length),
                    "truncated": count > max_length,
                    "model_max_length": max_length,
                    "tokenizer_class": type(tokenizer).__name__,
                    "tokenizer_revision": tokenizer_revision,
                }
            )
    return results


def encoder_tokenizer_and_limit(encoder: Any) -> tuple[Any, int]:
    if hasattr(encoder, "article_tokenizer"):
        return encoder.article_tokenizer, 512
    model = getattr(encoder, "model", None)
    tokenizer = getattr(model, "tokenizer", None)
    max_length = int(getattr(model, "max_seq_length", 0) or 0)
    if tokenizer is None or max_length <= 0:
        raise RuntimeError(f"Cannot inspect tokenizer limits for {encoder.name}.")
    return tokenizer, max_length


def tokenizer_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_length=True,
    )
    length = encoded.get("length")
    if isinstance(length, list):
        return int(length[0])
    if length is not None:
        return int(length)
    return len(encoded["input_ids"])


def _compact_subgraph_lines(lines: list[str]) -> list[str]:
    relation_counts: dict[str, int] = {}
    relation_type_line = next(
        (line for line in lines if line.startswith("RELATION TYPES ")),
        "",
    )
    for item in relation_type_line.removeprefix("RELATION TYPES ").split(";"):
        name, separator, value = item.strip().partition("=")
        if separator and value.isdigit():
            relation_counts[name] = int(value)

    compacted: list[str] = []
    for line in lines:
        if line.startswith("ONTOLOGY VERSION pbpk-recommendation-ontology-"):
            compacted.append(
                "ONTOLOGY VERSION "
                + line.removeprefix(
                    "ONTOLOGY VERSION pbpk-recommendation-ontology-"
                )
            )
            continue
        if line.startswith("CENTER ["):
            compacted.append(line[:128].rstrip() + ("..." if len(line) > 128 else ""))
            continue
        if line.startswith("CENTER IDENTIFIERS "):
            raw = line.removeprefix("CENTER IDENTIFIERS ")
            try:
                identifiers = json.loads(raw)
            except json.JSONDecodeError:
                identifiers = {}
            if identifiers:
                identifiers = {
                    key: value
                    for key, value in identifiers.items()
                    if key != "PBPKKB"
                }
                compacted.append(
                    "CENTER IDS "
                    + "; ".join(
                        f"{key}={value}"
                        for key, value in sorted(identifiers.items())
                    )
                )
            else:
                compacted.append(line)
            continue
        if line.startswith("RELATION TYPES "):
            compacted.append(
                "RELATION TYPES "
                "direction=subject-to-object "
                f"distinct={len(relation_counts)} total={sum(relation_counts.values())}"
            )
            continue
        if line.startswith("RELATION GLOSS "):
            name = line.removeprefix("RELATION GLOSS ").split("=", 1)[0]
            gloss = RELATION_COMPACT_GLOSSES.get(name)
            if gloss:
                compacted.append(
                    f"RELATION GLOSS {name}={gloss}; "
                    f"count={relation_counts.get(name, 0)}"
                )
            else:
                compacted.append(line)
            continue
        if line.startswith("EVENT "):
            prefix, _, context = line.partition(" context=")
            event_head, _, trigger = prefix.partition(" trigger=")
            compacted.append(
                f"{event_head} trigger={trigger[:48]} "
                f"context={context[:40]}"
            )
            continue
        if line.startswith("CITATION "):
            head, _, quote = line.partition(" quote=")
            fields = {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in head.split()[1:]
                if "=" in item
            }
            location = fields.get("location", "")
            compacted.append(
                "CITATION "
                f"source={fields.get('source', '')} "
                f"location_sha256={hashlib.sha256(location.encode()).hexdigest()[:16]} "
                f"quote={quote[:40]}"
            )
            continue
        compacted.append(line)
    return compacted


def prepare_model_input(
    *,
    encoder: Any,
    text: str,
    object_kind: str,
) -> tuple[str, dict[str, Any]]:
    if str(getattr(encoder, "name", "")) == "hashing_embedding":
        return text, {
            "compacted": False,
            "original_token_count": 0,
            "prepared_token_count": 0,
            "dropped_line_count": 0,
            "dropped_line_prefix_counts": {},
            "tokenizer_unavailable_for_baseline": True,
        }
    tokenizer, max_length = encoder_tokenizer_and_limit(encoder)
    original_tokens = tokenizer_length(tokenizer, text)
    if original_tokens <= max_length:
        return text, {
            "compacted": False,
            "original_token_count": original_tokens,
            "prepared_token_count": original_tokens,
            "dropped_line_count": 0,
            "dropped_line_prefix_counts": {},
        }

    if object_kind != "subgraph":
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prepared = tokenizer.decode(
            encoded["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prepared_tokens = tokenizer_length(tokenizer, prepared)
        if prepared_tokens > max_length:
            raise RuntimeError(f"Tokenizer compaction failed for {encoder.name}.")
        return prepared, {
            "compacted": True,
            "original_token_count": original_tokens,
            "prepared_token_count": prepared_tokens,
            "dropped_line_count": 0,
            "dropped_line_prefix_counts": {"token_tail": 1},
        }

    lines = _compact_subgraph_lines(
        [line for line in text.splitlines() if line.strip()]
    )
    essential_prefixes = (
        "CENTER [",
        "CENTER IDENTIFIERS ",
        "CENTER IDS ",
        "CENTER SEMANTICS ",
        "HOP LIMIT ",
        "ONTOLOGY VERSION ",
        "SOURCE FAMILIES ",
        "RELATION TYPES ",
        "RELATION GLOSS ",
    )
    selected_indices: set[int] = {
        index
        for index, line in enumerate(lines)
        if line.startswith(essential_prefixes)
    }
    for prefix in ("EDGE ", "EVENT ", "CITATION "):
        index = next(
            (
                position
                for position, line in enumerate(lines)
                if line.startswith(prefix)
            ),
            None,
        )
        if index is not None:
            selected_indices.add(index)
    prepared = "\n".join(
        line for index, line in enumerate(lines) if index in selected_indices
    )
    if tokenizer_length(tokenizer, prepared) > max_length:
        raise RuntimeError(
            f"Essential subgraph semantics exceed {encoder.name} token budget."
        )

    optional_priority = (
        "EDGE ",
        "EVENT ",
        "CITATION ",
        "RELATION SEARCH TERMS ",
        "NODE TYPES ",
        "NODE ",
    )
    for prefix in optional_priority:
        for index, line in enumerate(lines):
            if index in selected_indices or not line.startswith(prefix):
                continue
            candidate_indices = selected_indices | {index}
            candidate = "\n".join(
                value
                for position, value in enumerate(lines)
                if position in candidate_indices
            )
            if tokenizer_length(tokenizer, candidate) <= max_length:
                selected_indices.add(index)
                prepared = candidate

    missing = []
    for prefix in ("CENTER [", "SOURCE FAMILIES ", "ONTOLOGY VERSION "):
        if not any(line.startswith(prefix) for line in prepared.splitlines()):
            missing.append(prefix)
    for prefix in ("EDGE ", "EVENT ", "CITATION "):
        if any(line.startswith(prefix) for line in lines) and not any(
            line.startswith(prefix) for line in prepared.splitlines()
        ):
            missing.append(prefix)
    relation_names = [
        line.removeprefix("RELATION GLOSS ").split("=", 1)[0]
        for line in lines
        if line.startswith("RELATION GLOSS ")
    ]
    for name in relation_names:
        if f"RELATION GLOSS {name}=" not in prepared:
            missing.append(f"RELATION GLOSS {name}=")
    if missing:
        raise RuntimeError(
            f"Critical subgraph sections were lost for {encoder.name}: {missing}"
        )

    dropped = [
        line for index, line in enumerate(lines) if index not in selected_indices
    ]
    dropped_prefix_counts = Counter(line.split(" ", 1)[0] for line in dropped)
    return prepared, {
        "compacted": True,
        "original_token_count": original_tokens,
        "prepared_token_count": tokenizer_length(tokenizer, prepared),
        "dropped_line_count": len(dropped),
        "dropped_line_prefix_counts": dict(sorted(dropped_prefix_counts.items())),
        "compact_relation_glosses_used": True,
    }


def _rank_metrics(
    scores: Any,
    corpus_ids: list[str],
    positives: set[str],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    import numpy as np

    order = np.argsort(-np.asarray(scores), kind="stable")
    ranked = [corpus_ids[int(index)] for index in order]
    positive_ranks = [index + 1 for index, value in enumerate(ranked) if value in positives]
    metrics: dict[str, float] = {}
    for k in ks:
        found = len(set(ranked[:k]) & positives)
        metrics[f"hit_rate_at_{k}"] = 1.0 if found else 0.0
        metrics[f"recall_at_{k}"] = found / len(positives)
        metrics[f"precision_at_{k}"] = found / k
    first_rank = min(positive_ranks, default=0)
    metrics["reciprocal_rank"] = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in positive_ranks
        if rank <= 10
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(positives), 10) + 1)
    )
    metrics["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    metrics["first_relevant_rank"] = float(first_rank)
    return metrics


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "hit_rate_at_1",
        "hit_rate_at_5",
        "hit_rate_at_10",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "precision_at_1",
        "precision_at_5",
        "precision_at_10",
        "reciprocal_rank",
        "ndcg_at_10",
    )

    def means(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            name: sum(float(row["metrics"][name]) for row in rows) / len(rows)
            for name in metric_names
        } if rows else {name: 0.0 for name in metric_names}

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        by_category[str(row["category"])].append(row)
    category_metrics = {
        category: means(rows)
        for category, rows in sorted(by_category.items())
    }
    overall = means(per_query)
    semantic_categories = [
        category
        for category in category_metrics
        if category != "entity_query"
    ]
    semantic_macro = {
        name: (
            sum(category_metrics[category][name] for category in semantic_categories)
            / len(semantic_categories)
            if semantic_categories
            else 0.0
        )
        for name in metric_names
    }
    ddi_hit = category_metrics.get("ddi_role", {}).get("hit_rate_at_10", 0.0)
    population_hit = category_metrics.get("special_population", {}).get("hit_rate_at_10", 0.0)
    composite = (
        0.35 * semantic_macro["reciprocal_rank"]
        + 0.25 * semantic_macro["ndcg_at_10"]
        + 0.20 * overall["hit_rate_at_10"]
        + 0.10 * ddi_hit
        + 0.10 * population_hit
    )
    return {
        "overall": overall,
        "by_category": category_metrics,
        "semantic_macro": semantic_macro,
        "composite_score": composite,
    }


def _evaluate_dense(
    *,
    encoder: Encoder,
    corpora: dict[int, list[dict[str, Any]]],
    queries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, Any], dict[str, list[str]]]:
    import numpy as np

    started = time.time()
    corpus_vectors: dict[int, Any] = {}
    preparation_audit: list[dict[str, Any]] = []
    query_vectors: dict[int, Any] = {}
    query_positions: dict[int, list[int]] = defaultdict(list)
    for index, query in enumerate(queries):
        query_positions[int(query["target_hop"])].append(index)
    for hop in (0, 1, 2):
        prepared_documents = [
            prepare_model_input(
                encoder=encoder,
                text=str(row["embedding_text"]),
                object_kind="subgraph",
            )
            for row in corpora[hop]
        ]
        preparation_audit.extend(value[1] for value in prepared_documents)
        corpus_vectors[hop] = np.asarray(
            encoder.encode_documents([value[0] for value in prepared_documents]),
            dtype=np.float32,
        )
        positions = query_positions[hop]
        query_vectors[hop] = np.asarray(
            encoder.encode_queries([str(queries[index]["query_text"]) for index in positions]),
            dtype=np.float32,
        )

    per_query: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    for hop in (0, 1, 2):
        corpus_ids = [str(row["subgraph_id"]) for row in corpora[hop]]
        similarities = query_vectors[hop] @ corpus_vectors[hop].T
        for row_index, query_index in enumerate(query_positions[hop]):
            query = queries[query_index]
            metrics = _rank_metrics(
                similarities[row_index],
                corpus_ids,
                set(str(value) for value in query["positive_subgraph_ids"]),
            )
            order = np.argsort(-similarities[row_index], kind="stable")
            rankings[str(query["query_id"])] = [
                corpus_ids[int(index)] for index in order
            ]
            per_query.append(
                {
                    "query_id": query["query_id"],
                    "category": query["category"],
                    "target_hop": hop,
                    "metrics": metrics,
                }
            )
    aggregate = _aggregate(per_query)
    aggregate.update(
        {
            "model": encoder.name,
            "revision": encoder.revision,
            "dimension": encoder.dimension,
            "elapsed_seconds": round(time.time() - started, 3),
            "query_count": len(queries),
            "document_preparation": {
                "serialization_version": MODEL_INPUT_SERIALIZATION_VERSION,
                "record_count": len(preparation_audit),
                "compacted_record_count": sum(
                    bool(row["compacted"]) for row in preparation_audit
                ),
                "maximum_original_tokens": max(
                    (int(row["original_token_count"]) for row in preparation_audit),
                    default=0,
                ),
                "maximum_prepared_tokens": max(
                    (int(row["prepared_token_count"]) for row in preparation_audit),
                    default=0,
                ),
                "dropped_line_count": sum(
                    int(row["dropped_line_count"]) for row in preparation_audit
                ),
            },
            "per_query": sorted(per_query, key=lambda row: str(row["query_id"])),
        }
    )
    return aggregate, corpus_vectors, rankings


def _evaluate_tfidf(
    corpora: dict[int, list[dict[str, Any]]],
    queries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    import numpy as np

    from sklearn.feature_extraction.text import TfidfVectorizer

    started = time.time()
    per_query: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    for hop in (0, 1, 2):
        hop_queries = [query for query in queries if int(query["target_hop"]) == hop]
        corpus_texts = [str(row["embedding_text"]) for row in corpora[hop]]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_features=100_000,
            norm="l2",
        )
        corpus_matrix = vectorizer.fit_transform(corpus_texts)
        query_matrix = vectorizer.transform([str(query["query_text"]) for query in hop_queries])
        similarities = query_matrix @ corpus_matrix.T
        corpus_ids = [str(row["subgraph_id"]) for row in corpora[hop]]
        for index, query in enumerate(hop_queries):
            scores = similarities.getrow(index).toarray()[0]
            metrics = _rank_metrics(
                scores,
                corpus_ids,
                set(str(value) for value in query["positive_subgraph_ids"]),
            )
            order = np.argsort(-scores, kind="stable")
            rankings[str(query["query_id"])] = [
                corpus_ids[int(position)] for position in order
            ]
            per_query.append(
                {
                    "query_id": query["query_id"],
                    "category": query["category"],
                    "target_hop": hop,
                    "metrics": metrics,
                }
            )
    result = _aggregate(per_query)
    result.update(
        {
            "model": "lexical_tfidf",
            "revision": "sklearn-word-bigram-v1",
            "dimension": None,
            "elapsed_seconds": round(time.time() - started, 3),
            "query_count": len(queries),
            "per_query": sorted(per_query, key=lambda row: str(row["query_id"])),
        }
    )
    return result, rankings


def _evaluate_hashing(
    corpora: dict[int, list[dict[str, Any]]],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    from pbpk_kb.storage.vector_index import hash_embedding

    class HashEncoder:
        name = "hashing_embedding"
        revision = "hashing-embedding-v0"
        dimension = 1536

        @staticmethod
        def encode_documents(texts: list[str]) -> Any:
            return np.asarray([hash_embedding(text) for text in texts], dtype=np.float32)

        @staticmethod
        def encode_queries(texts: list[str]) -> Any:
            return np.asarray([hash_embedding(text) for text in texts], dtype=np.float32)

    result, _, _ = _evaluate_dense(
        encoder=HashEncoder(),
        corpora=corpora,
        queries=queries,
    )
    return result


def _evaluate_rrf(
    *,
    dense_result: dict[str, Any],
    dense_rankings: dict[str, list[str]],
    lexical_rankings: dict[str, list[str]],
    queries: list[dict[str, Any]],
    k: int = 60,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        dense_ranking = dense_rankings[query_id]
        lexical_ranking = lexical_rankings[query_id]
        scores: dict[str, float] = defaultdict(float)
        for rank, object_id in enumerate(dense_ranking, 1):
            scores[object_id] += 1.0 / (k + rank)
        for rank, object_id in enumerate(lexical_ranking, 1):
            scores[object_id] += 1.0 / (k + rank)
        ranking = sorted(scores, key=lambda value: (-scores[value], value))
        ranking_scores = [scores[value] for value in ranking]
        metrics = _rank_metrics(
            ranking_scores,
            ranking,
            set(str(value) for value in query["positive_subgraph_ids"]),
        )
        per_query.append(
            {
                "query_id": query_id,
                "category": query["category"],
                "target_hop": query["target_hop"],
                "metrics": metrics,
            }
        )
    result = _aggregate(per_query)
    result.update(
        {
            "model": f"hybrid_rrf::{dense_result['model']}",
            "retrieval_method": "dense_lexical_rrf",
            "dense_model": dense_result["model"],
            "revision": f"{dense_result['revision']};tfidf-word-bigram-v1;rrf-k{k}",
            "dense_revision": dense_result["revision"],
            "dimension": dense_result["dimension"],
            "rrf_k": k,
            "elapsed_seconds": 0.0,
            "query_count": len(queries),
            "per_query": sorted(per_query, key=lambda row: str(row["query_id"])),
        }
    )
    return result


def _model_factory(
    model_name: str,
    *,
    cache_dir: Path,
    device: str,
    batch_size: int,
    query_only: bool = False,
) -> Encoder:
    if model_name == "ncbi/MedCPT":
        return MedCPTEncoder(
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
            query_only=query_only,
        )
    if model_name == "BAAI/bge-base-en-v1.5":
        return SentenceTransformerEncoder(
            name=model_name,
            cache_dir=cache_dir,
            device=device,
            query_prefix="Represent this sentence for searching relevant passages: ",
            batch_size=batch_size,
        )
    if model_name == "Qwen/Qwen3-Embedding-0.6B":
        return SentenceTransformerEncoder(
            name=model_name,
            cache_dir=cache_dir,
            device=device,
            query_prefix=(
                "Instruct: Given a PBPK modeling or biomedical pharmacokinetics query, "
                "retrieve the most relevant provenance-backed knowledge-graph subgraph.\n"
                "Query: "
            ),
            truncate_dimension=768,
            batch_size=min(batch_size, 8),
        )
    return SentenceTransformerEncoder(
        name=model_name,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
    )


def evaluate_models(
    *,
    subgraph_dir: Path,
    benchmark_dir: Path,
    output_dir: Path,
    model_names: list[str],
    cache_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    query_path = benchmark_dir / "locked_retrieval_silver.jsonl"
    acceptance_path = benchmark_dir / "preregistered_acceptance.json"
    if manifest.get("query_sha256") != sha256_file(query_path):
        raise RuntimeError("Locked benchmark query checksum mismatch.")
    if manifest.get("acceptance_sha256") != sha256_file(acceptance_path):
        raise RuntimeError("Preregistered acceptance checksum mismatch.")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    thresholds = acceptance["thresholds"]
    if len(load_jsonl(query_path)) < int(thresholds["minimum_query_count"]):
        raise RuntimeError("Locked benchmark no longer meets its preregistered size.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA embedding run requested but no CUDA device is available.")

    corpora = {
        hop: load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
        for hop in (0, 1, 2)
    }
    queries = load_jsonl(query_path)
    lexical_result, lexical_rankings = _evaluate_tfidf(corpora, queries)
    results = [lexical_result, _evaluate_hashing(corpora, queries)]
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "lexical_tfidf_metrics.json", results[0])
    _atomic_json(output_dir / "hashing_embedding_metrics.json", results[1])

    dense_vectors: dict[str, dict[int, Any]] = {}
    for model_name in model_names:
        print(json.dumps({"loading_model": model_name}), flush=True)
        encoder = _model_factory(
            model_name,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
        )
        result, vectors, dense_rankings = _evaluate_dense(
            encoder=encoder,
            corpora=corpora,
            queries=queries,
        )
        result["retrieval_method"] = "dense_only"
        hybrid_result = _evaluate_rrf(
            dense_result=result,
            dense_rankings=dense_rankings,
            lexical_rankings=lexical_rankings,
            queries=queries,
            k=60,
        )
        results.extend((result, hybrid_result))
        dense_vectors[model_name] = vectors
        safe_name = model_name.replace("/", "__")
        _atomic_json(output_dir / f"{safe_name}_metrics.json", result)
        _atomic_json(output_dir / f"{safe_name}_hybrid_rrf_metrics.json", hybrid_result)
        print(
            json.dumps(
                {
                    "model": model_name,
                    "dense_composite_score": result["composite_score"],
                    "dense_hit_rate_at_10": result["overall"]["hit_rate_at_10"],
                    "hybrid_composite_score": hybrid_result["composite_score"],
                    "hybrid_hit_rate_at_10": hybrid_result["overall"]["hit_rate_at_10"],
                }
            ),
            flush=True,
        )
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baselines = [row for row in results if row["model"] in {"lexical_tfidf", "hashing_embedding"}]
    real_models = [
        row
        for row in results
        if row["model"] not in {"lexical_tfidf", "hashing_embedding"}
    ]
    best_baseline = max(baselines, key=lambda row: float(row["composite_score"]))
    selected = max(real_models, key=lambda row: float(row["composite_score"]))
    category_regressions = {}
    all_categories = sorted(selected["by_category"])
    for category in all_categories:
        best_baseline_hit = max(
            float(row["by_category"].get(category, {}).get("hit_rate_at_10", 0.0))
            for row in baselines
        )
        selected_hit = float(selected["by_category"][category]["hit_rate_at_10"])
        category_regressions[category] = best_baseline_hit - selected_hit

    checks = {
        "query_count_meets_minimum": len(queries) >= int(thresholds["minimum_query_count"]),
        "selected_hit_rate_at_10_meets_threshold": (
            float(selected["overall"]["hit_rate_at_10"])
            >= float(thresholds["minimum_selected_hit_rate_at_10"])
        ),
        "selected_mrr_meets_threshold": (
            float(selected["overall"]["reciprocal_rank"])
            >= float(thresholds["minimum_selected_mrr"])
        ),
        "selected_ndcg_at_10_meets_threshold": (
            float(selected["overall"]["ndcg_at_10"])
            >= float(thresholds["minimum_selected_ndcg_at_10"])
        ),
        "selected_composite_beats_best_baseline": (
            float(selected["composite_score"]) - float(best_baseline["composite_score"])
            >= float(thresholds["minimum_composite_improvement_over_best_baseline"])
        ),
        "no_category_regresses_beyond_limit": (
            max(category_regressions.values(), default=0.0)
            <= float(thresholds["maximum_category_hit_rate_regression_vs_best_baseline"])
        ),
    }
    accepted = all(checks.values())
    comparison = {
        "accepted": accepted,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_level": "silver",
        "benchmark_query_sha256": sha256_file(query_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "checks": checks,
        "thresholds": thresholds,
        "selected_retrieval_method": selected.get("retrieval_method", "dense_only"),
        "selected_model": selected.get("dense_model", selected["model"]),
        "selected_revision": selected.get("dense_revision", selected["revision"]),
        "selected_dimension": selected["dimension"],
        "best_baseline": best_baseline["model"],
        "composite_improvement": (
            float(selected["composite_score"]) - float(best_baseline["composite_score"])
        ),
        "category_hit_rate_regressions": category_regressions,
        "ranking": [
            {
                "model": row["model"],
                "retrieval_method": row.get("retrieval_method", row["model"]),
                "revision": row["revision"],
                "dimension": row["dimension"],
                "composite_score": row["composite_score"],
                "mrr": row["overall"]["reciprocal_rank"],
                "ndcg_at_10": row["overall"]["ndcg_at_10"],
                "hit_rate_at_10": row["overall"]["hit_rate_at_10"],
                "semantic_macro_mrr": row["semantic_macro"]["reciprocal_rank"],
            }
            for row in sorted(results, key=lambda value: -float(value["composite_score"]))
        ],
        "generated_at": utc_now(),
        "limitations": [
            "This is a locked KG-derived silver retrieval benchmark, not clinical recommendation validation.",
            "Model selection is valid only for the current corpus and query distribution until an independent gold set exists.",
        ],
    }
    _atomic_json(output_dir / "embedding_model_comparison.json", comparison)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run the PBPK subgraph embedding benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--subgraph-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--subgraph-dir", type=Path, required=True)
    run.add_argument("--benchmark-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--models",
        default=(
            "ncbi/MedCPT,NeuML/pubmedbert-base-embeddings,"
            "Alibaba-NLP/gte-modernbert-base,BAAI/bge-base-en-v1.5,"
            "Qwen/Qwen3-Embedding-0.6B"
        ),
    )
    run.add_argument("--cache-dir", type=Path, default=recommendation_model_cache_dir())
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = build_locked_benchmark(args.subgraph_dir, args.output_dir)
    else:
        result = evaluate_models(
            subgraph_dir=args.subgraph_dir,
            benchmark_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            model_names=[value.strip() for value in args.models.split(",") if value.strip()],
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("accepted") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
