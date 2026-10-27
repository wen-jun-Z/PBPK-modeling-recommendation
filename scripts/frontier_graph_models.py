"""Inference adapters for pinned official GFM-RAG and G-reasoner weights.

The author's graph processing, GNN forward pass and document ranker are used.
Project adaptations: typed metadata graph import, exact structured-entity seeds,
local SentenceTransformer embedding backend, and the author's native PyG kernels.
No model downloads, API calls, scientific labels or training happen in this file.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_COMMIT = "b3e43219c49929da505244df4f0ba54c0947e4c4"
REPO = ROOT / "dependencies/frontier_gfm" / REPO_COMMIT
EXTRA_PACKAGES = ROOT / "runtime/frontier-gfm-packages"


def _local(path):
    result = Path(path).resolve()
    result.relative_to(ROOT)
    return result


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _entity(node):
    return f"{node['node_type']}: {' '.join(str(node['label']).casefold().replace('_', ' ').split())}"


def query_record_variant(query, variant="original"):
    """Apply a recorded input perturbation, including its removed graph evidence."""
    if variant == "original":
        return query
    patch = query["variants"][variant]
    result = {**query, "text": patch["text"]}
    if "features" not in patch:
        return result
    result["features"] = patch["features"]
    feature_types = {"Enzyme": "enzymes", "Transporter": "transporters", "ADMEProcess": "process_types",
                     "Species": "species", "Route": "routes", "Population": "populations", "Endpoint": "endpoints"}
    allowed = {t: {_entity({"node_type": t, "label": label}) for label in patch["features"].get(key, [])}
               for t, key in feature_types.items()}
    nodes = [node for node in query["graph"]["nodes"]
             if node["node_type"] not in allowed or _entity(node) in allowed[node["node_type"]]]
    node_ids = {node["node_id"] for node in nodes}
    result["graph"] = {**query["graph"], "nodes": nodes,
                       "edges": [edge for edge in query["graph"]["edges"]
                                 if edge["source_id"] in node_ids and edge["target_id"] in node_ids]}
    result["input_evidence_ids"] = patch.get("retained_evidence_ids", query.get("input_evidence_ids", []))
    return result


def prepare_graph(candidates, cache_dir):
    """Import directed typed edges; map each candidate to a retrievable document."""
    fingerprint = _hash([{k: c[k] for k in ("candidate_id", "family_name", "text", "graph")} for c in candidates])
    data_name = f"pbpk_{fingerprint[:20]}"
    directory = _local(cache_dir) / data_name / "processed/stage1"
    directory.mkdir(parents=True, exist_ok=True)
    nodes, edges, document_ids = {}, set(), {}
    original_edges = 0
    for ordinal, candidate in enumerate(candidates):
        cid = candidate["candidate_id"]
        # Readable, unique names; registry UUIDs do not enter graph embeddings.
        document = f"PBPK document {ordinal + 1}: {candidate['family_name']}"
        model = f"PBPK model {ordinal + 1}: {candidate['family_name']}"
        document_ids[cid] = document
        nodes[document] = {"name": document, "type": "document", "attributes": {"content": candidate["text"]}}
        nodes[model] = {"name": model, "type": "entity", "attributes": {"pbpk_type": "ModelVersion", "description": candidate["family_name"]}}
        local_mapping = {}
        for node in candidate["graph"]["nodes"]:
            if node["node_type"] == "ModelVersion" or node["node_id"].endswith(":root"):
                local_mapping[node["node_id"]] = model
                continue
            key = _entity(node)
            nodes.setdefault(key, {"name": key, "type": "entity", "attributes": {"pbpk_type": node["node_type"], "label": node["label"]}})
            local_mapping[node["node_id"]] = key
        for edge in candidate["graph"]["edges"]:
            source, target = local_mapping[edge["source_id"]], local_mapping[edge["target_id"]]
            # Preserve the original direction and predicate. The author dataset
            # adds explicit inverse relation types during its normal processing.
            edges.add((source, edge["predicate"], target))
            original_edges += 1
        # GFM-RAG v1 needs entity-to-document membership separately from its KG.
        for member in set(local_mapping.values()):
            edges.add((member, "mentioned in PBPK document", document))
    relations = sorted({r for _, r, _ in edges})
    files = {
        "nodes.csv": (["name", "type", "attributes"], [nodes[n] for n in sorted(nodes)]),
        "relations.csv": (["name", "attributes"], [{"name": r, "attributes": {}} for r in relations]),
        "edges.csv": (["source", "relation", "target", "attributes"], [{"source": s, "relation": r, "target": t, "attributes": {}} for s, r, t in sorted(edges)]),
    }
    for name, (fields, rows) in files.items():
        path = directory / name
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    # Upstream graph-only retrieval still reads the document corpus in
    # load_qa_data(). Provide actual sanitized documents, with no QA labels.
    raw_directory = directory.parents[1] / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    document_path = raw_directory / "documents.json"
    if not document_path.exists():
        document_path.write_text(json.dumps({document_ids[c["candidate_id"]]: c["text"] for c in candidates},
                                           ensure_ascii=False, indent=2) + "\n")
    return {"data_name": data_name, "fingerprint": fingerprint, "document_ids": document_ids,
            "node_count": len(nodes), "edge_count_before_author_inverse_edges": len(edges),
            "original_candidate_edges": original_edges, "relation_count_before_inverses": len(relations),
            "entity_keys": sorted(n for n, v in nodes.items() if v["type"] == "entity"),
            "schema": "official nodes.csv/relations.csv/edges.csv"}


class LocalTextEncoder:
    """Same checkpoint prompts, dimensions and normalization using local weights."""
    _instances = {}

    def __init__(self, text_emb_model_name, normalize=False, batch_size=32,
                 query_instruct=None, passage_instruct=None, truncate_dim=None,
                 model_kwargs=None, device="cuda:0", **unused):
        from sentence_transformers import SentenceTransformer
        import torch
        self.path = str(_local(text_emb_model_name))
        self.normalize, self.batch_size = normalize, batch_size
        self.query_instruct, self.passage_instruct = query_instruct, passage_instruct
        self.truncate_dim, self.device = truncate_dim, device
        key = (self.path, device)
        if key not in self._instances:
            actual_kwargs = dict(model_kwargs or {})
            config = json.loads((Path(self.path) / "config.json").read_text())
            if str(device).startswith("cuda") and config.get("model_type") == "qwen3":
                actual_kwargs.setdefault("torch_dtype", torch.bfloat16)
            self._instances[key] = SentenceTransformer(self.path, device=device,
                local_files_only=True, trust_remote_code=False, model_kwargs=actual_kwargs)
        self.model = self._instances[key]

    def encode(self, text, is_query=False, show_progress_bar=True):
        import torch
        if not text:
            return torch.empty((0, self.truncate_dim or self.model.get_sentence_embedding_dimension()))
        prompt = self.query_instruct if is_query else self.passage_instruct
        with torch.no_grad():
            values = self.model.encode(text, prompt=prompt or "", batch_size=self.batch_size,
                normalize_embeddings=self.normalize, convert_to_tensor=True,
                show_progress_bar=show_progress_bar, device=self.device).float()
        if self.truncate_dim and values.shape[1] != self.truncate_dim:
            raise ValueError("Embedding dimension differs from graph checkpoint; substitution is not allowed")
        return values.cpu()


def _namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
    return module


def load_official_inference():
    """Load official inference modules without eager optional LLM/trainer imports.

    Upstream __init__ eagerly imports vLLM, NVEmbed, trainers and many unrelated
    optional backends. Namespace loading skips those initializers, while the
    author's GNN, datasets, propagation and ranker source files stay unchanged.
    """
    if str(EXTRA_PACKAGES) not in sys.path:
        sys.path.insert(0, str(EXTRA_PACKAGES))
    if getattr(sys.modules.get("gfmrag"), "_pbpk_inference_loaded", False):
        return
    for name in ["gfmrag", "gfmrag.models", "gfmrag.utils", "gfmrag.text_emb_models", "gfmrag.graph_index_datasets"]:
        _namespace(name, REPO.joinpath(*name.split(".")))
    setup = importlib.import_module("gfmrag.utils.setup_training")
    sys.modules["gfmrag.utils"].get_rank = setup.get_rank
    base_text = importlib.import_module("gfmrag.text_emb_models.base_model")
    text_package = sys.modules["gfmrag.text_emb_models"]
    text_package.BaseTextEmbModel = base_text.BaseTextEmbModel
    text_package.PBPKLocalTextEncoder = LocalTextEncoder
    for name in ["graph_index_dataset", "graph_index_dataset_v1"]:
        module = importlib.import_module(f"gfmrag.graph_index_datasets.{name}")
        cls_name = "GraphIndexDatasetV1" if name.endswith("_v1") else "GraphIndexDataset"
        setattr(sys.modules["gfmrag.graph_index_datasets"], cls_name, getattr(module, cls_name))
    importlib.import_module("gfmrag.models.gfm_reasoner")
    importlib.import_module("gfmrag.models.gfm_rag_v1")
    sys.modules["gfmrag"]._pbpk_inference_loaded = True


def _native_message_path(module, args, kwargs):
    """Select the upstream native PyG branch, without computing parameter grads."""
    positional = list(args)
    if len(positional) >= 7:
        positional[6] = positional[6].detach().requires_grad_(True)
    else:
        kwargs["edge_weight"] = kwargs["edge_weight"].detach().requires_grad_(True)
    return tuple(positional), kwargs


class FrontierGraphScorer:
    def __init__(self, model_path, embedding_path, cache_dir, device="cuda:0", batch_size=4,
                 embedding_batch_size=16, ranker_override=None):
        if ranker_override not in (None, "idf_continuous"):
            raise ValueError("ranker_override must be None or 'idf_continuous'")
        self.model_path, self.embedding_path = _local(model_path), _local(embedding_path)
        self.cache_dir = _local(cache_dir)
        self.device, self.batch_size = device, batch_size
        self.embedding_batch_size = embedding_batch_size
        self.ranker_override = ranker_override
        self.model = None
        self._text_to_query = {}
        self._context_candidates = None
        self._graph_fingerprint = None
        self._graph_info = None
        self._graph_candidates = None
        self._dataset = None
        self._missing_seed_query_ids = set()
        self._scored_query_rows = 0
        self._total_score_seconds = 0.0
        self.metadata = {"official_repo_commit": REPO_COMMIT,
            "adaptations": ["directed typed PBPK metadata graph", "exact structured query entity seeds",
                            "local SentenceTransformer backend with checkpoint prompts and normalization",
                            "upstream native PyG message/aggregate path"],
            "fine_tuned_on_pbpk": False, "requires_external_api": False,
            "ranker_override": ranker_override,
            "implementation_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__).resolve(),
                             REPO / "gfmrag/models/gfm_rag_v1/rankers.py",
                             REPO / "gfmrag/models/ultra/layers.py")
            }}

    def prepare_context(self, queries, candidates):
        """Optional per-query runner interface; does not load a model or use GPU."""
        self._context_candidates = candidates
        self._graph_info = prepare_graph(candidates, self.cache_dir)
        self._graph_candidates = candidates
        for query in queries:
            for variant in ["original", *query.get("variants", {})]:
                record = query_record_variant(query, variant)
                previous = self._text_to_query.get(record["text"])
                if previous is not None:
                    previous_entities = {_entity(n) for n in previous["graph"]["nodes"] if n["node_type"] != "ModelingRequest"}
                    current_entities = {_entity(n) for n in record["graph"]["nodes"] if n["node_type"] != "ModelingRequest"}
                    if previous_entities != current_entities:
                        raise ValueError("Identical query text has different structured graph seeds; use batched score()")
                self._text_to_query[record["text"]] = record
        return self

    def score_query(self, text, docs):
        if self._context_candidates is None or text not in self._text_to_query:
            raise ValueError("Call prepare_context with the frozen original queries first")
        if list(docs) != [c["text"] for c in self._context_candidates]:
            raise ValueError("Graph candidate documents/order differ from prepared context")
        return self.score([self._text_to_query[text]], self._context_candidates)[0]

    def score(self, queries, candidates, progress_callback=None):
        import numpy as np
        import torch
        started = time.perf_counter()
        load_official_inference()
        from hydra.utils import get_class, instantiate
        from omegaconf import OmegaConf
        config = json.loads((self.model_path / "config.json").read_text())
        if self.model is None:
            default_ranker = config["model_config"].get("ranker")
            if self.ranker_override is not None and config["model_config"].get("_target_") != "gfmrag.models.gfm_rag_v1.GNNRetriever":
                raise ValueError("The continuous IDF ranker override is only supported for the official GFM-RAG v1 retriever")
            self.model = instantiate(config["model_config"])
            state = torch.load(self.model_path / "model.pth", map_location="cpu", weights_only=True)
            self.model.load_state_dict(state["model"], strict=True)
            self.metadata["checkpoint_default_ranker"] = default_ranker
            self.metadata["effective_ranker"] = default_ranker
            if self.ranker_override == "idf_continuous":
                from gfmrag.models.gfm_rag_v1.rankers import IDFWeightedRanker
                self.model.ranker = IDFWeightedRanker()
                self.metadata["effective_ranker"] = {
                    "_target_": "gfmrag.models.gfm_rag_v1.rankers.IDFWeightedRanker"}
                self.metadata["adaptations"].append(
                    "official continuous IDF document ranker replaces checkpoint TopK ranker; separate declared configuration")
                self.metadata["ranker_override_semantics"] = (
                    "Official parameter-free IDFWeightedRanker aggregates unchanged entity scores; "
                    "GNN checkpoint, graph, encoder and query seeds are unchanged")
            self.model.eval().to(self.device)
            if str(self.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            # Official layers already implement this identical mathematical path.
            # The custom fused extension otherwise requires a separate CUDA toolkit.
            for layer in self.model.entity_model.layers:
                layer.register_forward_pre_hook(_native_message_path, with_kwargs=True)
        if self._graph_candidates is not candidates or self._graph_info is None:
            self._graph_info = prepare_graph(candidates, self.cache_dir)
            self._graph_candidates = candidates
        graph_info = self._graph_info
        dataset_cfg = dict(config["dataset_config"])
        dataset_class = get_class("gfmrag.graph_index_datasets." + dataset_cfg.pop("class_name"))
        text_cfg = dict(dataset_cfg["text_emb_model_cfgs"])
        expected_encoder = text_cfg["text_emb_model_name"]
        encoder_config = json.loads((self.embedding_path / "config.json").read_text())
        dimension = encoder_config.get("hidden_size", encoder_config.get("d_model"))
        if dimension != config["model_config"]["feat_dim"]:
            raise ValueError(f"Wrong text encoder dimension: {dimension}; expected {config['model_config']['feat_dim']}")
        if ("Qwen3" in expected_encoder) != (encoder_config.get("model_type") == "qwen3"):
            raise ValueError("Text encoder architecture differs from official checkpoint")
        if expected_encoder == "sentence-transformers/all-mpnet-base-v2" and encoder_config.get("model_type") != "mpnet":
            raise ValueError("GFM-RAG-8M requires the author's all-mpnet-base-v2 encoder")
        text_cfg.update({"_target_": "gfmrag.text_emb_models.PBPKLocalTextEncoder",
                         "text_emb_model_name": str(self.embedding_path), "device": self.device,
                         "batch_size": self.embedding_batch_size})
        if str(self.device).startswith("cuda") and encoder_config.get("model_type") == "qwen3":
            # Include dtype in the author's graph-cache fingerprint as well.
            text_cfg["model_kwargs"] = {**(text_cfg.get("model_kwargs") or {}), "torch_dtype": "bfloat16"}
        dataset_cfg["text_emb_model_cfgs"] = OmegaConf.create(text_cfg)
        if self._graph_fingerprint != graph_info["fingerprint"]:
            self._dataset = dataset_class(root=str(self.cache_dir), data_name=graph_info["data_name"], **dataset_cfg)
            self._dataset.graph = self._dataset.graph.to(self.device)
            self._graph_fingerprint = graph_info["fingerprint"]
        dataset = self._dataset
        graph = dataset.graph
        encoder = instantiate(dataset_cfg["text_emb_model_cfgs"])
        embeddings = encoder.encode([q["text"] for q in queries], is_query=True, show_progress_bar=False)
        document_indices = torch.tensor([dataset.node2id[graph_info["document_ids"][c["candidate_id"]]] for c in candidates], device=self.device)
        masks = torch.zeros((len(queries), graph.num_nodes), dtype=torch.float32)
        missing = []
        for i, query in enumerate(queries):
            for node in query["graph"]["nodes"]:
                if node["node_type"] == "ModelingRequest":
                    continue
                key = _entity(node)
                if key in dataset.node2id:
                    masks[i, dataset.node2id[key]] = 1
            if not masks[i].any():
                missing.append(query["query_id"])
        # Missing entity seeds remain zero, as in author's structured start masks.
        # They are recorded explicitly; no rule score or fabricated seed is added.
        result = np.empty((len(queries), len(candidates)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(queries), self.batch_size):
                end = min(start + self.batch_size, len(queries))
                pred = self.model(graph, {"question_embeddings": embeddings[start:end].to(self.device),
                                         "start_nodes_mask": masks[start:end].to(self.device)})
                values = pred[:, document_indices].float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Official graph retriever returned nonfinite scores")
                result[start:end] = values
                if progress_callback:
                    progress_callback(end, len(queries))
        self.metadata.update({k: v for k, v in graph_info.items() if k not in {"entity_keys", "document_ids"}})
        self._missing_seed_query_ids.update(missing)
        self._scored_query_rows += len(queries)
        elapsed = time.perf_counter() - started
        self._total_score_seconds += elapsed
        self.metadata.update({"checkpoint_text_encoder": expected_encoder,
                              "encoder_dtype": str(next(encoder.model.parameters()).dtype),
                              "exact_seed_missing_query_ids": sorted(self._missing_seed_query_ids),
                              "query_count": len(queries), "candidate_count": len(candidates),
                              "scored_query_rows_including_variants": self._scored_query_rows,
                              "last_score_seconds_including_load_and_index": elapsed,
                              "total_score_seconds_including_load_and_index": self._total_score_seconds,
                              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if str(self.device).startswith("cuda") else None,
                              "score_semantics": "official graph document score, not a calibrated probability"})
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--inputs", default=str(ROOT / "experiments/model_recommendation/frontier_comparison_20260914/inputs"))
    parser.add_argument("--cache-dir", default=str(ROOT / "runtime/cache/frontier_graph_index_20260914"))
    args = parser.parse_args()
    if args.check_imports:
        load_official_inference()
        print(json.dumps({"official_inference_imports": "passed", "commit": REPO_COMMIT}))
    if args.prepare_only:
        candidates = [json.loads(line) for line in (_local(args.inputs) / "candidates.jsonl").read_text().splitlines()]
        info = prepare_graph(candidates, args.cache_dir)
        print(json.dumps({k: v for k, v in info.items() if k not in {"entity_keys", "document_ids"}}, indent=2))
