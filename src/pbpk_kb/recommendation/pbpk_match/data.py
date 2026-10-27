"""Frozen development inputs, leakage boundaries and local embedding features."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CORE = ROOT / "experiments/model_recommendation/frontier_core_145_20260921"
OUTPUT = ROOT / "experiments/model_recommendation/pbpk_match"
FIELDS = ("compounds", "species", "routes", "populations", "endpoints",
          "enzymes", "transporters", "process_types", "formulations",
          "model_structure", "software", "compartments")
TYPES = ("description", "objective", *FIELDS)
VARIANTS = ("original", "reordered", "missing_evidence")
TASKS = ("compound_parameterization_template", "ddi_exposure_prediction",
         "mechanistic_template_recommendation", "plasma_concentration_time_prediction",
         "special_population_prediction")
FOLD_SEED = "pbpk-match-development-compound-folds-v1"


def scoped(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("PBPK-Match path must stay within the project")
    return p


def sha(path):
    h = hashlib.sha256()
    with scoped(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    p = scoped(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(p)


def read_rows(path):
    with scoped(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(values):
    return sorted({re.sub(r"\s+", " ", str(v).strip().casefold()) for v in values if str(v).strip()})


def fields_for(record, variant="original", candidate=False):
    features = record.get("features", {})
    if not candidate and variant == "missing_evidence":
        features = record["variants"][variant]["features"]
    fields = {key: normalize(features.get(key, [])) for key in FIELDS}
    fields["compounds"] = normalize(record.get("compounds", fields["compounds"]))
    # No input ID, provenance hash, legacy split, outcome or judge output is encoded.
    return fields


def node_specs(record, variant="original", candidate=False):
    fields = fields_for(record, variant, candidate)
    parts = [f"{key}: {', '.join(values)}" for key, values in fields.items() if values]
    if not candidate:
        parts.insert(0, "objective: " + record["task_type"])
    nodes = [(0, "; ".join(parts) or "PBPK model with unspecified metadata")]
    if not candidate:
        nodes.append((1, "PBPK intended task: " + record["task_type"]))
    for key, values in fields.items():
        nodes.extend((TYPES.index(key), f"PBPK {key}: {v}") for v in values)
    return nodes


def evidence_hashes(q):
    return {e["evidence_text_sha256"] for e in q.get("provenance", {}).get("evidence", [])
            if e.get("evidence_text_sha256")}


def prepare_snapshot(output=OUTPUT, core=CORE):
    """Freeze only the nine development compounds' labels at a byte boundary.

    The mixed live journal is parsed to export development rows only. Test label
    values are not exported, counted, or used for fit/selection. Input-only
    duplicate hashes prevent exact source text overlap in fitting.
    """
    output, core = scoped(output), scoped(core)
    target = output / "development_snapshot.json"
    if target.exists():
        saved = json.loads(target.read_text())
        for name, digest in saved["input_sha256"].items():
            if sha(core / name) != digest:
                raise ValueError(f"Frozen input changed: {name}")
        return saved
    queries = read_rows(core / "inputs/queries.jsonl")
    candidates = read_rows(core / "inputs/candidates.jsonl")
    selection = json.loads((core / "selection_manifest.json").read_text())
    dev = [q for q in queries if selection["compound_split"][q["compound_id"]] == "validation"]
    test = [q for q in queries if selection["compound_split"][q["compound_id"]] == "test"]
    assert len(dev) == 43 and len(test) == 102 and len(candidates) == 280
    assert not {q["compound_id"] for q in dev} & {q["compound_id"] for q in test}
    dev_ids = {q["query_id"] for q in dev}
    candidate_ids = {c["candidate_id"] for c in candidates}
    assessment = core / "assessment_repaired_v2"
    protocol_hash = sha(assessment / "judge_protocol.json")
    label_file = assessment / "judgments.jsonl"
    size = label_file.stat().st_size
    with label_file.open("rb") as f:
        frozen_bytes = f.read(size)
    # The live writer may have been between writes; ignore an unfinished last row.
    frozen_bytes = frozen_bytes[:frozen_bytes.rfind(b"\n") + 1]
    records = {}
    for line in frozen_bytes.splitlines():
        row = json.loads(line)
        if row.get("query_id") not in dev_ids or row.get("judge") != "primary":
            continue
        if row.get("protocol_sha256") != protocol_hash:
            raise ValueError("Development label protocol differs from current frozen judge protocol")
        if row["candidate_id"] not in candidate_ids:
            raise ValueError("Unknown candidate in development label")
        key = (row["query_id"], row["candidate_id"])
        clean = {k: row.get(k) for k in ("query_id", "candidate_id", "grade",
                  "documented_applicability", "status", "source_packet_sha256", "label_source")}
        if key in records and records[key] != clean:
            raise ValueError("Conflicting development labels; explicit resolution required")
        records[key] = clean
    test_hashes = set().union(*(evidence_hashes(q) for q in test))
    overlap = [q["query_id"] for q in dev if evidence_hashes(q) & test_hashes]
    compounds = sorted({q["compound_id"] for q in dev},
                       key=lambda x: hashlib.sha256((FOLD_SEED + x).encode()).hexdigest())
    folds = {cid: i % 3 for i, cid in enumerate(compounds)}
    snapshot = {
        "schema": "pbpk-match-development-snapshot-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development only; not the locked 102-query quality test",
        "input_sha256": {p: sha(core / p) for p in ("inputs/queries.jsonl", "inputs/candidates.jsonl", "selection_manifest.json")},
        "judge_protocol_sha256": protocol_hash, "label_snapshot_prefix_bytes": len(frozen_bytes),
        "label_snapshot_prefix_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "development_query_ids": sorted(dev_ids),
        "development_compounds": compounds, "fold_by_compound": folds,
        "test_query_ids_excluded": [q["query_id"] for q in test],
        "exact_test_source_overlap_excluded_from_fitting": overlap,
        "labels": sorted(records.values(), key=lambda r: (r["query_id"], r["candidate_id"])),
        "status_counts": dict(Counter(r["status"] for r in records.values())),
        "limitations": ["Partial primary judgments; candidate coverage is incomplete.",
                        "Nine development compounds, not independent pair samples.",
                        "Judge development previously used both partition inputs; not a fully blind benchmark.",
                        "Exact text deduplication does not establish independent studies."]}
    write_json(target, snapshot)
    return snapshot


def build_data(output=OUTPUT, core=CORE, device="cuda:0", batch_size=8):
    """Build reproducible outcome-free metadata features using local frozen Qwen."""
    output, core = scoped(output), scoped(core)
    snapshot = prepare_snapshot(output, core)
    all_q = read_rows(core / "inputs/queries.jsonl")
    allowed = set(snapshot["development_query_ids"])
    queries = [q for q in all_q if q["query_id"] in allowed]
    candidates = read_rows(core / "inputs/candidates.jsonl")
    qnodes = [[node_specs(q, v) for q in queries] for v in VARIANTS]
    cnodes = [node_specs(c, candidate=True) for c in candidates]
    texts = sorted({t for variant in qnodes for row in variant for _, t in row} |
                   {t for row in cnodes for _, t in row})
    registry = json.loads((ROOT / "experiments/model_recommendation/frontier_comparison_20260914/results/qwen3_embedding_06b/weights.json").read_text())
    model_path = scoped(registry["path"])
    token = hashlib.sha256(json.dumps({"texts": texts, "revision": registry["revision"],
                   "encoding": "last-token-l2-typed-fields-v1"}, sort_keys=True).encode()).hexdigest()
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / "frozen_embeddings.npz"
    cache_meta = cache / "embedding_manifest.json"
    if cache_file.exists() and cache_meta.exists():
        manifest = json.loads(cache_meta.read_text())
        if manifest["input_key"] != token or manifest["array_sha256"] != sha(cache_file):
            raise ValueError("Embedding cache differs; use a new explicit output directory")
        vectors = np.load(cache_file, allow_pickle=False)["embeddings"].astype(np.float32)
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("GPU embedding requested; no implicit CPU substitution")
        if torch.cuda.mem_get_info(device)[0] < 6 * 1024**3:
            raise RuntimeError("Insufficient free GPU memory for isolated embedding; existing jobs untouched")
        torch.cuda.set_per_process_memory_fraction(5 * 1024**3 / torch.cuda.mem_get_info(device)[1], device)
        options = dict(local_files_only=True, token=False, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side="left", **options)
        lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
        if max(lengths) > 4096:
            raise ValueError("Canonical input exceeds 4096 tokens; do not silently truncate")
        model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16,
                      attn_implementation="sdpa", **options).to(device).eval()
        ordered = sorted(range(len(texts)), key=lambda i: lengths[i])
        results = {}
        size = max(1, batch_size)
        pos = 0
        while pos < len(ordered):
            ids = ordered[pos:pos+size]
            try:
                enc = tokenizer([texts[i] for i in ids], padding=True, truncation=False,
                                return_tensors="pt").to(device)
                with torch.inference_mode():
                    hidden = model(**enc, use_cache=False).last_hidden_state[:, -1].float()
                    values = torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy()
                results.update(zip(ids, values))
                pos += len(ids)
                if pos % 100 < size or pos == len(texts):
                    print(f"Frozen embeddings {pos}/{len(texts)}", flush=True)
                del enc, hidden
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                size = max(1, size // 2)
                torch.cuda.empty_cache()
        vectors = np.stack([results[i] for i in range(len(texts))])
        np.savez_compressed(cache_file, embeddings=vectors.astype(np.float16))
        vectors = vectors.astype(np.float16).astype(np.float32)
        manifest = {"input_key": token, "repo": registry["repo"], "revision": registry["revision"],
                    "text_count": len(texts), "maximum_input_tokens": max(lengths), "truncated_inputs": 0,
                    "dimension": vectors.shape[1], "array_sha256": sha(cache_file), "device": device,
                    "max_allocated_gib": torch.cuda.max_memory_allocated(device)/1024**3,
                    "gpu_exclusive": False, "trainable_encoder": False}
        write_json(cache_meta, manifest)
        del model
        torch.cuda.empty_cache()
    lookup = {t: vectors[i] for i, t in enumerate(texts)}
    dim = vectors.shape[-1]
    nq = max(len(row) for variant in qnodes for row in variant)
    nc = max(len(row) for row in cnodes)
    qe = np.zeros((3, len(queries), nq, dim), np.float32)
    qt = np.zeros(qe.shape[:-1], np.int64); qm = np.zeros(qe.shape[:-1], bool)
    ce = np.zeros((len(candidates), nc, dim), np.float32)
    ct = np.zeros(ce.shape[:-1], np.int64); cm = np.zeros(ce.shape[:-1], bool)
    for v, rows in enumerate(qnodes):
        for j, nodes in enumerate(rows):
            for k, (kind, text) in enumerate(nodes):
                qe[v,j,k] = lookup[text]; qt[v,j,k] = kind; qm[v,j,k] = True
    for j, nodes in enumerate(cnodes):
        for k, (kind, text) in enumerate(nodes):
            ce[j,k] = lookup[text]; ct[j,k] = kind; cm[j,k] = True
    feature_names = [f"{field}.{feature}" for field in FIELDS for feature in
                     ("query_present", "candidate_present", "jaccard", "query_coverage", "listed_disjoint")]
    feature_names.append("canonical_description_cosine")
    pair = np.zeros((3,len(queries),len(candidates),len(feature_names)),np.float32)
    cf = [fields_for(c,candidate=True) for c in candidates]
    for v, variant in enumerate(VARIANTS):
        for qi,q in enumerate(queries):
            qf = fields_for(q,variant)
            for ci,c in enumerate(candidates):
                values=[]
                for field in FIELDS:
                    a,b=set(qf[field]),set(cf[ci][field]);shared=len(a&b)
                    values.extend([bool(a),bool(b), shared/max(1,len(a|b)),shared/max(1,len(a)),bool(a and b and not shared)])
                values.append(float(qe[v,qi,0] @ ce[ci,0]))
                pair[v,qi,ci]=values
    grades=np.full((len(queries),len(candidates)),-1,np.int64)
    app=np.full_like(grades,-1)
    qidx={q['query_id']:i for i,q in enumerate(queries)}; cidx={c['candidate_id']:i for i,c in enumerate(candidates)}
    for r in snapshot['labels']:
        i,j=qidx[r['query_id']],cidx[r['candidate_id']]
        if r['status'] in ('invalid_output',):
            continue
        g=r.get('grade');a=r.get('documented_applicability')
        if g is not None and r['status']=='judged':
            if g not in (0,1,2,3):raise ValueError('Invalid grade')
            grades[i,j]=int(g)
        if r['status'] in ('judged','insufficient_evidence','source_conflict'):
            app[i,j]=0 if a is True else 1 if a is False else 2
    # Equal metadata must stay equal; packet/family IDs serve sampling/audit only.
    packet_groups=[hashlib.sha256(json.dumps(fields_for(c,candidate=True),sort_keys=True).encode()).hexdigest() for c in candidates]
    data=dict(q_embeddings=qe,q_types=qt,q_mask=qm,c_embeddings=ce,c_types=ct,c_mask=cm,
              pair_features=pair, task_ids=np.asarray([TASKS.index(q['task_type']) for q in queries]),
              grades=grades,applicability=app,query_records=queries,candidate_records=candidates,
              candidate_packet_groups=packet_groups, variant_names=list(VARIANTS), num_types=len(TYPES),
              num_tasks=len(TASKS),feature_names=feature_names,snapshot=snapshot,embedding_manifest=manifest)
    assert np.array_equal(qe[0],qe[1]) and np.array_equal(pair[0],pair[1])
    return data


def fitting_indices(data, indices, excluded_indices=(), exclude_locked_overlap=True):
    queries=data['query_records']
    hashes=set().union(*(evidence_hashes(queries[i]) for i in excluded_indices)) if excluded_indices else set()
    forbidden=set(data['snapshot']['exact_test_source_overlap_excluded_from_fitting']) if exclude_locked_overlap else set()
    return [i for i in indices if queries[i]['query_id'] not in forbidden and not evidence_hashes(queries[i]) & hashes]
