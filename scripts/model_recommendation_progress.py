#!/usr/bin/env python3
"""Read-only terminal progress for the frozen 145-query recommendation run.

Uses only the standard library. No runner imports, GPU initialization, file writes,
process signals, or inspection of processes not identified by this project's PIDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/model_recommendation/frontier_core_145_20260921"
ASSESS = OUT / "assessment"
QUAL = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_citation_complete_v8"
RUN = ROOT / "runtime/deployment/frontier_core_145"
QUAL_RUN = ROOT / "runtime/deployment/frontier_scoring_reviewed_v8"
_SHA_CACHE = {}
REPAIR_PROFILE = "core145-repair-v1"
REPAIR_LAYOUTS = (("core145-repair-v2", "_v2"), (REPAIR_PROFILE, ""))
REPAIR_PROFILES = {profile for profile, _ in REPAIR_LAYOUTS}


def available_repair_layout():
    """Newest recognized frozen repair takes precedence over older runs."""
    for profile, suffix in REPAIR_LAYOUTS:
        repaired = OUT / ("assessment_repaired" + suffix)
        if read_json(repaired / "core_plan.json").get("version") == profile:
            return profile, suffix
    return None


def select_layout():
    """Select only an explicitly frozen project repair plan; never import runners."""
    global ASSESS, QUAL, RUN, QUAL_RUN
    layout = available_repair_layout()
    if layout:
        _, suffix = layout
        ASSESS, QUAL = OUT / ("assessment_repaired" + suffix), OUT / ("qualification_repaired" + suffix)
        RUN = ROOT / ("runtime/deployment/frontier_core_145_repair" + suffix)
        QUAL_RUN = RUN / "qualification"


def scoped(path):
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Refusing to read outside this project")
    return path


def read_json(path):
    try:
        return json.loads(scoped(path).read_text())
    except FileNotFoundError:
        return {}


def file_sha(path):
    try:
        stat = scoped(path).stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if path not in _SHA_CACHE or _SHA_CACHE[path][0] != signature:
            _SHA_CACHE[path] = (signature, hashlib.sha256(path.read_bytes()).hexdigest())
        return _SHA_CACHE[path][1]
    except FileNotFoundError:
        return None


def ids(path, field):
    with scoped(path).open() as handle:
        return {json.loads(line)[field] for line in handle if line.strip()}


def process_alive(saved):
    """Inspect only a saved PID after matching current UID and project cwd."""
    try:
        pid = int(saved["pid"])
        if pid <= 0:
            return False
        proc = Path("/proc") / str(pid)
        if proc.stat().st_uid != os.getuid() or (proc / "cwd").resolve() != ROOT:
            return False
        # comm may contain spaces; fields after the last ')' begin at field 3.
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == str(saved["start_ticks"])
    except (OSError, ValueError, KeyError, IndexError):
        return False


class RecordCounter:
    """Tail committed JSON lines incrementally; count each valid identity once."""
    def __init__(self, path, key_fields, allowed, status_field, expected_judge=None, expected_fields=None):
        self.path, self.key_fields = path, key_fields
        self.allowed, self.status_field = allowed, status_field
        self.expected_judge = expected_judge
        self.expected_fields = expected_fields or {}
        self.reset()

    def reset(self):
        self.offset, self.inode, self.binding = 0, None, None
        self.seen, self.statuses = set(), Counter()
        self.duplicates = self.errors = self.tail_bytes = 0
        self.last_at = None
        self.origins = Counter()

    def update(self, protocol_hash):
        try:
            stat = scoped(self.path).stat()
        except FileNotFoundError:
            self.reset()
            return self.summary()
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.inode or stat.st_size < self.offset or self.binding != protocol_hash:
            self.reset()
            self.inode, self.binding = identity, protocol_hash
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            remaining = stat.st_size - self.offset
            self.tail_bytes = 0
            while remaining:
                # Read only bytes present at this snapshot; an in-flight tail waits.
                raw = handle.readline(min(remaining, 16 * 1024 * 1024))
                if not raw or not raw.endswith(b"\n"):
                    self.tail_bytes = remaining
                    break
                remaining -= len(raw)
                self.offset += len(raw)
                self.tail_bytes = 0
                try:
                    row = json.loads(raw)
                    key = tuple(row[field] for field in self.key_fields)
                    bound = row.get("pilot_protocol_sha256", row.get("protocol_sha256"))
                    if not protocol_hash or bound != protocol_hash or key not in self.allowed:
                        raise ValueError("Record identity or protocol does not match this run")
                    status = row[self.status_field] if self.status_field else (
                        "reused_generation" if row.get("generation_origin") == "reused_actual_v1_generation" else "generated")
                    statuses = {"valid", "invalid_output"} if self.status_field == "final_status" else {
                        "judged", "insufficient_evidence", "invalid_output", "source_conflict"} if self.status_field else {"generated", "reused_generation"}
                    if status not in statuses or (self.expected_judge and row.get("judge") != self.expected_judge):
                        raise ValueError("Invalid final status or wrong judge stage")
                    if any(row.get(k) != v for k, v in self.expected_fields.items()):
                        raise ValueError("Repair model or kind does not match the journal")
                    fresh = None
                    if row.get("derivation_profile") in REPAIR_PROFILES:
                        attempts = row.get("attempts")
                        if not isinstance(attempts, list) or not attempts or any(not isinstance(a, dict) for a in attempts):
                            raise ValueError("Derived record has no valid attempt provenance")
                        fresh = sum(str(a.get("generation_origin", "")).startswith("new_") for a in attempts)
                    if key in self.seen:
                        self.duplicates += 1
                        continue
                    self.seen.add(key)
                    self.statuses[status] += 1
                    if fresh is not None:
                        self.origins["derived"] += 1
                        self.origins["with_new_generation" if fresh else "reused_without_new_generation"] += 1
                    at = row.get("at", row.get("completed_at"))
                    if isinstance(at, str) and (self.last_at is None or at > self.last_at):
                        self.last_at = at
                except (ValueError, KeyError, TypeError):
                    self.errors += 1
        return self.summary()

    def summary(self):
        return {"done": len(self.seen), "statuses": dict(self.statuses), "duplicates": self.duplicates,
                "malformed_or_mismatched": self.errors, "pending_tail_bytes": self.tail_bytes,
                "last_record_at": self.last_at, "origins": dict(self.origins)}


def review_bound(review):
    if not review:
        return False
    fields = {"protocol_sha256": QUAL / "judge_protocol.json",
              "validation_manifest_sha256": QUAL / "validation/manifest.json",
              "fresh_manifest_sha256": QUAL / "validation/fresh/manifest.json"}
    if any(not review.get(k) or review[k] != file_sha(p) for k, p in fields.items()):
        return False
    for field, folder in (("pilot_record_sha256", "validation"), ("fresh_pilot_record_sha256", "validation/fresh")):
        for model in ("qwen", "phi"):
            digest = review.get(field, {}).get(model)
            if not digest or digest != file_sha(QUAL / folder / f"{model}_records.jsonl"):
                return False
    return True


def qualification_ok(review, acceptance, phases):
    profile = acceptance.get("version")
    repaired = profile in REPAIR_PROFILES
    if ASSESS.name.startswith("assessment_repaired") and not repaired:
        return False
    if not (review_bound(review) and review.get("accepted") is True and
            acceptance.get("accepted") is True and acceptance.get("numeric_accepted") is True and
            acceptance.get("cases_per_model") == 765 and
            (review.get("profile") == profile if repaired else acceptance.get("profile") == "v8") and
            acceptance.get("source_review_sha256") == file_sha(QUAL / "source_review.json") and
            all(phases[m]["done"] == phases[m]["total"] == 765 for m in ("qwen", "phi"))):
        return False
    bindings = acceptance.get("files", {})
    required = [QUAL / "judge_protocol.json", QUAL / "source_review.json"] + [
        QUAL / folder / name for folder in ("validation", "validation/fresh")
        for name in ("manifest.json", "qwen_records.jsonl", "phi_records.jsonl",
                     "qwen_raw_generation_journal.jsonl", "phi_raw_generation_journal.jsonl")]
    if repaired:
        if (read_json(ASSESS / "core_plan.json").get("version") != profile or
                acceptance.get("scientific_gold_accuracy", "missing") is not None):
            return False
        required += [ASSESS / "core_plan.json"] + [QUAL / folder / "cases.jsonl"
            for folder in ("validation", "validation/fresh")]
        required += list(scoped(QUAL / "validation").glob("*_repair_journal.jsonl"))
        required += list(scoped(QUAL / "validation/fresh").glob("*_repair_journal.jsonl"))
    if any(str(p.relative_to(ROOT)) not in bindings for p in required):
        return False
    for relative, digest in bindings.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not digest or file_sha(ROOT / path) != digest:
            return False
    return True


class Monitor:
    def __init__(self):
        select_layout()
        self.plan = read_json(ASSESS / "core_plan.json")
        self.profile = self.plan.get("version")
        self.repaired = self.profile in REPAIR_PROFILES
        if self.plan.get("queries") != 145 or self.plan.get("candidates") != 280:
            raise ValueError("The frozen 145-request core plan is missing or changed")
        self.counters, self.protocol_paths = {}, {}
        self.pilot_ids = set()
        for cohort in ("validation", "validation/fresh"):
            case_ids = ids(QUAL / cohort / "cases.jsonl", "case_id")
            self.pilot_ids.update(case_ids)
            for model in ("qwen", "phi"):
                name = (model, cohort)
                self.counters[name] = RecordCounter(QUAL / cohort / f"{model}_records.jsonl",
                    ("case_id",), {(cid,) for cid in case_ids}, "final_status")
                self.protocol_paths[name] = QUAL / cohort / f"{model}_protocol.json"
                if self.repaired:
                    for kind in ("ordinary", "focused"):
                        key = (model, cohort, kind)
                        self.counters[key] = RecordCounter(
                            QUAL / cohort / f"{model}_{kind}_repair_journal.jsonl",
                            ("case_id",), {(cid,) for cid in case_ids}, None,
                            expected_fields={"model": model, "repair_kind": kind})
                        self.protocol_paths[key] = self.protocol_paths[name]
        queries = ids(OUT / "inputs/queries.jsonl", "query_id")
        candidates = ids(OUT / "inputs/candidates.jsonl", "candidate_id")
        pairs = {(q, c) for q in queries for c in candidates}
        secondary = {(q, c) for q, c in pairs if int(hashlib.sha256(
            ("pbpk-secondary-v1" + q + c).encode()).hexdigest(), 16) % 20 == 0}
        if len(pairs) != self.plan["expected_primary_pairs"] or len(secondary) != self.plan["expected_secondary_pairs"]:
            raise ValueError("Frozen pair counts disagree with actual selected inputs")
        for stage, filename, allowed in (("primary", "judgments.jsonl", pairs),
                                        ("secondary", "secondary_judgments.jsonl", secondary)):
            self.counters[stage] = RecordCounter(ASSESS / filename,
                ("query_id", "candidate_id"), allowed, "status", expected_judge=stage)
            self.protocol_paths[stage] = ASSESS / "judge_protocol.json"
        self.totals = {"qwen": len(self.pilot_ids), "phi": len(self.pilot_ids),
                       "primary": len(pairs), "secondary": len(secondary)}
        self.review_ids = set(read_json(QUAL / "source_review_plan.json")["case_ids"])

    def snapshot(self):
        layout = available_repair_layout()
        if layout and layout[0] != getattr(self, "profile", None):
            self.__init__()
        stats = {key: counter.update(file_sha(self.protocol_paths[key])) for key, counter in self.counters.items()}
        phases = {}
        for model in ("qwen", "phi"):
            parts = [stats[(model, cohort)] for cohort in ("validation", "validation/fresh")]
            statuses = Counter()
            for part in parts:
                statuses.update(part["statuses"])
            phases[model] = {"done": sum(p["done"] for p in parts), "total": self.totals[model],
                             "statuses": dict(statuses)}
            if getattr(self, "repaired", False):
                origins = Counter()
                for part in parts:
                    origins.update(part.get("origins", {}))
                phases[model]["origins"] = dict(origins)
                phases[model]["new_gpu_generations"] = sum(stats[(model, cohort, kind)]["statuses"].get("generated", 0)
                    for cohort in ("validation", "validation/fresh") for kind in ("ordinary", "focused"))
                phases[model]["reused_gpu_generations"] = sum(stats[(model, cohort, kind)]["statuses"].get("reused_generation", 0)
                    for cohort in ("validation", "validation/fresh") for kind in ("ordinary", "focused"))
        for stage in ("primary", "secondary"):
            phases[stage] = {**stats[stage], "total": self.totals[stage]}
        pipeline = read_json(RUN / "pipeline_status.json")
        stage = pipeline.get("stage")
        active_path = QUAL_RUN / f"pilot_{stage[6:]}_status.json" if stage in {"pilot_qwen", "pilot_phi"} else RUN / "status.json"
        worker = read_json(active_path)
        worker_matches = worker.get("pid") == pipeline.get("worker_pid") and bool(worker.get("pid"))
        if not worker_matches:
            worker = {}
        review = read_json(QUAL / "source_review.json")
        reviewed = (set(review.get("checked_case_ids", [])) & self.review_ids) if review_bound(review) else set()
        acceptance = read_json(ASSESS / "qualification_acceptance.json")
        admitted = qualification_ok(review, acceptance, phases) and reviewed == self.review_ids
        # The evaluator can be large; read only when its mtime changes.
        report_path = ASSESS / "automatic_evaluation.json"
        stamp = scoped(report_path).stat().st_mtime_ns if report_path.exists() else None
        if not hasattr(self, "report_stamp") or stamp != self.report_stamp:
            self.report_stamp, self.report = stamp, read_json(report_path)
        evaluated = (self.report.get("state") == "complete" and
                     self.report.get("completion_acceptance", {}).get("state") == "passed")
        agreement = read_json(ASSESS / "judge_agreement.json")
        agreement_bound = (bool(agreement) and agreement.get("protocol_sha256") == file_sha(ASSESS / "judge_protocol.json")
                           and agreement.get("source_review_sha256") == file_sha(QUAL / "source_review.json"))
        alerts = []
        pipeline_alive, worker_alive = process_alive(pipeline), process_alive(worker)
        if pipeline.get("state") in {"running", "waiting_for_source_review"} and not pipeline_alive:
            alerts.append("Scheduler exited while the state file remained stale; the run is not continuing")
        for name, stat in stats.items():
            if stat["duplicates"] or stat["malformed_or_mismatched"]:
                alerts.append(f"{name}: duplicates: {stat['duplicates']} records, format/protocol/scope anomalies: {stat['malformed_or_mismatched']} records (not counted twice)")
        for value in (pipeline, worker):
            for field in ("guard_reason", "error_message"):
                if value.get(field):
                    alerts.append(str(value[field])[:200])
        done, total = sum(p["done"] for p in phases.values()), sum(self.totals.values())
        clean = not any(s["duplicates"] or s["malformed_or_mismatched"] or s["pending_tail_bytes"] for s in stats.values())
        complete = (done == total and admitted and evaluated and agreement_bound and clean
                    and pipeline.get("state") == "complete")
        repair = {}
        if getattr(self, "repaired", False):
            repair = {"profile": self.profile,
                "derived_records": sum(phases[m]["origins"].get("derived", 0) for m in ("qwen", "phi")),
                "reused_without_new_generation": sum(phases[m]["origins"].get("reused_without_new_generation", 0) for m in ("qwen", "phi")),
                "new_gpu_generations": sum(phases[m]["new_gpu_generations"] for m in ("qwen", "phi")),
                "reused_gpu_generations": sum(phases[m]["reused_gpu_generations"] for m in ("qwen", "phi")),
                "pending_additional_repairs": worker.get("pending_additional_repairs"),
                "repair_kind": worker.get("repair_kind")}
        return {"at": datetime.now(timezone.utc).isoformat(), "phases": phases,
            "done": done, "total": total, "complete": complete,
            "pipeline": pipeline, "pipeline_alive": pipeline_alive, "worker": worker,
            "worker_alive": worker_alive, "reviewed": len(reviewed), "review_total": len(self.review_ids),
            "review_accepted": admitted, "review_exists": bool(review), "evaluation_complete": evaluated,
            "pending_tail_bytes": sum(s["pending_tail_bytes"] for s in stats.values()),
            "alerts": list(dict.fromkeys(alerts)), "repair": repair}


ACTIVE_PHASE = {"pilot_qwen": "qwen", "pilot_phi": "phi", "primary": "primary", "secondary": "secondary"}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
COLORS = {"cyan": "38;5;81", "purple": "38;5;147", "green": "38;5;78",
          "amber": "38;5;221", "red": "38;5;203", "muted": "38;5;244", "bold": "1"}


def columns(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in ANSI.sub("", text))


def fit(text, width):
    """Clip styled text by terminal columns, preserving Chinese alignment."""
    output, used = [], 0
    for part in re.split(r"(\x1b\[[0-9;]*[A-Za-z])", text):
        if ANSI.fullmatch(part):
            output.append(part)
            continue
        for char in part:
            size = columns(char)
            if used + size > width:
                return "".join(output) + ("\x1b[0m" if "\x1b" in text else "")
            if char.isprintable():
                output.append(char)
                used += size
    return "".join(output)


def duration(seconds):
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f"{days}d " if days else "") + f"{hours:02}:{minutes:02}:{seconds:02}"


def seconds_since(stamp):
    try:
        return max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds())
    except (ValueError, TypeError):
        return None


class Throughput:
    """Current-stage committed records/s; never extrapolate across judge stages."""
    def __init__(self):
        self.identity, self.samples = None, deque()

    def update(self, snapshot):
        phase = ACTIVE_PHASE.get(snapshot["pipeline"].get("stage"))
        worker = snapshot["worker"]
        identity = (phase, worker.get("pid"), worker.get("start_ticks"))
        active = (phase and snapshot["pipeline_alive"] and snapshot["worker_alive"] and
                  snapshot["pipeline"].get("state") == "running" and worker.get("state") == "judging")
        if identity != self.identity or not active:
            self.identity, self.samples = identity, deque()
        if not active:
            note = {"verifying_weights": "Verifying local weights", "loading_gpu": "Loading model on GPU",
                    "replaying": "Reviewing original responses", "complete": "Current stage complete"}.get(worker.get("state"), "Waiting for scoring")
            return {"rate": None, "eta": None, "note": note}
        repaired_pilot = bool(snapshot.get("repair")) and phase in {"qwen", "phi"}
        count = snapshot["phases"][phase].get("new_gpu_generations", 0) if repaired_pilot else snapshot["phases"][phase]["done"]
        now = time.monotonic()
        if self.samples and count < self.samples[-1][1]:
            self.samples.clear()
        self.samples.append((now, count))
        while len(self.samples) > 1 and now - self.samples[0][0] > 900:
            self.samples.popleft()
        age = seconds_since(worker.get("last_commit_at"))
        if age is not None and age >= 1200:
            return {"rate": None, "eta": None, "note": "Commit progress stalled"}
        changes = sum(b[1] > a[1] for a, b in zip(self.samples, list(self.samples)[1:]))
        elapsed = now - self.samples[0][0]
        if elapsed < 60 or changes < 2:
            return {"rate": None, "eta": None, "note": "Collecting timing samples"}
        rate = (count - self.samples[0][1]) / elapsed
        remaining = snapshot["phases"][phase]["total"] - count
        return {"rate": rate, "eta": remaining / rate if rate > 0 and not repaired_pilot else None,
                "note": "New GPU correction calls only; reused outputs excluded from throughput" if repaired_pilot else "Observed over the last 15 minutes"}


def render(s, *, color=False, width=96, height=None, throughput=None, interval=5):
    width = max(1, min(110, width))
    narrow = width < 68
    def paint(text, tone):
        return f"\x1b[{COLORS[tone]}m{text}\x1b[0m" if color else text
    def pad(text, size):
        return text + " " * max(0, size - columns(text))
    def progress(label, done, total, tone="cyan"):
        ratio = min(1, max(0, done / total)) if total else 0
        suffix = f" {100*ratio:5.1f}% {done:,}/{total:,}"
        prefix = pad(label, 8 if narrow else 18) + " "
        size = max(2, width - columns(prefix) - columns(suffix) - 2)
        whole = int(size * ratio)
        partial = "╸" if 0 < ratio < 1 else ""
        fill = paint("━" * whole + partial, tone)
        rest = paint("─" * max(0, size - whole - len(partial)), "muted")
        if width < 48:
            return paint(label, tone) + f"  {100*ratio:.1f}%  {done:,}/{total:,}"
        return paint(prefix, tone) + fill + rest + suffix
    pipeline, worker = s["pipeline"], s["worker"]
    phase = ACTIVE_PHASE.get(pipeline.get("stage"))
    running = s["pipeline_alive"] and pipeline.get("state") == "running"
    status, tone = ("RUNNING", "green") if running else ("PENDING", "amber")
    if s["complete"]:
        status, tone = "COMPLETE", "green"
    elif pipeline.get("state") == "waiting_for_source_review" and s["pipeline_alive"]:
        status = "REVIEW"
    elif pipeline.get("state") in {"failed", "stopped_on_guard", "stopped_on_validation", "interrupted"} or s["alerts"]:
        status, tone = "CHECK", "red"
    elif not s["pipeline_alive"] and pipeline:
        status = "STOPPED"
    title = "PBPK / MODEL RECOMMENDATION" if not narrow else "PBPK / RECOMMENDATION"
    heading = paint(title, "bold") + " " * max(2, width - columns(title) - len(status) - 3) + paint("● " + status, tone)
    subtitle = "145 requests × 280 candidates  ·  12 rankers reused"
    if s.get("repair"):
        subtitle += "  ·  " + s["repair"]["profile"]
    lines = [heading, paint(subtitle, "muted"),
             paint("─" * width, "muted"), progress("TOTAL", s["done"], s["total"], "purple"), ""]
    labels = {"qwen": "Qwen validation", "phi": "Phi validation", "primary": "Primary scoring", "secondary": "Phi repeat judgment"}
    if s.get("repair"):
        labels.update(qwen="Qwen repair review", phi="Phi repair review")
    short = {"qwen": "Qwen", "phi": "Phi", "primary": "Primary", "secondary": "Review"}
    for index, (name, label) in enumerate(labels.items(), 1):
        p = s["phases"][name]
        current = name == phase and running
        marker = "›" if current else "✓" if p["done"] == p["total"] else "·"
        row_label = marker + " " + (short[name] if narrow else f"{index:02} {label}")
        lines.append(progress(row_label, p["done"], p["total"], "cyan" if current else "green" if p["done"] == p["total"] else "muted"))
    lines += ["", paint("─" * width, "muted")]
    metrics = throughput or {"rate": None, "eta": None, "note": "Collecting timing samples"}
    elapsed = duration(seconds_since(pipeline.get("started_at")))
    rate = f"{metrics['rate']:.3f} records/s" if metrics["rate"] is not None else "-- records/s"
    eta = "~" + duration(metrics["eta"]) if metrics["eta"] is not None else "--:--:--"
    if phase:
        model = "Qwen3.6-27B" if phase in {"qwen", "primary"} else "Phi-4"
        lines.append(paint(model, "cyan") + "  " + labels[phase] + paint("  ·  " + metrics["note"], "muted"))
    elif pipeline.get("stage") == "source_review":
        lines.append(paint("Waiting for source review", "amber"))
    elif pipeline.get("stage") == "final_evaluation":
        lines.append(paint("Aggregating quality metrics", "cyan"))
    timing = f"elapsed {elapsed}  |  {rate}  |  ETA(stage) {eta}"
    if narrow:
        lines += [f"elapsed {elapsed}  |  {rate}", f"ETA(stage) {eta}  ·  {metrics['note']}"]
    else:
        lines.append(timing)
    if phase:
        names = {"valid": "valid", "judged": "judged", "invalid_output": "invalid",
                 "insufficient_evidence": "unknown", "source_conflict": "conflict"}
        pieces = []
        for name, count in s["phases"][phase]["statuses"].items():
            pieces.append(paint(f"{names.get(name, name)}={count:,}", "amber" if name in {"invalid_output", "source_conflict"} else "muted"))
        if pieces:
            lines.append("  ".join(pieces))
    age = duration(seconds_since(worker.get("last_commit_at")))
    live = paint("LIVE", "green") if s["pipeline_alive"] else paint("OFF", "amber")
    gpu_worker = paint("LIVE", "green") if s["worker_alive"] else paint("IDLE", "muted")
    lines.append(f"last commit {age} ago  |  scheduler {live}  worker {gpu_worker}")
    review = paint("PASS", "green") if s["review_accepted"] else paint("PENDING", "amber")
    evaluated = paint("DONE", "green") if s["evaluation_complete"] else paint("PENDING", "muted")
    gates = f"Source review {s['reviewed']}/{s['review_total']} {review}  ·  Metric aggregation {evaluated}"
    lines += [gates,
              paint("─" * width, "muted")]
    if s.get("repair"):
        repair = s["repair"]
        lines.append(paint(f"Derived review {repair['derived_records']:,}/1,530 · Original output reuse {repair['reused_without_new_generation']:,}", "purple"))
        pending = repair.get("pending_additional_repairs")
        kind = {"ordinary": "Standard correction", "focused": "Focused correction"}.get(repair.get("repair_kind"), "Waiting for correction")
        lines.append(paint(f"New GPU corrections {repair['new_gpu_generations']:,} times · Reused corrections {repair.get('reused_gpu_generations', 0):,} times", "muted"))
        lines.append(paint(f"Pending {pending if pending is not None else '--'} records · {kind}", "muted"))
    if s["alerts"]:
        lines.extend(paint("! " + alert, "red") for alert in s["alerts"])
    if s["pending_tail_bytes"]:
        lines.append(paint("Waiting for the final record to finish writing before counting.", "amber"))
    lines += [paint("Record progress is not elapsed-time progress; ETA is a rough estimate for the current stage and remains blank without enough timing samples.", "muted"),
              paint(f"{s['at'][11:19]} UTC  ·  refresh {interval:g}s  ·  Ctrl+C closes the display only", "muted")]
    if height is not None and len(lines) > max(1, height):
        compact = [heading, progress("TOTAL", s["done"], s["total"], "purple")]
        if phase:
            p = s["phases"][phase]
            compact.append(progress(short[phase], p["done"], p["total"]))
        if s.get("repair"):
            repair = s["repair"]
            compact.append(f"Review {repair['derived_records']}/1530 · New GPU corrections {repair['new_gpu_generations']} times")
        compact.extend(paint("! " + a, "red") for a in s["alerts"][:1])
        compact += [gates, timing, paint("Record progress differs from elapsed time; ETA covers this stage only", "muted"),
                    paint("Ctrl+C closes the display only", "muted")]
        lines = compact[:max(1, height)]
    return "\n".join(fit(line, width) for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--json", action="store_true", help="Print one machine-readable snapshot")
    parser.add_argument("--plain", action="store_true", help="Disable ANSI colors and live cursor updates")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="Terminal color mode")
    parser.add_argument("--interval", type=float, default=5, help="Refresh seconds (at least 1; default 5)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("--interval must be finite and at least 1 second")
    terminal = sys.stdout.isatty() and (os.environ.get("TERM") != "dumb" or args.color == "always")
    interactive = terminal and not (args.once or args.json or args.plain)
    color = (args.color == "always" or (terminal and args.color == "auto" and "NO_COLOR" not in os.environ)) and not (args.json or args.plain)
    screen_open = False
    try:
        monitor = Monitor()
        throughput = Throughput()
        if interactive:
            # Alternate screen preserves shell history; cursor restored in finally.
            print("\033[?1049h\033[?25l", end="", flush=True)
            screen_open = True
        while True:
            try:
                snapshot = monitor.snapshot()
                terminal_size = shutil.get_terminal_size((98, 28))
                output = json.dumps(snapshot, ensure_ascii=False, indent=2) if args.json else render(
                    snapshot, color=color, width=terminal_size.columns - 2,
                    height=terminal_size.lines - 1 if interactive else None,
                    throughput=throughput.update(snapshot), interval=args.interval)
            except (OSError, ValueError, KeyError) as error:
                output = "Cannot read complete progress yet; retrying: " + str(error)
                if not interactive:
                    print(output, file=sys.stderr)
                    return 1
            if interactive:
                # Rewrite in place without clearing the full screen on every tick.
                print("\033[H" + "\n".join(line + "\033[K" for line in output.splitlines()) + "\033[J", end="", flush=True)
            else:
                print(output)
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("Cannot read progress for this experiment: " + str(error), file=sys.stderr)
        return 1
    finally:
        if screen_open:
            print("\033[?25h\033[?1049l", end="", flush=True)
            print("Progress display closed; background experiments continue.")


if __name__ == "__main__":
    raise SystemExit(main())
