#!/usr/bin/env python3
"""Curate explicit primary-paper arms; never fit a model or expose sealed values.

The selected table cells are deterministic, reviewed locators, not an LLM
extraction. Input XML and source models are reused in place and SHA-bound.
Protocols contain no PK outcomes. Sealed files are curator/scorer inputs only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "experiments/blueprint_design/independent_cohort_20260920"
OUTPUT = ROOT / "experiments/blueprint_design/external_validation/newstudy_bundle_20260920"


def txt(node):
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path):
    path = path.resolve()
    path.relative_to(ROOT)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite a different curated artifact: {path}")
    if not path.exists():
        path.write_text(content)
    return reference(path)


def article_from_container(path, doi):
    root = ET.parse(path).getroot()
    return next(a for a in root.iter("article")
                if any(z.text == doi for z in a.findall(".//article-meta/article-id")))


def endpoint(name, row, col, label, unit, window, primary=False, n=None):
    return dict(endpoint=name, raw_row=row, raw_column=col, expected_label=label,
                unit=unit, window=window, primary=primary, endpoint_n=n)


CASES = [
    dict(compound="bupropion", pmcid="PMC7368561", case_id="bupropion_teaima2020_wellbutrin_sr_150mg",
         study_id="teaima2020_rec_fopcu_pi2333", arm="Wellbutrin SR reference; single dose",
         table_id="T0003", table_arm="Wellbutrin", source_names=["Teaima", "32764875", "258571", "PI 2333"],
         article_year=2020, clinical_site="Faculty of Pharmacy, Cairo University, Egypt",
         registry_id=None, ethics_id="REC-FOPCU PI 2333",
         dose=dict(value=150, unit="mg", basis="marketed Wellbutrin SR labelled dose", route="PO",
                   formulation="Wellbutrin SR sustained-release tablet", administrations_h=[0],
                   water_ml=240, salt="bupropion hydrochloride marketed product; article calls dose BUP",
                   conversion_rule="Freeze labelled-dose to model-species mass conversion; do not equate salt mass and free base silently."),
         food=dict(predose_fasting_h=10, first_meal_h=4, drinking_water_resumes_h=2),
         population=dict(n=6, sex="male", healthy=True, country="Egypt", age_years=None, weight_kg=None,
                         caution="Age/height/weight of two radiographic volunteers are NOT demographics of the separate six-person PK cohort."),
         sampling_h=[0,1,2,3,4,5,6,8,24,48,72],
         targets=[endpoint("Cmax",2,1,"Wellbutrin","ng/mL","nominal sampled maximum",True,6),
                  endpoint("AUC0_72",2,2,"Wellbutrin","ng*h/mL",[0,72],True,6),
                  endpoint("AUCinf",2,3,"Wellbutrin","ng*h/mL","terminal extrapolation",False,6)],
         caveats=["Only the marketed reference arm is admitted; novel floating formulation is not substituted.",
                  "Six-person PK cohort; aggregate data, not six independently extractable individuals.",
                  "Reported plasma parent bupropion; original table values preserved, no adjustment using expected exposure.",
                  "SR release profile and population priors must be frozen from non-target sources before predicting."],
         model_papers=[("bupropion", "10.3390/pharmaceutics13030331")]),
    dict(compound="dextromethorphan", pmcid="PMC5297975", case_id="dextromethorphan_yamazaki2016_day1_30mg",
         study_id="NCT01651325", arm="Dextromethorphan Alone day 1, before any isavuconazole",
         table_id="cpdd281-tbl-0001", table_arm="Dextromethorphan Alone", source_names=["27273149", "cpdd.281", "NCT01651325"],
         article_year=2016, clinical_site="PAREXEL Early Phase Clinical Unit, Baltimore, Maryland, USA; May-July 2012",
         registry_id="NCT01651325", ethics_id="Aspire IRB LLC, Santee, California",
         dose=dict(value=30, unit="mg", basis="dextromethorphan hydrobromide", route="PO",
                   formulation="ROBITUSSIN oral capsules", administrations_h=[0], water_ml=None,
                   salt="dextromethorphan hydrobromide; hydration state not specified",
                   conversion_rule="Resolve salt/hydration convention against native compound before execution; record exact factor."),
         food=dict(predose_fasting_h=10, first_meal_h=4),
         population=dict(n=24, healthy=True, age_range_years=[18,55], min_weight_kg=45,
                         bmi_range=[18,32], sex="men and women; proportions in supplemental Table S1",
                         cyp2d6="23 completers: 12 intermediate, 11 extensive; baseline phenotype of one withdrawal unknown",
                         caution="Do not present completer phenotype fractions as an exact 24-person baseline mixture."),
         sampling_h=[0,.5,1,1.5,2,3,4,5,6,8,10,12,16,20,24,48,72],
         targets=[endpoint("Cmax",4,7,"Cmax","ng/mL","nominal sampled maximum",True,24),
                  endpoint("AUClast",3,7,"AUClast","ng*h/mL","individual last measurable concentration",False,24),
                  endpoint("AUCinf",2,7,"AUC∞","ng*h/mL","terminal extrapolation",False,22)],
         caveats=["Single-dose baseline only; no isavuconazole dose or other substrate probe in this cohort.",
                  "Same publication contains separately registered bupropion and other trials; publication is one grouping unit for conservative inference.",
                  "AUCinf excludes two subjects for >20% extrapolation; Cmax and AUClast remain n=24.",
                  "Unknown individual BLQ times preclude equating AUClast with AUC0-72.",
                  "Freeze a baseline CYP2D6 mixture approximation plus one-person assignment sensitivity; do not estimate it from these outcomes."],
         model_papers=[("dextromethorphan","10.1002/psp4.12776"),("quinidine","10.1002/psp4.12981")]),
    dict(compound="clarithromycin", pmcid="PMC10333645", case_id="clarithromycin_oh2023_alone_day5_500mg_bid",
         study_id="NCT02052336", arm="R2 clarithromycin alone, day 5 morning dosing interval",
         table_id="T3", table_arm="Alone", source_names=["37440779", "tcp.2023.31.e11", "NCT02052336"],
         article_year=2023, clinical_site="Busan Paik Hospital, South Korea", registry_id="NCT02052336",
         ethics_id="Busan Paik Hospital IRB and South Korea MFDS",
         dose=dict(value=500, unit="mg", basis="clarithromycin labelled dose", route="PO",
                   formulation="oral product; manufacturer/release designation not stated in full-text methods",
                   administrations_h=[0,12,24,36,48,60,72,84,96,108],
                   observation_origin_h=96, dosing_interval_h=12, water_ml=240,
                   water_ml_days1_4=150, salt="parent clarithromycin", conversion_rule="Use parent mass; source release assumption must be explicitly frozen."),
         food=dict(morning_predose_fasting_h=8, morning_postdose_fasting_h=4,
                   evening_predose_fasting_h=2, evening_postdose_fasting_h=2),
         population=dict(n=23, enrolled_n=24, healthy=True, sex="male", country="South Korea",
                         enrolled_mean_age_years=24.79, enrolled_mean_weight_kg=68.92,
                         enrolled_mean_bmi=22.82, age_range_years=[19,45], bmi_range=[19,28],
                         caution="Reported demographic means describe 24 enrolled; outcomes describe 23 completers."),
         sampling_h=[0,1,2,3,4,6,8,12],
         targets=[endpoint("Cmax_ss_day5",2,3,"Css,max","ug/L","day5 nominal sampled maximum",True,23),
                  endpoint("AUCtau_day5",3,3,"AUCtau","ug*h/L",[96,108],True,23)],
         caveats=["This is five-day repeated dosing, not a single-dose case; preserve auto-inhibition/accumulation.",
                  "The 12-hour endpoint precedes the next dose; choose left-limit sampled concentration there.",
                  "Arithmetic mean/SD alone-column values, not geometric coadministration ratios.",
                  "Regimen is twice daily; 12-hour spacing is the explicit execution convention because clock times are not given."],
         model_papers=[]),
    dict(compound="voriconazole", pmcid="PMC9985189", case_id="voriconazole_li2023_reference_iv_4mgkg",
         study_id="li2023_qyfyec2020_018_01", arm="Reference Vfend 4 mg/kg intravenous infusion",
         table_id="Tab2", table_arm="R", source_names=["36869387", "s40360-023-00652-3", "QYFYEC 2020"],
         article_year=2023, clinical_site="Affiliated Hospital of Qingdao University, China",
         registry_id=None, ethics_id="QYFYEC 2020-018-01",
         dose=dict(value=4, unit="mg/kg", basis="voriconazole active drug", route="IV",
                   formulation="Vfend 200 mg/vial reference solution", administrations_h=[0], infusion_duration_min=80,
                   salt="parent voriconazole", conversion_rule="Dose actual representative/population weight, not fixed 200 mg vial amount."),
         food=dict(status="not specified; intravenous route"),
         population=dict(n=24, male_n=21, female_n=3, healthy=True, country="China", ethnicity="Han",
                         age_mean_years=30.50, age_sd_years=7.12, age_range_years=[20,42],
                         weight_mean_kg=66.71, weight_sd_kg=7.61, weight_range_kg=[52.50,83.50],
                         height_mean_cm=169.79, height_sd_cm=5.27,
                         genotype="CYP2C19 not reported; do not infer participant genotypes from nationality"),
         sampling_h=[0,.5,1,1.33,1.42,1.5,1.75,2,2.5,3,4,6,8,12,24,36,48],
         targets=[endpoint("Cmax",2,2,"Cmax","ug/mL","nominal sampled maximum",True,24),
                  endpoint("AUClast",3,2,"AUC0–t","ug*h/mL","individual last measured quantifiable sample",False,24),
                  endpoint("AUCinf",4,2,"AUC0–∞","ug*h/mL","terminal extrapolation",False,24)],
         caveats=["Reference formulation only; crossover test is not a second independent cohort.",
                  "Published parenthesized CV entries for exposure rows are arithmetically inconsistent with mean and SD; preserve but do not use as weights/CI.",
                  "AUC0-t is not certified AUC0-48; post-Cmax BLQ values are ND in original NCA.",
                  "Infusion end is 80/60 hours while nominal table says 1.33 hours; preserve both, do not shift observations."],
         model_papers=[]),
    dict(compound="paroxetine", pmcid="PMC8620498", case_id="paroxetine_elshafeey2021_market_tablet_25mg",
         study_id="elshafeey2021_grc_1_21_r4", arm="Market tablet reference, swallowed with water",
         table_id="pharmaceutics-13-01869-t005", table_arm="Market Tablet", source_names=["34834284", "13111869", "Elshafeey", "GRC/1/21/R4"],
         article_year=2021, clinical_site="Genuine Research Center, Cairo, Egypt", registry_id=None,
         ethics_id="GRC/1/21/R4",
         dose=dict(value=25, unit="mg", basis="paroxetine labelled dose", route="PO",
                   formulation="Paroxetine 25 mg oral market tablet, EVA Pharma, Cairo",
                   administrations_h=[0], water_ml=200,
                   salt="salt/hydration not stated in clinical protocol", conversion_rule="Freeze native salt-to-active mapping; do not infer release type solely from dose strength."),
         food=dict(predose_fasting_h=10, water_resumes_h=2, first_meal_h=4),
         population=dict(n=6, sex="male", healthy=True, country="Egypt", age_mean_years=40.6,
                         age_sd_years=5.0, weight_mean_kg=72.4, weight_sd_kg=6.7,
                         genotype="CYP2D6 not reported"),
         sampling_h=[0,.33,.66,1,1.5,2,2.5,3,3.5,4,4.5,5,6,8,12,24,48],
         targets=[endpoint("Cmax",2,2,"Cmax","ng/mL","nominal sampled maximum",True,6),
                  endpoint("AUC0_48",3,2,"AUC0–48","ng*h/mL",[0,48],True,6),
                  endpoint("AUCinf",4,2,"AUC0-∞","ng*h/mL","terminal extrapolation",False,6)],
         caveats=["Only swallowed market tablet admitted; buccal nano-film must not be treated as ordinary oral tablet.",
                  "PK cohort n=6 is separate from eight-person taste study.",
                  "Release designation is not explicit; freeze external product-supported mapping or report formulation sensitivity.",
                  "Endpoint means are arithmetic, despite skew/large SD; preserve original statistic without conversion."],
         model_papers=[("paroxetine","10.3390/pharmaceutics14081734")]),
]
vori6 = copy.deepcopy(next(c for c in CASES if c["compound"] == "voriconazole"))
vori6.update(case_id="voriconazole_li2023_reference_iv_6mgkg", arm="Reference Vfend 6 mg/kg intravenous infusion", table_id="Tab3")
vori6["dose"].update(value=6, infusion_duration_min=120)
vori6["population"].update(male_n=19, female_n=5, ethnicity="23 Han, 1 Manchu",
                         age_mean_years=27.0, age_sd_years=6.76, age_range_years=[19,44],
                         weight_mean_kg=67.73, weight_sd_kg=9.71, weight_range_kg=[49.50,86],
                         height_mean_cm=170.52, height_sd_cm=7.64)
vori6["sampling_h"]=[0,.5,1,1.5,1.75,2,2.08,2.17,2.33,2.5,3,4,6,8,12,24,36,48]
vori6["caveats"][-1]="Infusion duration 120 minutes; observed Cmax uses actual samples whereas protocol exposes nominal times."
CASES.append(vori6)


def audit(case, inventory):
    drug = next(d for d in inventory["drug_inventory"] if d["compound"] == case["compound"])
    needles = [n.casefold() for n in case["source_names"]]
    histories = [{k:v for k,v in s.items() if k not in {"observed_point_count", "observed_series_present"}}
                 for s in drug["historical_clinical_studies"]]
    refs = {r["path"] for m in drug["models"] for r in m["reference_documents"]}
    matches=[]
    for s in histories:
        if any(n in json.dumps(s,ensure_ascii=False).casefold() for n in needles):
            matches.append({"type":"registered_clinical_history","label":s["source_study_label"]})
    for p in refs:
        if any(n in (ROOT/p).read_text().casefold() for n in needles):
            matches.append({"type":"model_document","path":p})
    paper_checks=[]
    for compound,doi in case["model_papers"]:
        d=next(d for d in inventory["drug_inventory"] if d["compound"]==compound)
        cand=next(a for a in d["primary_article_candidates"] if a["identifiers"].get("doi")==doi)
        article=article_from_container(ROOT/cand["source_file"],doi)
        references=article.findall(".//ref-list/ref")
        found=[r.get("id") for r in references if any(n in txt(r).casefold() for n in needles)]
        if found:matches.append({"type":"upstream_model_paper_reference","doi":doi,"ref_ids":found})
        paper_checks.append({"doi":doi,"container":reference(ROOT/cand["source_file"]),
                             "reference_count":len(references),"matching_references":found})
    if matches:
        raise RuntimeError(f"Possible model-source overlap; adjudicate before promotion: {case['case_id']}: {matches}")
    rationale={
      "bupropion":"Teaima 2020 Egypt six-person SR trial absent from 53 unique registered clinical labels and all 91 references of Marok 2021; site/protocol PI2333 identifies newly recruited primary trial. The 2016 Yamazaki and 2018 Dennison bupropion trials were explicitly rejected because Marok identifies them as clinical model data.",
      "dextromethorphan":"Yamazaki NCT01651325 is a separately recruited 2012 Baltimore single-dose trial; absent from native registered DEX clinical cohorts, the 53-reference original Ruedesheim 2022 DEX paper and the subsequent quinidine model reference list. Upstream DEX source was checked, not just the DDI wrapper.",
      "clarithromycin":"Oh 2023 NCT02052336 Busan cohort absent from registered native observations and complete local Clarithromycin Evaluation References/Data documents. Historical named clinical sources are distinct author/study cohorts (Chu, Abduljalil, Quinney, Kees, Lappin etc.). Publication date alone is not the criterion.",
      "voriconazole":"Primary China trial explicitly has ethics QYFYEC2020-018-01 and 48 newly recruited Qingdao subjects. Upstream Li model was already indexed in PubMed on 2019-12-20; its complete 72-reference bibliography in the primary PubMed record has no target-study identifiers. Registered wrapper documentation attributes the voriconazole source to Li2019/2020, and registered source documents contain no target-study identifiers. Chronology plus the newly recruited institutional protocol supports disjointness from that published model; exact unpublished participant reuse cannot be verified. Future replacement models require a new source audit.",
      "paroxetine":"Elshafeey 2021 six-person Egypt reference-tablet trial and GRC/1/21/R4 absent from all registered clinical labels and all 93 references of Ruedesheim 2022. Not a renamed source cohort or buccal treatment substituted for reference."}
    if case["compound"]=="voriconazole":
        pm=OUTPUT/"source_audits/pubmed_31853755.xml"
        pr=ET.parse(pm).getroot()
        refs_pm=pr.findall(".//Reference")
        assert len(refs_pm)==72
        found=[idx for idx,r in enumerate(refs_pm) if any(n in txt(r).casefold() for n in needles)]
        assert not found, "Original upstream references overlap external study"
        pd=pr.find('.//PubMedPubDate[@PubStatus="pubmed"]')
        assert [pd.findtext(k) for k in ["Year","Month","Day"]]==["2019","12","20"]
        paper_checks.append({"doi":"10.1007/s40262-019-00856-z","pmid":"31853755",
                             "container":reference(pm),"reference_count":72,"matching_references":[],
                             "source_url":"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=31853755&retmode=xml",
                             "date_in_pubmed":"2019-12-20","publication_to_new_protocol_chronology":"target ethics number identifies 2020 institutional protocol; chronology is supporting evidence combined with original recruitment and source bibliography"})
    return dict(status="verified_disjoint_within_audited_clinical_sources",
                rationale=rationale[case["compound"]],
                audited_scope="registered target-drug clinical histories, all associated local model markdown, and listed original upstream full-text bibliography; generic physiology sources do not count as clinical outcome overlap",
                source_inventory=reference(COHORT/"cohort_inventory.json"),
                model_snapshots=[reference(ROOT/m["snapshot_path"]) for m in drug["models"]],
                registered_history_labels=sorted(set(s["source_study_label"] for s in histories)),
                model_documents=[reference(ROOT/p) for p in sorted(refs)],
                original_model_papers=paper_checks, matching_target_sources=matches,
                unknowns=["Participant identities unavailable; no claim of universal individual-level disjointness."]+
                         (["Original Li2020 supplementary study-to-fit-role table not locally available; primary 72-reference bibliography and pre-2020 source chronology checked instead, without inventing fit/test membership."] if case["compound"]=="voriconazole" else []))


def clean_cell(cell):
    cell=copy.deepcopy(cell)
    for node in list(cell.iter()):
        for child in list(node):
            if child.tag=="xref":node.remove(child)
    return txt(cell)


def parse_summary(text):
    numbers=re.findall(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?",text)
    if len(numbers)<2:raise ValueError(f"Expected mean and SD: {text!r}")
    values=[float(n.replace(',','')) for n in numbers]
    if values[0]<=0 or values[1]<0:raise ValueError("Non-positive mean or negative SD")
    return dict(mean=values[0],sd=values[1],extra_printed_numbers=values[2:],
                statistic="arithmetic_mean",dispersion="standard_deviation",raw=text)


def run(stage):
    inventory=json.loads((COHORT/"cohort_inventory.json").read_text())
    records=[]
    for c in CASES:
        source=COHORT/"retrieval_round1/sealed_articles"/(c["pmcid"]+".xml")
        article=ET.parse(source).getroot()
        identifiers={a.get("pub-id-type"):a.text for a in article.findall(".//article-meta/article-id")
                     if a.get("pub-id-type") in {"pmid","pmcid","doi"}}
        source_audit=audit(c,inventory)
        audit_ref=save(OUTPUT/"source_audits"/(c["compound"]+".json"),source_audit)
        protocol={"schema_version":"independent-pk-protocol-1","case_id":c["case_id"],
                  "compound":c["compound"],"study_id":c["study_id"],"publication_group_id":c["pmcid"],
                  "source_identifiers":identifiers,"source_title":txt(article.find(".//article-title")),
                  "source_url":"https://pmc.ncbi.nlm.nih.gov/articles/"+c["pmcid"]+"/",
                  "clinical_site":c["clinical_site"],"registry_id":c["registry_id"],"ethics_id":c["ethics_id"],
                  "arm":c["arm"],"dose":c["dose"],"food":c["food"],"population":c["population"],
                  "nominal_sampling_hours_from_observation_origin":c["sampling_h"],
                  "matrix":"plasma","analyte":"unconjugated parent drug, total plasma concentration",
                  "endpoint_contracts":[{k:v for k,v in e.items() if k not in {"raw_row","raw_column","expected_label"}} for e in c["targets"]],
                  "caveats":c["caveats"],"source_independence_audit":audit_ref,
                  "source_ready":True,"strict_source_independence":True,
                  "run_ready":False,"run_preconditions":["Freeze native compound and release/infusion mapping without these outcomes.",
                   "Freeze representative/population assumptions and endpoint estimator; retain approximations.",
                   "Denylist entire target article/trial and derived evidence from fitting/retrieval/selection.",
                   "Commit prediction file SHAs and configuration before scorer reads sealed outcomes."],
                  "curator_only_source_xml":reference(source),"outcomes_exposed_to_designer":False,
                  "recruitment_units":1,"dose_cases_are_not_independent_studies":True,
                  "inference_limit":"Aggregate endpoint validation; no individual records or complete concentration curves."}
        pref=save(OUTPUT/"protocols"/(c["case_id"]+".json"),protocol)
        rec={"case_id":c["case_id"],"compound":c["compound"],"study_id":c["study_id"],
             "publication_group_id":c["pmcid"],"protocol":pref,"endpoint_count":len(c["targets"]),
             "source_independence":source_audit["status"],"source_ready":True,"run_ready":False}
        if stage=="seal":
            table=article.find(f'.//table-wrap[@id="{c["table_id"]}"]')
            if table is None:raise ValueError("Missing selected table")
            rows=table.findall(".//tr")
            if c["table_arm"].casefold() not in txt(table).casefold():raise ValueError("Missing arm identity")
            outputs=[]
            for e in c["targets"]:
                row=rows[e["raw_row"]]
                if e["expected_label"].casefold() not in txt(row).casefold():raise ValueError("Endpoint label shifted")
                cell=list(row)[e["raw_column"]]
                summary=parse_summary(clean_cell(cell))
                outputs.append({**{k:v for k,v in e.items() if k not in {"raw_row","raw_column","expected_label"}},
                                **summary,"provenance":{"table_id":c["table_id"],"raw_xml_row_zero_based":e["raw_row"],
                                  "raw_xml_column_zero_based":e["raw_column"],"original_cell":txt(cell),
                                  "original_row":[txt(z) for z in row],
                                  "original_headers":[[txt(z) for z in r] for r in rows[:2]],
                                  "source_xml":reference(source)}})
            sealed={"schema_version":"sealed-primary-table-pk-1","case_id":c["case_id"],
                    "protocol":pref,"endpoints":outputs,"table_caption":txt(table.find("caption")),
                    "table_footnotes":txt(table.find("table-wrap-foot")),
                    "reported_statistic":"arithmetic_mean_and_SD","endpoint_count":len(outputs),
                    "no_outcome_imputation":True,"no_curve_digitization":True,"no_model_fitting":True,
                    "cv_extra_numbers_are_not_used":"Preserved only; Voriconazole printed CV inconsistency disallows deriving weights from it.",
                    "scoring_access_rule":"Require frozen prediction lock bound to this protocol SHA; designers/fitters must not read this file."}
            rec["sealed_observations"]=save(OUTPUT/"sealed"/(c["case_id"]+".json"),sealed)
        records.append(rec)
    receipt={"schema_version":"independent-pk-curation-receipt-1","stage":stage,"records":records,
             "compound_count":len(set(r["compound"] for r in records)),"case_count":len(records),
             "study_count":len(set(r["study_id"] for r in records)),
             "publication_count":len(set(r["publication_group_id"] for r in records)),
             "endpoint_count":sum(r["endpoint_count"] for r in records),
             "strict_source_disjoint_compounds":5,"provisional_source_disjoint_compounds":0,
             "source_download_bytes":35071,"gpu_used":False,"solver_calls":0,"human_labels_required":False,
             "outcomes_exposed_to_designer":False,"script":reference(Path(__file__)),
             "excluded_candidates":[{"compound":"bupropion","pmcid":"PMC5297975","reason":"Marok2021 clinical-data paragraph explicitly includes reference29; snapshot-only absence was a false negative."},
                                    {"compound":"bupropion","pmcid":"PMC6221039","reason":"Marok2021 clinical-data paragraph includes reference36 Dennison2018."},
                                    {"compound":"carbamazepine","pmcids":["PMC6099239","PMC11823298"],"reason":"PK endpoint tables describe fingolimod/zongertinib, not parent carbamazepine."}],
             "remaining_work":["Freeze explicit native/population/release mappings and run predictions before outcome access.",
                               "Re-audit if the source model version changes; source-role details absent from the Li supplement remain documented rather than inferred."]}
    receipt_ref=save(OUTPUT/("protocol_manifest.json" if stage=="protocols" else "curation_receipt.json"),receipt)
    print(json.dumps({"stage":stage,"cases":len(records),"compounds":receipt["compound_count"],
                      "endpoint_count":receipt["endpoint_count"],"receipt":receipt_ref},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["protocols","seal"],required=True)
    run(parser.parse_args().stage)
