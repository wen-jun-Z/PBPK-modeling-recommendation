from pbpk_kb.graph.final_evidence_graph import extract_graph_facts, find_target_mentions


def evidence(text: str, *, compound: str = "midazolam", role: str = "target_compound") -> dict:
    return {
        "evidence_type": "adme",
        "source_text": text,
        "attributed_compound_name": compound,
        "compound_role": role,
    }


def test_extracts_direct_enzyme_substrate_relationship() -> None:
    facts = extract_graph_facts(evidence("Midazolam is metabolized by CYP3A4."))

    assert [(fact.target.key, fact.relationship_type) for fact in facts] == [("CYP3A4", "substrate_of")]


def test_extracts_transporter_relationships_with_normalized_identity() -> None:
    facts = extract_graph_facts(
        evidence(
            "Metformin is transported into hepatocytes by OCT1 and is a substrate of MATE1.",
            compound="metformin",
        )
    )

    assert {(fact.target.key, fact.relationship_type) for fact in facts} == {
        ("SLC22A1", "transported_by"),
        ("SLC47A1", "substrate_of"),
    }


def test_does_not_assign_other_drug_inhibition_to_target_compound() -> None:
    facts = extract_graph_facts(
        evidence(
            "Fluvoxamine inhibits CYP1A2 and impairs the clearance of caffeine.",
            compound="caffeine",
            role="victim_drug",
        )
    )

    assert facts == []


def test_preserves_explicit_negative_substrate_relation() -> None:
    facts = extract_graph_facts(evidence("Digoxin is not a substrate of BCRP.", compound="digoxin"))

    assert [(fact.target.key, fact.relationship_type) for fact in facts] == [("ABCG2", "not_substrate_of")]


def test_finds_common_transporter_aliases() -> None:
    mentions = find_target_mentions("P-gp, BCRP, OATP1B1, and CYP2C19 were evaluated.")

    assert {mention.key for mention in mentions} == {"ABCB1", "ABCG2", "SLCO1B1", "CYP2C19"}

from pbpk_kb.graph.graph_loader import _has_relevant_target, _relationship_type


def test_chembl_loader_rejects_non_target_activity() -> None:
    assert not _has_relevant_target(
        {
            "target_chembl_id": "CHEMBL2362975",
            "target_pref_name": "No relevant target",
        }
    )
    assert not _has_relevant_target(
        {
            "target_chembl_id": "CHEMBL376",
            "target_pref_name": "Rattus norvegicus",
            "target_organism": "Rattus norvegicus",
        }
    )


def test_chembl_loader_keeps_real_protein_target() -> None:
    activity = {
        "target_chembl_id": "CHEMBL287",
        "target_pref_name": "Sigma non-opioid intracellular receptor 1",
        "target_organism": "Homo sapiens",
        "standard_type": "Ki",
    }

    assert _has_relevant_target(activity)
    assert _relationship_type(activity) == "inhibits_or_binds"
