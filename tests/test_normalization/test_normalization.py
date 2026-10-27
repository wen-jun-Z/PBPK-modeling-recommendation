from pbpk_kb.normalization.compound_resolver import CompoundResolver, normalize_name


def test_compound_resolver_combines_pubchem_and_label_identity() -> None:
    records = [
        {
            "source_id": "pubchem",
            "record_type": "compound_metadata",
            "source_location": "CID:4192",
            "record": {
                "pubchem_cid": "4192",
                "inchi_key": "DDLIGBOFAVUZHB-UHFFFAOYSA-N",
                "synonyms": ["Midazolam", "Versed"],
            },
        },
        {
            "source_id": "dailymed",
            "record_type": "label_detail",
            "source_location": "SPL_SET_ID:test",
            "record": {
                "title": "MIDAZOLAM HYDROCHLORIDE",
                "active_ingredients": [{"name": "Midazolam Hydrochloride"}],
                "active_moieties": [{"name": "Midazolam"}],
            },
        },
    ]

    resolved = CompoundResolver().resolve(records, query="midazolam")

    assert resolved.preferred_name == "Midazolam"
    assert resolved.normalized_name == "midazolam"
    assert resolved.pubchem_cid == "4192"
    assert resolved.inchi_key == "DDLIGBOFAVUZHB-UHFFFAOYSA-N"
    assert resolved.source_id == "pubchem"
    assert resolved.needs_review is False
    assert "Midazolam Hydrochloride" in resolved.synonyms


def test_compound_resolver_does_not_let_combination_label_override_pubchem_query() -> None:
    records = [
        {
            "source_id": "pubchem",
            "record_type": "compound_metadata",
            "source_location": "CID:2519",
            "record": {
                "pubchem_cid": "2519",
                "inchi_key": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                "synonyms": ["Caffeine"],
            },
        },
        {
            "source_id": "dailymed",
            "record_type": "label_detail",
            "source_location": "SPL_SET_ID:combo",
            "record": {
                "title": "BUTALBITAL, ACETAMINOPHEN, CAFFEINE",
                "active_ingredients": [{"name": "Butalbital"}, {"name": "Caffeine"}],
                "active_moieties": [{"name": "Butalbital"}, {"name": "Caffeine"}],
            },
        },
    ]

    resolved = CompoundResolver().resolve(records, query="caffeine")

    assert resolved.preferred_name == "Caffeine"
    assert resolved.normalized_name == "caffeine"
    assert resolved.pubchem_cid == "2519"


def test_compound_resolver_uses_query_match_inside_salt_name_when_pubchem_missing() -> None:
    records = [
        {
            "source_id": "dailymed",
            "record_type": "label_detail",
            "source_location": "SPL_SET_ID:combo",
            "record": {
                "title": "GLIPIZIDE AND METFORMIN HYDROCHLORIDE",
                "active_ingredients": [{"name": "Glipizide"}, {"name": "Metformin Hydrochloride"}],
                "active_moieties": [{"name": "Glipizide"}, {"name": "Metformin"}],
            },
        },
    ]

    resolved = CompoundResolver().resolve(records, query="metformin")

    assert resolved.preferred_name == "Metformin"
    assert resolved.normalized_name == "metformin"


def test_normalize_name_removes_punctuation_and_case() -> None:
    assert normalize_name("Midazolam HCl, Syrup") == "midazolam hcl syrup"


# [FIX 2026-07-02 · claude-opus-4-8] (F3) an identifier must come from its authoritative source, not any record.
def test_identifier_is_scoped_to_authoritative_source() -> None:
    records = [
        {
            "source_id": "pubchem",
            "record_type": "compound_metadata",
            "source_location": "CID:4192",
            # A stray chembl_id sitting on a pubchem record must be ignored.
            "record": {"pubchem_cid": "4192", "synonyms": ["Midazolam"], "chembl_id": "CHEMBL_WRONG"},
        },
        {
            "source_id": "chembl",
            "record_type": "compound_bioactivity",
            "source_location": "chembl:CHEMBL655",
            "record": {"molecule_chembl_id": "CHEMBL655"},
        },
    ]
    resolved = CompoundResolver().resolve(records, query="midazolam")
    assert resolved.chembl_id == "CHEMBL655"

from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value


def test_unit_normalizer_handles_common_pk_unit_aliases() -> None:
    value, unit = to_canonical_value(2, "days")
    assert value == 48
    assert unit == "h"

    value, unit = to_canonical_value(1.5, "µg/mL")
    assert value == 1500
    assert unit == "ng/mL"

    value, unit = to_canonical_value(3, "mg·h/L")
    assert value == 3000
    assert unit == "ng*h/mL"

    normalized = normalize_unit("mg/m²")
    assert normalized.canonical_unit == "mg/m2"
    assert normalized.dimension == "dose_per_bsa"


# [FIX 2026-07-02 · claude-opus-4-8] (F4) molar concentrations now canonicalize to nM (exact-case only).
def test_molar_concentrations_canonicalize_to_nm() -> None:
    value, unit = to_canonical_value(2, "µM")
    assert value == 2000
    assert unit == "nM"

    value, unit = to_canonical_value(1, "mM")
    assert value == 1_000_000
    assert unit == "nM"

    assert normalize_unit("nM").dimension == "molar_concentration"


# [FIX 2026-07-02 · claude-opus-4-8] (F4) ambiguous lowercase length units must NOT be mis-read as molar.
def test_length_like_units_stay_unknown_not_molar() -> None:
    assert normalize_unit("nm").dimension != "molar_concentration"
    assert normalize_unit("mm").dimension != "molar_concentration"


def test_fold_is_a_dimensionless_ratio_unit() -> None:
    normalized = normalize_unit("fold")

    assert normalized.dimension == "dimensionless_ratio"
    assert normalized.canonical_unit == "ratio"
