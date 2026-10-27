from __future__ import annotations

from pathlib import Path

import pytest

from pbpk_kb.blueprints.caffeine_v2_sources import (
    CaffeineV2SourceError,
    _extract_pathway_fractions,
    _extract_search_domains,
)


def test_extracts_locked_human_demethylation_fractions(tmp_path: Path) -> None:
    source = tmp_path / "pubmed.xml"
    source.write_text(
        """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>3756066</PMID>
      <Article><Abstract><AbstractText>
        When only demethylation pathways are considered PX, TB and TP
        accounted for 83.9 +/- 5.4%, 12.1 +/- 4.1% and 4.0 +/- 1.4%,
        respectively of the CA demethylations.
      </AbstractText></Abstract></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
""",
        encoding="utf-8",
    )

    fractions = _extract_pathway_fractions(source)

    assert [item.pathway_id for item in fractions] == [
        "n3_paraxanthine",
        "n1_theobromine",
        "n7_theophylline",
    ]
    assert [item.fraction for item in fractions] == pytest.approx(
        [0.839, 0.121, 0.04]
    )
    assert sum(item.fraction for item in fractions) == pytest.approx(1.0)


def test_extracts_search_domains_without_importing_fitted_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "parameter_identification.xml"
    source.write_text(
        """<?xml version="1.0"?>
<ParameterIdentification xmlns="http://www.systems-biology.com">
  <IdentificationParameterList>
    <IdentificationParameter name="Specific clearance"
      startValue="0.035848049857049" minValue="0"
      maxValue="0.35848049857049" dimension="Inversed time" />
    <IdentificationParameter name="Intestinal permeability (transcellular)"
      startValue="3.85111864778853E-06" minValue="0"
      maxValue="3.85111864778853E-05" dimension="Velocity" />
  </IdentificationParameterList>
</ParameterIdentification>
""",
        encoding="utf-8",
    )

    domains = _extract_search_domains(source)

    assert [item.parameter_id for item in domains] == [
        "total_cyp1a2_specific_clearance",
        "specific_intestinal_permeability_transcellular",
    ]
    assert [item.lower_bound for item in domains] == [0.0, 0.0]
    assert [item.upper_bound for item in domains] == pytest.approx(
        [0.35848049857049, 3.85111864778853e-05]
    )
    assert all(item.use == "search_domain_only" for item in domains)


def test_rejects_drifted_search_domain(tmp_path: Path) -> None:
    source = tmp_path / "parameter_identification.xml"
    source.write_text(
        """<ParameterIdentification xmlns="http://www.systems-biology.com">
  <IdentificationParameterList>
    <IdentificationParameter name="Specific clearance" minValue="0"
      maxValue="1" />
    <IdentificationParameter name="Intestinal permeability (transcellular)"
      minValue="0" maxValue="3.85111864778853E-05" />
  </IdentificationParameterList>
</ParameterIdentification>""",
        encoding="utf-8",
    )

    with pytest.raises(CaffeineV2SourceError, match="search domain drifted"):
        _extract_search_domains(source)
