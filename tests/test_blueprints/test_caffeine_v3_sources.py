from __future__ import annotations

import json
from pathlib import Path

from pbpk_kb.blueprints.caffeine_v3_sources import (
    extract_caffeine_absorption_evidence,
)


def test_extracts_constraints_without_direct_permeability_mapping(
    tmp_path: Path,
) -> None:
    pksim = tmp_path / "pksim.html"
    pksim.write_text(
        "Specific intestinal permeability: calibration of these in vitro "
        "values has to be performed for a defined set of compounds.",
        encoding="utf-8",
    )
    comments = tmp_path / "comments.json"
    comments.write_text(
        json.dumps(
            [
                {"body": "Pint from PK-Sim only makes sense in PK-Sim."},
                {"body": "There is NO problem with unit conversion."},
            ]
        ),
        encoding="utf-8",
    )
    caco2 = tmp_path / "caco2.xml"
    caco2.write_text(
        """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>20027153</PMID><Article><Abstract><AbstractText Label="RESULTS">
In AP-BL direction Papp = 46.3-53.5 x 10-6 cm/s; in BL-AP direction
Papp = 45.6-49.4 x 10-6 cm/s. Thus, the transport seems to be
transcellular mediated by passive diffusion.</AbstractText></Abstract></Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>""",
        encoding="utf-8",
    )
    bioavailability = tmp_path / "bioavailability.xml"
    bioavailability.write_text(
        """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>6832208</PMID><Article><Abstract><AbstractText>
The peak (Tp) plasma concentration after 29.8 +/- 8.1 min. The result was
F(%) = 108.3 +/- 3.6%.</AbstractText></Abstract></Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>""",
        encoding="utf-8",
    )
    human = tmp_path / "human.xml"
    human.write_text(
        """<article><abstract><p>The small uncharged solutes caffeine,
acetaminophen and antipyrine have very high permeabilities
(about 20 x 10<sup>-4</sup> cm/sec).</p></abstract></article>""",
        encoding="utf-8",
    )

    evidence = extract_caffeine_absorption_evidence(
        pksim_doc_path=pksim,
        osp_comments_path=comments,
        caco2_pubmed_path=caco2,
        bioavailability_pubmed_path=bioavailability,
        human_permeability_path=human,
    )

    by_id = {item.evidence_id: item for item in evidence}
    assert len(evidence) == 7
    assert by_id["pksim_in_vitro_values_require_calibration"].use == (
        "prohibit_direct_mapping"
    )
    assert by_id["caffeine_caco2_ap_bl_high"].lower == 46.3
    assert by_id["caffeine_caco2_ap_bl_high"].upper == 53.5
    assert by_id["caffeine_caco2_ap_bl_high"].use == "structure_only"
    assert by_id["blanchard_absolute_bioavailability"].value == 108.3
    assert by_id["blanchard_oral_tmax"].value == 29.8
