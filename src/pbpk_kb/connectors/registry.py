from __future__ import annotations

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.chembl import ChEMBLConnector
from pbpk_kb.connectors.clinicaltrials import ClinicalTrialsConnector
from pbpk_kb.connectors.dailymed import DailyMedConnector
from pbpk_kb.connectors.drugcentral import DrugCentralConnector
from pbpk_kb.connectors.drugs_fda import DrugsFDAConnector
from pbpk_kb.connectors.hpa import HumanProteinAtlasConnector
from pbpk_kb.connectors.osp_model_library import OSPModelLibraryConnector
from pbpk_kb.connectors.pkdb import PKDBConnector
from pbpk_kb.connectors.pubchem import PubChemConnector
from pbpk_kb.connectors.pubmed import PubMedConnector
from pbpk_kb.connectors.rxnorm import RxNormConnector


CONNECTOR_REGISTRY: dict[str, type[BaseConnector]] = {
    "pubmed": PubMedConnector,
    "dailymed": DailyMedConnector,
    "pubchem": PubChemConnector,
    "osp_pbpk_model_library": OSPModelLibraryConnector,
    "pkdb": PKDBConnector,
    "drugs_fda": DrugsFDAConnector,
    "rxnorm": RxNormConnector,
    "drugcentral": DrugCentralConnector,
    "chembl": ChEMBLConnector,
    "clinicaltrials": ClinicalTrialsConnector,
    "hpa": HumanProteinAtlasConnector,
}


def get_connector(source_id: str) -> type[BaseConnector]:
    try:
        return CONNECTOR_REGISTRY[source_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported connector source_id: {source_id}") from exc
