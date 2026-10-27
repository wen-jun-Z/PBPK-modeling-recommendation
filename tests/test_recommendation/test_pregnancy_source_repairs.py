"""Pregnancy source preservation boundaries, without observations or an ODE call."""
import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from recommendation_execution_repairs import pregnancy_export as repair
from recommendation_execution_repairs import pregnancy_source_assets as assets


def document():
    return ET.fromstring('''<PopulationSimulation id="original" name="Pregnant cohort" xmlVersion="77">
      <Model><Root><Children><Container name="Organism"><Children>
        <Container name="Fetus"/><Container name="PlacentaMaternal"/>
        <Container name="PlacentaFetal"/><Container name="AmnioticFluid"/>
        <Para name="gestational age" value="35"/>
      </Children></Container></Children></Root><FormulaCache/></Model>
      <Properties><ModelProperties origin="MoBi"/></Properties>
      <UsedBuildingBlocks><UsedBuildingBlock><RandomPopulation id="pop" name="Pregnant" seed="31">
        <Settings numberOfIndividuals="1000"/>
      </RandomPopulation></UsedBuildingBlock></UsedBuildingBlocks>
      <UsedObservedData><Reference id="not-read"/></UsedObservedData>
    </PopulationSimulation>''')


def test_population_source_is_retained_but_never_reported_as_executed():
    row = repair.structural_inventory(document())
    assert row['complete_pregnancy_marker_set_present']
    assert row['parameter_element_count'] == 1
    assert row['populations'][0]['declared_number_of_individuals'] == '1000'
    assert row['population_covariates_applied_to_native_instance'] is False
    assert row['scientific_applicability_inferred'] is False


def test_saved_population_inputs_exclude_observation_references(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, 'ROOT', tmp_path)
    source = document()
    ET.SubElement(source.find('Properties'), 'Parameter', {'name': 'Unit metadata', 'displayUnit': 'µmol/l'})
    before = ET.tostring(source)
    row = repair.preserve_population_inputs(source, tmp_path)
    assert ET.tostring(source) == before
    saved = ET.parse(tmp_path / row['path']).getroot()
    assert saved.find('UsedObservedData') is None
    assert saved.find('.//RandomPopulation/Settings').get('numberOfIndividuals') == '1000'
    assert row['population_execution_claimed'] is False
    assert row['original_element_sha256']['UsedBuildingBlocks'] == repair.source_pkml.element_sha(source.find('UsedBuildingBlocks'))


def test_outcome_data_cannot_be_hidden_in_population_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, 'ROOT', tmp_path)
    source = document()
    ET.SubElement(source.find('UsedBuildingBlocks'), 'DataRepository')
    with pytest.raises(ValueError, match='Outcome data'):
        repair.preserve_population_inputs(source, tmp_path)


def test_source_path_rejects_outside_and_sealed():
    with pytest.raises(ValueError, match='inside this project'):
        assets.inside(ROOT.parent / 'outside')
    with pytest.raises(ValueError, match='Sealed'):
        assets.inside(ROOT / 'sealed' / 'target.json')
