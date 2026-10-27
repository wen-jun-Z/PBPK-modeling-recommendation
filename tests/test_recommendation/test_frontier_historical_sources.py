"""English report changes must not alter frozen inference or source guards."""
import ast
from pathlib import Path

import pytest

from frozen_frontier_sources import ROOT, SOURCE_SHA256, historical_sha, source_blob


@pytest.mark.parametrize("name", SOURCE_SHA256)
def test_original_sources_remain_available_with_exact_fingerprints(name):
    assert historical_sha(name) == SOURCE_SHA256[name]


def scientific_ast(source, report_function):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 or node.name != report_function]
    return ast.dump(tree, include_attributes=False)


@pytest.mark.parametrize("name", SOURCE_SHA256)
def test_english_revision_changes_only_the_named_report_function(name):
    report = "write_readme_results" if name == "run_frontier_assessment.py" else "publish_summary"
    original = source_blob(name)
    current = (ROOT / "scripts" / name).read_bytes()
    assert scientific_ast(current, report) == scientific_ast(original, report)


@pytest.mark.parametrize("path", ["../run_frontier_scoring.py", "/tmp/source.py", "nested/source.py"])
def test_historical_lookup_rejects_nonproject_paths(path):
    with pytest.raises(ValueError, match="top-level project"):
        source_blob(path)
