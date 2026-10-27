from pathlib import Path

import pytest

from pbpk_kb.storage.import_licensed_model import (
    LicensedModelImportError,
    load_licensed_model_bundle,
    stable_uuid,
)


def test_stable_uuid_is_deterministic_and_kind_scoped() -> None:
    first = stable_uuid("model", "licensed-model:1")
    second = stable_uuid("model", "licensed-model:1")
    different_kind = stable_uuid("version", "licensed-model:1")

    assert first == second
    assert first != different_kind


def test_bundle_loader_rejects_incomplete_bundle(tmp_path: Path) -> None:
    with pytest.raises(
        LicensedModelImportError,
        match="Required bundle file is missing",
    ):
        load_licensed_model_bundle(tmp_path)
