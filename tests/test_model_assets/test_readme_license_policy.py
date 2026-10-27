from __future__ import annotations

from pathlib import Path

from pbpk_kb.model_assets.license_policy import resolve_repository_license
from pbpk_kb.model_assets.models import LicenseStatus


def test_explicit_model_license_statement_in_readme_is_traceable(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "## License\nThe model code is distributed under the GPLv2 License.\n",
        encoding="utf-8",
    )

    decision = resolve_repository_license(tmp_path)

    assert decision.status == LicenseStatus.ALLOWED
    assert decision.spdx_id == "GPL-2.0-only"
    assert decision.source_location == "README.md"
    assert decision.reason_code == "MODEL_LICENSE_STATEMENT_EXPLICIT"
