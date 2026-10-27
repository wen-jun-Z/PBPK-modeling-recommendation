from __future__ import annotations

from pbpk_kb.recommendation.model_matching.repository_version_census import (
    parse_remote_tags,
)


def test_remote_tag_parser_prefers_peeled_annotated_commit() -> None:
    output = "\n".join(
        [
            f"{'a' * 40}\trefs/tags/v1.0",
            f"{'b' * 40}\trefs/tags/v2.0",
            f"{'c' * 40}\trefs/tags/v2.0^{{}}",
        ]
    )

    assert parse_remote_tags(output) == {
        "v1.0": "a" * 40,
        "v2.0": "c" * 40,
    }
