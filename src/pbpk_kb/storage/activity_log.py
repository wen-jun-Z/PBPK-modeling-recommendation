from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def record_user_activity(
    db: Session,
    *,
    user_identifier: str | None,
    activity_type: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            INSERT INTO user_activity_log (
                user_identifier,
                activity_type,
                entity_type,
                entity_id,
                details
            )
            VALUES (
                :user_identifier,
                :activity_type,
                :entity_type,
                CAST(:entity_id AS uuid),
                CAST(:details AS jsonb)
            )
            RETURNING
                user_activity_log_id::text,
                user_identifier,
                activity_type,
                entity_type,
                entity_id::text,
                details,
                created_at
            """
        ),
        {
            "user_identifier": user_identifier,
            "activity_type": activity_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "details": json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
        },
    ).mappings().one()
    return dict(row)
