from __future__ import annotations

import json
from numbers import Number
from typing import Any

import streamlit as st


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} | PBPK KB", page_icon=None, layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d8dee8;
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            background: #ffffff;
        }
        div[data-testid="stMetricLabel"] p {
            color: #53606f;
            font-size: 0.82rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        .pbpk-small {
            color: #53606f;
            font-size: 0.86rem;
        }
        .pbpk-status {
            border-left: 4px solid #476582;
            padding: 0.55rem 0.75rem;
            background: #f7f9fc;
            margin: 0.25rem 0 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def status_note(text: str) -> None:
    st.markdown(f'<div class="pbpk-status">{text}</div>', unsafe_allow_html=True)


def serialize_table_cell(value: Any) -> Any:
    if isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    return value


def normalize_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        {key: serialize_table_cell(value) for key, value in row.items()}
        for row in rows
    ]
    columns = {key for row in normalized for key in row}
    for column in columns:
        values = [row.get(column) for row in normalized if row.get(column) is not None]
        kinds = {_arrow_value_kind(value) for value in values}
        if len(kinds) <= 1:
            continue
        for row in normalized:
            value = row.get(column)
            if value is not None:
                row[column] = str(value)
    return normalized


def _arrow_value_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Number):
        return "number"
    if isinstance(value, bytes):
        return "bytes"
    return "text"


def display_table(
    rows: list[dict[str, Any]],
    *,
    height: int = 420,
    hide_index: bool = True,
) -> None:
    if rows:
        st.dataframe(
            normalize_table_rows(rows),
            width="stretch",
            height=height,
            hide_index=hide_index,
        )
    else:
        st.info("No records match the current filters.")
