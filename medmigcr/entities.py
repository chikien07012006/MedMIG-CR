"""Canonical entity keys for PrimeKG rows."""

from __future__ import annotations

import pandas as pd


def entity_key(node_type: str, source: str, node_id) -> str:
    src = "" if pd.isna(source) else str(source).strip()
    nid = "" if pd.isna(node_id) else str(node_id).strip()
    return f"{node_type}|{src}|{nid}"


def head_key(row: pd.Series) -> str:
    return entity_key(str(row["x_type"]), str(row["x_source"]), row["x_id"])


def tail_key(row: pd.Series) -> str:
    return entity_key(str(row["y_type"]), str(row["y_source"]), row["y_id"])
