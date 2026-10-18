"""Lossless structural compression of mandatory state; never prefix truncation."""

import json


class MandatoryContextOverflow(ValueError):
    """The complete required state cannot fit the declared context allowance."""


def pack_records(records):
    """Dictionary-encode repeated objects without losing keys, types or values.

    A columns/rows representation removes repeated field names. Full original
    records can be reconstructed with dict(zip(columns, row)).
    """
    if not records:
        return {"columns": [], "rows": []}
    columns = list(records[0])
    if any(set(row) != set(columns) for row in records):
        # Differently shaped records stay intact; missing is not the same as null.
        return {"records": records}
    return {"columns": columns, "rows": [[row[key] for key in columns] for row in records]}


def encode_records(records):
    return json.dumps(pack_records(records), separators=(",", ":"), ensure_ascii=False)
