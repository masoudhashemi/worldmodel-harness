"""Immutable measurement checkpoints with disposable per-attempt knowledge."""

import hashlib
import sqlite3
from contextlib import closing


def checkpoint_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_checkpoint(source, destination):
    if not source.is_file() or destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("Require an existing checkpoint and a fresh distinct trial destination")
    before = checkpoint_hash(source)
    with (
        closing(sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)) as src,
        closing(sqlite3.connect(destination)) as dest,
    ):
        src.backup(dest)
    if checkpoint_hash(source) != before:
        raise ValueError("Checkpoint changed during snapshot")
    return before
