"""SQLite evidence and versioned knowledge; historical rows are never overwritten."""

import sqlite3
from pathlib import Path

from .knowledge import Claim, Episode, Scope


class KnowledgeStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, body TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL,
                PRIMARY KEY (id, version)
            );
        """)

    def close(self) -> None:
        self.db.close()

    def append_episode(self, episode: Episode) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO episodes VALUES (?, ?, ?)",
                (episode.episode_id, episode.event_id, episode.model_dump_json()),
            )

    def episode(self, episode_id: str) -> Episode:
        row = self.db.execute("SELECT body FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return Episode.model_validate_json(row[0])

    def episodes(self, scope: Scope) -> list[Episode]:
        records = [
            Episode.model_validate_json(row[0])
            for row in self.db.execute("SELECT body FROM episodes ORDER BY rowid")
        ]
        return [e for e in records if e.scope.applies_to(scope)]

    def claim(self, claim_id: str) -> Claim:
        row = self.db.execute(
            "SELECT body FROM claims WHERE id=? ORDER BY version DESC LIMIT 1", (claim_id,)
        ).fetchone()
        if row is None:
            raise KeyError(claim_id)
        return Claim.model_validate_json(row[0])

    def claims(self, scope: Scope | None = None) -> list[Claim]:
        rows = self.db.execute("""
            SELECT c.body FROM claims c JOIN
            (SELECT id, MAX(version) AS version FROM claims GROUP BY id) latest
            ON c.id=latest.id AND c.version=latest.version ORDER BY c.id
        """)
        records = [Claim.model_validate_json(row[0]) for row in rows]
        return records if scope is None else [
            c for c in records if c.scope.applies_to(scope)
            and (not c.payload.get("workspace_fact")
                 or (c.payload.get("source_tenant") is not None
                     and c.payload["source_tenant"] == scope.tenant))
        ]

    def put_claim(self, claim: Claim) -> None:
        try:
            previous = self.claim(claim.claim_id)
        except KeyError:
            previous = None
        if claim.version != (previous.version + 1 if previous else 1):
            raise ValueError("Claim versions must be contiguous and append-only")
        if previous and claim.scope != previous.scope:
            raise ValueError("Scope changes require a new claim identity")
        if claim.status == "supported" and not claim.supporting_episodes:
            raise ValueError("Supported claims require observed evidence")
        for episode_id in (*claim.supporting_episodes, *claim.contradicting_episodes):
            episode = self.episode(episode_id)
            if not claim.scope.applies_to(episode.scope):
                raise ValueError("Claim evidence has incompatible scope")
        with self.db:
            self.db.execute(
                "INSERT INTO claims VALUES (?, ?, ?)",
                (claim.claim_id, claim.version, claim.model_dump_json()),
            )
