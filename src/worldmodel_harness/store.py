"""SQLite evidence and versioned knowledge; historical rows are never overwritten."""

import json
import sqlite3
from pathlib import Path

from .knowledge import Claim, Dependency, Episode, Scope, fields_overlap


class KnowledgeStore:
    def __init__(self, path: Path, *, frozen: bool = False):
        self.frozen = frozen
        if frozen:
            self.db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        else:
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
                CREATE TABLE IF NOT EXISTS participants (
                    claim_id TEXT NOT NULL, version INTEGER NOT NULL, entity TEXT NOT NULL,
                    UNIQUE (claim_id, version, entity)
                );
                CREATE INDEX IF NOT EXISTS entity_lookup ON participants(entity);
                CREATE TABLE IF NOT EXISTS learning_tasks (
                    task_name TEXT PRIMARY KEY, phase TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS procedures (
                    id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY (id, version)
                );
            """)

    def close(self) -> None:
        self.db.close()

    def register_learning_task(self, task_name: str, phase: str) -> None:
        self._writable()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO learning_tasks VALUES (?, ?)", (task_name, phase)
            )

    def learning_tasks(self, *, exclude_validation: bool = False) -> set[str]:
        query = "SELECT task_name FROM learning_tasks"
        if exclude_validation:
            query += " WHERE phase != 'artifact_validation'"
        return {row[0] for row in self.db.execute(query)}

    def _writable(self) -> None:
        if self.frozen:
            raise PermissionError("Frozen knowledge cannot learn from evaluation tasks")

    def append_episode(self, episode: Episode) -> None:
        self._writable()
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

    def historical_episodes(self, scope: Scope) -> list[Episode]:
        """Historical examples across tasks; never eligible as current live facts."""
        records = [
            Episode.model_validate_json(row[0])
            for row in self.db.execute("SELECT body FROM episodes ORDER BY rowid")
        ]
        return [
            e
            for e in records
            if (e.scope.environment, e.scope.version, e.scope.role)
            == (scope.environment, scope.version, scope.role)
        ]

    def claim(self, claim_id: str, version: int | None = None) -> Claim:
        if version is None:
            row = self.db.execute(
                "SELECT body FROM claims WHERE id=? ORDER BY version DESC LIMIT 1", (claim_id,)
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT body FROM claims WHERE id=? AND version=?", (claim_id, version)
            ).fetchone()
        if row is None:
            raise KeyError(claim_id)
        return Claim.model_validate_json(row[0])

    def dependency_compatible(self, dependency: Dependency) -> bool:
        """Preserve pinned provenance while allowing unchanged required semantics.

        Every intervening revision must remain compatible: a changed-and-reverted
        contract cannot silently resurrect a previously invalidated artifact.
        Scope, freshness, and live conditions still require runtime checks.
        """
        try:
            original = self.claim(dependency.claim_id, dependency.version)
            current = self.claim(dependency.claim_id)
        except KeyError:
            return False

        def signature(claim):
            semantic = claim.model_dump(
                include={
                    "layer",
                    "relation",
                    "participants",
                    "scope",
                    "conditions",
                    "dependencies",
                    "origin",
                    "contradicting_episodes",
                    "invalidation_events",
                },
                mode="json",
            )
            if dependency.fields:
                values = {}
                for path in dependency.fields:
                    value = claim.payload
                    for part in path.split("."):
                        if not isinstance(value, dict) or part not in value:
                            raise KeyError(path)
                        value = value[part]
                    values[path] = value
                semantic["payload"] = values
            else:
                semantic.update(payload=claim.payload, statement=claim.statement)
            return json.dumps(semantic, sort_keys=True)

        try:
            expected = signature(original)
            return all(
                claim.status == "supported"
                and not claim.needs_review
                and signature(claim) == expected
                for claim in (
                    self.claim(dependency.claim_id, version)
                    for version in range(dependency.version, current.version + 1)
                )
            )
        except KeyError:
            return False

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
        self._writable()
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
            self.db.executemany(
                "INSERT INTO participants VALUES (?, ?, ?)",
                [(claim.claim_id, claim.version, p) for p in set(claim.participants)],
            )

    def related(self, participants: set[str], scope: Scope) -> list[Claim]:
        if not participants:
            return []
        placeholders = ",".join("?" for _ in participants)
        ids = {
            row[0]
            for row in self.db.execute(
                f"SELECT DISTINCT claim_id FROM participants WHERE entity IN ({placeholders})",
                tuple(sorted(participants)),
            )
        }
        return [c for c in self.claims(scope) if c.claim_id in ids]

    def procedures(self, scope: Scope | None = None):
        from .procedures import Procedure

        # Frozen pre-artifact development snapshots remain readable without migration.
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='procedures'").fetchone():
            return []
        rows = self.db.execute("""
            SELECT p.body FROM procedures p JOIN
            (SELECT id, MAX(version) AS version FROM procedures GROUP BY id) latest
            ON p.id=latest.id AND p.version=latest.version ORDER BY p.id
        """)
        records = [Procedure.model_validate_json(row[0]) for row in rows]
        return records if scope is None else [p for p in records if p.scope.applies_to(scope)]

    def put_procedure(self, procedure, *, tools=None) -> None:
        from .procedures import adopt

        self._writable()
        previous = next(
            (p for p in self.procedures() if p.procedure_id == procedure.procedure_id), None
        )
        if procedure.version != (previous.version + 1 if previous else 1):
            raise ValueError("Artifact versions must be contiguous and append-only")
        if previous and previous.scope != procedure.scope:
            raise ValueError("Scope changes require a new artifact identity")
        if procedure.status == "experimental":
            from .procedures import validate_candidate

            if tools is None:
                raise ValueError("Experimental artifacts require structural tool validation")
            validate_candidate(procedure, tools)
        if procedure.status == "active":
            if tools is None:
                raise ValueError("Activating an artifact requires tool and evidence validation")
            adopt(procedure, tools, self)
        with self.db:
            self.db.execute(
                "INSERT INTO procedures VALUES (?, ?, ?)",
                (procedure.procedure_id, procedure.version, procedure.model_dump_json()),
            )

    def suspend_dependents(self, claim_id: str, fields: set[str]) -> set[str]:
        self._writable()
        affected = set()
        for procedure in self.procedures():
            if procedure.status not in ("active", "experimental"):
                continue
            if any(
                d.claim_id == claim_id and d.mode == "requires" and fields_overlap(d.fields, fields)
                for d in procedure.required_claims
            ):
                self.put_procedure(
                    procedure.model_copy(
                        update={
                            "version": procedure.version + 1,
                            "status": "suspended",
                        }
                    )
                )
                affected.add(procedure.procedure_id)
        return affected
