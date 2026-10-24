"""Evidence-grounded advisory reference investigation.

Extracts cross-service references, duplicate pointers, and uninspected policy
checks from task evidence and presents them as non-blocking advisory context.
Never blocks tool execution, raises gate exceptions, or forces actions.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _is_read(episode: Any) -> bool:
    if getattr(episode, "effect_kind", None) != "unknown":
        return episode.effect_kind == "read"
    op = getattr(episode, "operation", "")
    args = getattr(episode, "arguments", {}) or {}
    return op == "api_search" or (
        op == "api_fetch" and str(args.get("method", "GET")).upper() in ("GET", "HEAD")
    )


def _already_inspected(
    target_terms: set[str],
    episodes: list[Any],
    *,
    exclude_episode_id: str | None = None,
    require_in_arguments: bool = False,
) -> bool:
    """Check if any successful read episode already inspected the target terms."""
    for ep in episodes:
        if exclude_episode_id is not None and getattr(ep, "episode_id", None) == exclude_episode_id:
            continue
        if getattr(ep, "outcome", "") != "success" or not _is_read(ep):
            continue
        args_text = json.dumps(getattr(ep, "arguments", {})).lower()
        if require_in_arguments:
            if any(term in args_text for term in target_terms if len(term) >= 3):
                return True
        else:
            resp_text = json.dumps(getattr(ep, "response", {})).lower()
            combined = args_text + " " + resp_text
            if any(term in combined for term in target_terms if len(term) >= 3):
                return True
    return False


def extract_unresolved_references(adapter: Any) -> list[dict[str, str]]:
    """Extract evidence-grounded references that have not yet been investigated."""
    unresolved: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    # Get all successful environment read episodes
    all_episodes = []
    if hasattr(adapter, "evidence") and hasattr(adapter.evidence, "db"):
        from .knowledge import Episode

        try:
            with adapter.evidence.db:
                rows = adapter.evidence.db.execute(
                    "SELECT body FROM episodes WHERE scope = ? ORDER BY rowid",
                    (adapter.scope.model_dump_json(),),
                ).fetchall()
                all_episodes = [Episode.model_validate_json(r[0]) for r in rows]
        except Exception:
            all_episodes = []
    if not all_episodes and hasattr(adapter, "task_state"):
        all_episodes = list(adapter.task_state.observations.values())

    # 1. Prompt policy check references
    prompt_text = ""
    if hasattr(adapter, "task") and hasattr(adapter.task, "prompt"):
        prompt_text = " ".join(
            m.get("content", "") for m in adapter.task.prompt if m.get("role") != "system"
        )
    if prompt_text:
        m_policy = re.search(
            r"(?:check|consult|follow)\s+(?:the\s+)?(?:email\s+processing\s+|notification\s+)?policy\s+in\s+spreadsheet\s+([A-Za-z0-9_-]+)(?:\s+\(worksheet\s+([A-Za-z0-9_-]+)\))?",
            prompt_text,
            re.IGNORECASE,
        )
        if m_policy:
            ss_id = m_policy.group(1)
            ws_id = m_policy.group(2) or ""
            target_terms = {ws_id.lower()} if ws_id else {ss_id.lower()}
            if not _already_inspected(target_terms, all_episodes, require_in_arguments=True):
                key = f"prompt_policy_{ss_id}_{ws_id}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    loc = f"{ss_id} ({ws_id})" if ws_id else ss_id
                    unresolved.append({
                        "category": "policy",
                        "target": loc,
                        "advice": f"Task requests checking policy in '{loc}' before taking action (not yet read).",
                    })

    # 2. Spreadsheet metadata policy sheets
    for ep in all_episodes:
        if not _is_read(ep) or getattr(ep, "outcome", "") != "success":
            continue
        resp = getattr(ep, "response", {}) or {}
        if isinstance(resp, dict) and "sheets" in resp:
            for s in resp["sheets"]:
                props = s.get("properties", {})
                title = props.get("title", "")
                if "policy" in title.lower():
                    ss_id = resp.get("spreadsheetId", "")
                    if not _already_inspected({title.lower()}, all_episodes, require_in_arguments=True):
                        key = f"sheet_policy_{title}"
                        if key not in seen_keys:
                            seen_keys.add(key)
                            unresolved.append({
                                "category": "policy_sheet",
                                "target": title,
                                "advice": f"Spreadsheet '{ss_id}' contains policy worksheet '{title}' with special handling rules (not yet read).",
                            })

    # 3. References in observed tabular records (e.g. Notes column in incident queue or badge tracker)
    for ep in all_episodes:
        if not _is_read(ep) or getattr(ep, "outcome", "") != "success":
            continue
        resp = getattr(ep, "response", {}) or {}
        values = resp.get("values", []) if isinstance(resp, dict) else []
        if not isinstance(values, list) or len(values) < 2:
            continue
        headers = [str(h).strip().lower() for h in values[0]]
        notes_idx = next(
            (i for i, h in enumerate(headers) if "note" in h or "comment" in h or "rule" in h),
            None,
        )
        title_idx = next(
            (i for i, h in enumerate(headers) if any(k in h for k in ("title", "contractor", "name", "id"))),
            0,
        )
        if notes_idx is None:
            continue

        source_ep_id = getattr(ep, "episode_id", None)
        for row in values[1:]:
            if not isinstance(row, list) or len(row) <= notes_idx:
                continue
            note = str(row[notes_idx]).strip()
            item_name = str(row[title_idx]).strip() if len(row) > title_idx else "Record"
            if not note:
                continue

            # Check duplicate / superseded
            m_dup = re.search(
                r"(?:duplicate|superseded|merged)\s+(?:of|by|into)\s+([A-Za-z0-9_-]+)",
                note,
                re.IGNORECASE,
            )
            if m_dup:
                dup_id = m_dup.group(1).strip()
                if not _already_inspected({dup_id.lower()}, all_episodes, exclude_episode_id=source_ep_id):
                    key = f"dup_{item_name}_{dup_id}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unresolved.append({
                            "category": "duplicate_reference",
                            "target": item_name,
                            "advice": f"Record '{item_name}' notes '{note}' — verify whether {dup_id} already resolves this before acting.",
                        })

            # Check external investigation pointer (see ... email / check ...)
            m_see = re.search(
                r"(?:see|check|refer\s+to)\s+([^,.;\n]+)",
                note,
                re.IGNORECASE,
            )
            if m_see:
                ext_target = m_see.group(1).strip()
                target_words = set(re.findall(r"\b[A-Za-z0-9_-]+\b", ext_target.lower())) - {
                    "the", "a", "an", "and", "or", "for", "to", "in", "at", "see", "check"
                }
                distinctive_words = {
                    w for w in target_words
                    if len(w) >= 5 and w not in {"email", "emails", "message", "messages", "team", "inbox"}
                }
                check_terms = {ext_target.lower()} | distinctive_words
                if not _already_inspected(check_terms, all_episodes, exclude_episode_id=source_ep_id):
                    key = f"see_{item_name}_{ext_target}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unresolved.append({
                            "category": "external_investigation",
                            "target": item_name,
                            "advice": f"Record '{item_name}' notes '{note}' — inspect {ext_target} to determine true status/severity before committing.",
                        })

            # Check review hold / reference code (e.g. SEC-2026-015 review)
            m_rev = re.search(
                r"(?:under|pending)\s+([A-Za-z0-9_-]+)\s+review",
                note,
                re.IGNORECASE,
            )
            if m_rev:
                code = m_rev.group(1).strip()
                key = f"hold_{item_name}_{code}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    unresolved.append({
                        "category": "review_hold",
                        "target": item_name,
                        "advice": f"Record '{item_name}' notes '{note}' — check applicable hold policy regarding {code} before notifying.",
                    })

            # Check do not page / do not contact
            m_dnp = re.search(
                r"do\s+not\s+(?:page|notify|action|contact)",
                note,
                re.IGNORECASE,
            )
            if m_dnp:
                key = f"dnp_{item_name}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    unresolved.append({
                        "category": "exclusion_note",
                        "target": item_name,
                        "advice": f"Record '{item_name}' notes '{note}' — do not page or act on this record.",
                    })

    return unresolved


def advisory_reference_context(adapter: Any, *, max_chars: int = 1500) -> str:
    """Render non-blocking advisory context for unresolved references and policies."""
    unresolved = extract_unresolved_references(adapter)
    if not unresolved:
        return ""

    lines = ["[Advisory Reference & Dependency Guidance - Non-blocking]:"]
    for item in unresolved:
        line = f"• {item['advice']}"
        if sum(len(l) + 1 for l in lines) + len(line) <= max_chars:
            lines.append(line)
        else:
            break

    return "\n".join(lines) if len(lines) > 1 else ""
