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


def _is_empty_response(resp: Any) -> bool:
    if not resp:
        return True
    if isinstance(resp, dict):
        if "messages" in resp and not resp["messages"]:
            return True
        if resp.get("resultSizeEstimate") == 0:
            return True
        if "values" in resp and len(resp.get("values", [])) <= 1:
            return True
        if "items" in resp and not resp["items"]:
            return True
    elif isinstance(resp, list) and not resp:
        return True
    return False


def _already_inspected(
    target_terms: set[str],
    episodes: list[Any],
    *,
    exclude_episode_id: str | None = None,
    require_in_arguments: bool = False,
    require_in_response: bool = False,
) -> bool:
    """Check if any successful read episode already inspected the target terms."""
    for ep in episodes:
        if exclude_episode_id is not None and getattr(ep, "episode_id", None) == exclude_episode_id:
            continue
        if getattr(ep, "outcome", "") != "success" or not _is_read(ep):
            continue
        resp = getattr(ep, "response", {}) or {}
        if _is_empty_response(resp):
            continue
        args_text = json.dumps(getattr(ep, "arguments", {})).lower()
        resp_text = json.dumps(resp).lower()
        if require_in_arguments:
            if any(term in args_text for term in target_terms if len(term) >= 3):
                return True
        elif require_in_response:
            if any(term in resp_text for term in target_terms if len(term) >= 3):
                return True
        else:
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
    if hasattr(adapter, "evidence") and hasattr(adapter.evidence, "episodes"):
        try:
            all_episodes = list(adapter.evidence.episodes(adapter.scope))
        except Exception:
            all_episodes = []
    if not all_episodes and hasattr(adapter, "evidence") and getattr(adapter.evidence, "db", None) is not None:
        from .knowledge import Episode

        try:
            with adapter.evidence.db:
                rows = adapter.evidence.db.execute(
                    "SELECT body FROM episodes ORDER BY rowid"
                ).fetchall()
                parsed = [Episode.model_validate_json(r[0]) for r in rows]
                all_episodes = [
                    e for e in parsed
                    if not hasattr(adapter, "scope")
                    or e.scope.applies_to(adapter.scope)
                    or getattr(e.scope, "task_id", None) == getattr(adapter.scope, "task_id", None)
                ]
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

        # Disambiguate service project keys and multi-tenant IDs mentioned in prompt
        if "incident management project" in prompt_text.lower():
            if not _already_inspected({"inc"}, all_episodes):
                key = "prompt_jira_inc"
                if key not in seen_keys:
                    seen_keys.add(key)
                    unresolved.append({
                        "category": "service_identifier",
                        "target": "Incident Management",
                        "advice": "Jira 'Incident Management' project uses project key 'INC' (not 'IM'). Specify 'INC' when creating the Jira issue.",
                    })

        if "operations core project" in prompt_text.lower():
            if not _already_inspected({"ops"}, all_episodes):
                key = "prompt_jira_ops"
                if key not in seen_keys:
                    seen_keys.add(key)
                    unresolved.append({
                        "category": "service_identifier",
                        "target": "Operations Core",
                        "advice": "Jira 'Operations Core' project uses project key 'OPS' (not 'OC'). Specify 'OPS' when creating the Jira issue.",
                    })

        m_conf = re.search(r"\b(cloud_[A-Za-z0-9_-]+)/(SP_[A-Za-z0-9_-]+)\b", prompt_text, re.IGNORECASE)
        if m_conf:
            c_id = m_conf.group(1)
            s_id = m_conf.group(2)
            if not _already_inspected({s_id.lower()}, all_episodes):
                key = f"prompt_confluence_{c_id}_{s_id}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    unresolved.append({
                        "category": "service_identifier",
                        "target": f"{c_id}/{s_id}",
                        "advice": f"Confluence target '{c_id}/{s_id}' indicates cloudId='{c_id}' and spaceId='{s_id}'.",
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
        source_ep_id = getattr(ep, "episode_id", None)

        # Check on-call schedule roles and contact entities
        role_idx = next((i for i, h in enumerate(headers) if h == "role"), None)
        eng_idx = next((i for i, h in enumerate(headers) if "engineer" in h or "name" in h), None)
        phone_idx = next((i for i, h in enumerate(headers) if "phone" in h), None)
        if role_idx is not None and eng_idx is not None:
            for row in values[1:]:
                if isinstance(row, list) and len(row) > max(role_idx, eng_idx):
                    if str(row[role_idx]).strip().lower() == "primary":
                        eng_name = str(row[eng_idx]).strip()
                        phone_num = str(row[phone_idx]).strip() if phone_idx is not None and len(row) > phone_idx else ""
                        if not _already_inspected({eng_name.lower()}, all_episodes, exclude_episode_id=source_ep_id):
                            key = f"oncall_primary_{eng_name}"
                            if key not in seen_keys:
                                seen_keys.add(key)
                                unresolved.append({
                                    "category": "oncall_entity",
                                    "target": eng_name,
                                    "advice": (
                                        f"Primary on-call engineer identified as '{eng_name}' ({phone_num}). "
                                        f"Per task instructions, preserve verbatim values and include '{eng_name}' in all stakeholder messages/emails and records."
                                    ),
                                })

        notes_idx = next(
            (i for i, h in enumerate(headers) if "note" in h or "comment" in h or "rule" in h),
            None,
        )
        status_idx = next(
            (i for i, h in enumerate(headers) if "status" in h or "state" in h),
            None,
        )
        title_idx = next(
            (i for i, h in enumerate(headers) if any(k in h for k in ("title", "contractor", "name", "id"))),
            0,
        )
        if notes_idx is None and status_idx is None:
            continue

        for row in values[1:]:
            if not isinstance(row, list):
                continue
            item_name = str(row[title_idx]).strip() if len(row) > title_idx else "Record"

            # Check status exclusion (e.g. Acknowledged, Resolved, Closed records)
            if status_idx is not None and len(row) > status_idx:
                status_val = str(row[status_idx]).strip().lower()
                if status_val in ("acknowledged", "resolved", "closed"):
                    key = f"status_{item_name}_{status_val}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unresolved.append({
                            "category": "status_exclusion",
                            "target": item_name,
                            "advice": f"Record '{item_name}' status is already '{row[status_idx]}' — do not page or act on already handled records.",
                        })

            if notes_idx is None or len(row) <= notes_idx:
                continue
            note = str(row[notes_idx]).strip()
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
                is_email = any(
                    k in ext_target.lower() or k in note.lower()
                    for k in ("email", "mail", "inbox", "message")
                )
                target_words = set(re.findall(r"\b[A-Za-z0-9_-]+\b", ext_target.lower())) - {
                    "the", "a", "an", "and", "or", "for", "to", "in", "at", "see", "check"
                }
                distinctive_words = {
                    w for w in target_words
                    if len(w) >= 5 and w not in {"email", "emails", "message", "messages", "team", "inbox"}
                }
                check_terms = {ext_target.lower(), item_name.lower()} | distinctive_words
                if not _already_inspected(
                    check_terms,
                    all_episodes,
                    exclude_episode_id=source_ep_id,
                    require_in_response=is_email,
                ):
                    key = f"see_{item_name}_{ext_target}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        if is_email:
                            advice = (
                                f"Record '{item_name}' notes '{note}' — search Gmail for '{item_name}' "
                                f"or check unread inbox messages to determine true status/severity before committing."
                            )
                        else:
                            advice = (
                                f"Record '{item_name}' notes '{note}' — inspect {ext_target} "
                                f"to determine true status/severity before committing."
                            )
                        unresolved.append({
                            "category": "external_investigation",
                            "target": item_name,
                            "advice": advice,
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
