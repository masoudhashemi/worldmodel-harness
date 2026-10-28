"""Resolve execution evidence without treating harness feedback as world actions.

Maintains entity-and-requirement obligation state machine across workflow steps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .knowledge import Episode


def extract_task_exclusions(task_text: str) -> list[str]:
    """Extract negative constraints / entity exclusions from task prompt."""
    if not task_text:
        return []
    patterns = [
        r"(?:except|excluding|exclude|skip|ignore|omit|do not process|do not notify|do not include|dont include|other than|don't bother with|dont bother with|do not bother with)\s+(?:for\s+)?(?:the\s+)?(?:contractors who've been\s+|badges that were\s+|tickets that are\s+)?(?:(?:already|previously)\s+)?([A-Za-z0-9_-]+)",
    ]
    exclusions = []
    for pat in patterns:
        for match in re.finditer(pat, task_text, re.IGNORECASE):
            term = match.group(1).strip().lower()
            if term and len(term) > 2 and term not in ("all", "any", "this", "that", "each", "every", "the"):
                exclusions.append(term)
    return list(dict.fromkeys(exclusions))


def check_exclusion_reason(
    *,
    name: str | None,
    badge: str | None,
    status: str,
    notes: str,
    task_exclusions: list[str],
    full_text: str = "",
) -> str | None:
    """Check why an entity record should be excluded based on task intent and applicable policy."""
    status_lower = (status or "").lower()
    notes_lower = (notes or "").lower()
    full_text_lower = (full_text or "").lower()

    # Exclusions must come from explicit task constraints or applicable environment policy
    for ex in task_exclusions:
        if status_lower and (ex == status_lower or ex in status_lower):
            return f"status '{status}' matches task exclusion: '{ex}'"
        if name and ex in name.lower():
            return f"matches task exclusion: '{ex}'"
        if badge and ex in badge.lower():
            return f"matches task exclusion: '{ex}'"
        if notes_lower and ex in notes_lower:
            return f"matches task exclusion: '{ex}'"
        if full_text_lower and ex in full_text_lower:
            return f"matches task exclusion: '{ex}'"

    # Raw statuses, names and notes are observations, not universal business policies.
    # Only explicit task exclusions are enforced here; policy inference belongs to
    # the evidence-backed learning layer.
    return None


def is_entity_excluded(
    *,
    name: str | None,
    badge: str | None,
    status: str,
    notes: str,
    task_exclusions: list[str],
    full_text: str = "",
) -> bool:
    """Determine whether an entity record should be filtered out from active obligations."""
    return check_exclusion_reason(
        name=name,
        badge=badge,
        status=status,
        notes=notes,
        task_exclusions=task_exclusions,
        full_text=full_text,
    ) is not None


def environment_actions(interaction: Episode | None, evidence) -> list[Episode]:
    """Return linked observations in execution order, never infer execution from a call."""
    if interaction is None:
        return []
    if interaction.origin == "environment":
        return [interaction]
    actions = []
    for episode_id in interaction.related_episode_ids:
        try:
            episode = evidence.episode(episode_id)
        except KeyError:
            continue
        if (
            episode.origin == "environment"
            and episode.task_id == interaction.task_id
            and episode.scope == interaction.scope
            and episode.interaction_id == interaction.interaction_id
        ):
            actions.append(episode)
    return actions


@dataclass
class ChannelObligation:
    channel: str  # "sms", "email", "slack", "asana", etc.
    status: str = "pending"  # "pending", "accepted", "delivered", "verified"
    evidence_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedEntity:
    entity_id: str
    name: str | None = None
    badge_id: str | None = None
    phone: str | None = None
    email: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    obligations: dict[str, ChannelObligation] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return bool(self.obligations) and all(
            ob.status in ("accepted", "delivered", "verified")
            for ob in self.obligations.values()
        )

    def next_pending_channel(self) -> str | None:
        for ch, ob in self.obligations.items():
            if ob.status == "pending":
                return ch
        return None


class EntityTracker:
    """Tracks entity obligations and requirement state across workflow steps."""

    def __init__(self, required_channels: tuple[str, ...] = ()):
        self.entities: dict[str, TrackedEntity] = {}
        self.default_channels = list(required_channels) or ["sms", "email", "slack"]
        self.task_exclusions: list[str] = []
        self.excluded_entities: dict[str, str] = {}

    def infer_required_channels(self, task_text: str) -> list[str]:
        text = task_text.lower()
        channels = []
        if "sms" in text or "twilio" in text or "phone" in text or "text" in text:
            channels.append("sms")
        if "email" in text or "gmail" in text:
            channels.append("email")
        if "slack" in text or "channel" in text:
            channels.append("slack")
        if "asana" in text or ("task" in text and "project" in text):
            channels.append("asana")
        if not channels:
            channels = ["action"]
        return channels

    def register_from_read(self, episode: Episode, task_text: str = "") -> list[str]:
        """Extract eligible entities and their required channels from a read observation."""
        if episode.outcome != "success":
            return []
        resp = episode.response
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
            except Exception:
                return []
        if not isinstance(resp, (dict, list)):
            return []

        channels = self.infer_required_channels(task_text) if task_text else self.default_channels
        if task_text:
            for ex in extract_task_exclusions(task_text):
                if ex not in self.task_exclusions:
                    self.task_exclusions.append(ex)

        # Google Sheets values (e.g. valueRanges or values)
        values = None
        if isinstance(resp, dict):
            values = resp.get("values")
            if not values and "valueRanges" in resp and isinstance(resp["valueRanges"], list):
                for vr in resp["valueRanges"]:
                    if isinstance(vr, dict) and "values" in vr:
                        values = vr["values"]
                        break

        new_entities = []
        if isinstance(values, list) and len(values) >= 2:
            headers = [str(h).strip().lower() for h in values[0]]
            name_idx = next((i for i, h in enumerate(headers) if any(k in h for k in ("name", "title", "summary", "contractor", "user", "assignee", "lead", "description"))), None)
            badge_idx = next((i for i, h in enumerate(headers) if any(k in h for k in ("badge", "id", "code", "ticket", "number"))), None)
            phone_idx = next((i for i, h in enumerate(headers) if any(k in h for k in ("phone", "mobile", "tel"))), None)
            email_idx = next((i for i, h in enumerate(headers) if any(k in h for k in ("email", "contact", "address"))), None)
            status_idx = next((i for i, h in enumerate(headers) if "status" in h), None)
            notes_idx = next((i for i, h in enumerate(headers) if "note" in h), None)

            for row in values[1:]:
                if not isinstance(row, list) or len(row) == 0:
                    continue
                name = str(row[name_idx]).strip() if name_idx is not None and name_idx < len(row) else None
                badge = str(row[badge_idx]).strip() if badge_idx is not None and badge_idx < len(row) else None
                phone = str(row[phone_idx]).strip() if phone_idx is not None and phone_idx < len(row) else None
                email = str(row[email_idx]).strip() if email_idx is not None and email_idx < len(row) else None
                status = str(row[status_idx]).strip() if status_idx is not None and status_idx < len(row) else ""
                notes = str(row[notes_idx]).strip() if notes_idx is not None and notes_idx < len(row) else ""

                reason = check_exclusion_reason(
                    name=name,
                    badge=badge,
                    status=status,
                    notes=notes,
                    task_exclusions=self.task_exclusions,
                    full_text=" ".join(str(c) for c in row),
                )
                if reason:
                    if name and len(name) > 2 and name.lower() not in ("all", "any", "the", "new", "open"):
                        self.excluded_entities[name] = reason
                    if badge and len(badge) > 2 and badge.lower() not in ("all", "any", "the", "new", "open"):
                        self.excluded_entities[badge] = reason
                    continue

                entity_id = badge or name
                if not entity_id or entity_id in self.entities:
                    continue

                obs = {ch: ChannelObligation(channel=ch, status="pending") for ch in channels}
                entity = TrackedEntity(
                    entity_id=entity_id,
                    name=name,
                    badge_id=badge,
                    phone=phone,
                    email=email,
                    metadata={"status": status, "notes": notes, "evidence_id": episode.episode_id},
                    obligations=obs,
                )
                self.entities[entity_id] = entity
                new_entities.append(entity_id)

        # Also support JSON lists of entity objects (REST API responses, e.g. Jira, Slack, etc.)
        records_list = []
        if isinstance(resp, list):
            records_list = resp
        elif isinstance(resp, dict) and not values:
            for k in ("records", "issues", "items", "data", "cards", "rows"):
                val = resp.get(k)
                if isinstance(val, list) and val:
                    records_list = val
                    break

        if records_list and not values:
            for item in records_list:
                if not isinstance(item, dict):
                    continue
                fields = item.get("fields") if isinstance(item.get("fields"), dict) else item
                name_val = (
                    fields.get("name")
                    or fields.get("summary")
                    or fields.get("title")
                    or fields.get("contractor")
                    or fields.get("user")
                    or fields.get("assignee")
                )
                if isinstance(name_val, dict):
                    name_val = name_val.get("displayName") or name_val.get("name") or name_val.get("email")
                name = str(name_val).strip() if name_val else None

                badge_val = (
                    item.get("id")
                    or item.get("key")
                    or fields.get("badge")
                    or fields.get("badge_id")
                    or fields.get("ticket")
                    or fields.get("code")
                )
                badge = str(badge_val).strip() if badge_val else None

                phone_val = fields.get("phone") or fields.get("mobile") or fields.get("tel")
                phone = str(phone_val).strip() if phone_val else None

                email_val = fields.get("email") or fields.get("contact") or fields.get("address")
                email = str(email_val).strip() if email_val else None

                status_val = fields.get("status")
                if isinstance(status_val, dict):
                    status_val = status_val.get("name")
                status = str(status_val).strip() if status_val else ""

                notes_val = fields.get("notes") or fields.get("description") or fields.get("comment")
                notes = str(notes_val).strip() if notes_val else ""

                reason = check_exclusion_reason(
                    name=name,
                    badge=badge,
                    status=status,
                    notes=notes,
                    task_exclusions=self.task_exclusions,
                    full_text=json.dumps(item),
                )
                if reason:
                    if name and len(name) > 2 and name.lower() not in ("all", "any", "the", "new", "open"):
                        self.excluded_entities[name] = reason
                    if badge and len(badge) > 2 and badge.lower() not in ("all", "any", "the", "new", "open"):
                        self.excluded_entities[badge] = reason
                    continue

                entity_id = badge or name
                if not entity_id or entity_id in self.entities:
                    continue

                obs = {ch: ChannelObligation(channel=ch, status="pending") for ch in channels}
                entity = TrackedEntity(
                    entity_id=entity_id,
                    name=name,
                    badge_id=badge,
                    phone=phone,
                    email=email,
                    metadata={"status": status, "notes": notes, "evidence_id": episode.episode_id},
                    obligations=obs,
                )
                self.entities[entity_id] = entity
                new_entities.append(entity_id)

        return new_entities

    def record_action(self, episode: Episode) -> str | None:
        """Update obligation state for the entity targeted by a mutation action."""
        if episode.outcome not in ("success", "queued"):
            return None
        args = episode.arguments or {}
        resp = episode.response or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
            except Exception:
                resp = {}

        url = str(args.get("url", "")).lower()
        op = str(episode.operation).lower()
        body_text = str(args.get("body", ""))

        channel = None
        if "twilio" in url or "twilio" in op or "sms" in op:
            channel = "sms"
        elif "gmail" in url or "gmail" in op or "email" in op:
            channel = "email"
        elif "slack" in url or "slack" in op or "chat.postmessage" in url:
            channel = "slack"
        elif "asana" in url or "asana" in op:
            channel = "asana"

        if not channel:
            return None

        body = args.get("body", {})
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except (ValueError, TypeError):
                return None
        if not isinstance(body, dict):
            return None

        if channel in ("sms", "email"):
            # Destination fields establish recipients; names mentioned in prose do not.
            recipient = body.get("To", body.get("to"))
            if channel == "email":
                from email.utils import getaddresses

                if recipient is None and isinstance(body.get("raw"), str):
                    import base64
                    from email import policy
                    from email.parser import BytesParser

                    try:
                        raw = body["raw"]
                        decoded = base64.b64decode(raw + "=" * (-len(raw) % 4),
                                                   altchars=b"-_", validate=True)
                        message = BytesParser(policy=policy.default).parsebytes(decoded)
                        recipient = message.get_all("To", [])
                    except (ValueError, TypeError):
                        return None
                headers = recipient if isinstance(recipient, list) else [recipient]
                if not all(isinstance(value, str) for value in headers):
                    return None
                destinations = {address.casefold() for _, address in getaddresses(headers)}
                matches = [entity for entity in self.entities.values()
                           if entity.email and entity.email.casefold() in destinations]
            else:
                matches = [entity for entity in self.entities.values()
                           if entity.phone and entity.phone == recipient]
        else:
            matches = [entity for entity in self.entities.values()
                       if ((entity.badge_id and entity.badge_id in body_text)
                           or (entity.name and entity.name.lower() in body_text.lower()))]

        # Shared addresses and ambiguous mentions cannot establish an entity binding.
        if len(matches) != 1:
            return None
        matched_entity = matches[0]

        if matched_entity and channel in matched_entity.obligations:
            # Preserve status semantics: queued is accepted by API, not confirmed delivered
            is_queued = (
                isinstance(resp, dict) and resp.get("status") == "queued"
            ) or episode.outcome == "queued"
            matched_entity.obligations[channel].status = "accepted"
            matched_entity.obligations[channel].evidence_id = episode.episode_id
            matched_entity.obligations[channel].details = {"queued": is_queued}
            return matched_entity.entity_id

        return None

    def active_entity(self) -> TrackedEntity | None:
        """Return the current active entity with pending obligations."""
        for entity in self.entities.values():
            if not entity.is_complete():
                return entity
        return None

    def pending_entities(self) -> list[TrackedEntity]:
        return [e for e in self.entities.values() if not e.is_complete()]

    def completed_entities(self) -> list[TrackedEntity]:
        return [e for e in self.entities.values() if e.is_complete()]

    def summary(self) -> dict[str, Any]:
        """Return a structured entity obligation summary."""
        active = self.active_entity()
        pending = self.pending_entities()
        active_summary = None
        if active:
            active_summary = {
                "entity": active.name or active.badge_id or active.entity_id,
                "badge": active.badge_id,
                "obligations": {
                    ch: {
                        "status": (
                            "request accepted (queued)"
                            if ob.details.get("queued")
                            else ob.status
                        ),
                        "evidence": ob.evidence_id,
                    }
                    for ch, ob in active.obligations.items()
                },
                "next_pending": active.next_pending_channel(),
            }
        res = {
            "active_entity": active_summary,
            "total_entities": len(self.entities),
            "completed_count": len(self.completed_entities()),
            "pending_count": len(pending),
            "pending_entities": [e.name or e.badge_id for e in pending if e != active],
        }
        if self.task_exclusions:
            res["negative_constraints"] = [
                f"Excluded from processing: {ex}. Rule: Do NOT mention or include {ex} in outgoing messages, reports, or records, even to explain why it was excluded."
                for ex in self.task_exclusions
            ]
        return res


def advance_frontier(
    store,
    scope,
    evidence,
    interaction,
    current_node,
    completed,
    *,
    entity_tracker: EntityTracker | None = None,
    task_text: str = "",
):
    """Advance only on successful environment observations; retain local failures separately."""
    from .behavior import endpoint_contract

    for episode in environment_actions(interaction, evidence):
        if episode.outcome not in ("success", "queued") or episode.operation in ("api_search", "lookup_knowledge"):
            continue
        node = episode.operation
        if node == "api_fetch":
            contract = endpoint_contract(store, scope, episode.arguments)
            if contract is not None:
                node = contract.payload.get("id") or contract.claim_id
        current_node = node
        completed.add(node)

        if entity_tracker is not None:
            method = episode.arguments.get("method", "GET").upper() if isinstance(episode.arguments, dict) else "GET"
            if episode.operation == "api_fetch" and method == "GET":
                entity_tracker.register_from_read(episode, task_text=task_text)
            else:
                entity_tracker.record_action(episode)

    return current_node
