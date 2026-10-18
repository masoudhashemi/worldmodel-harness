"""Conservative behavioral induction from permitted execution observations.

Store structural patterns, not task-specific values, in reusable claims. These
are qualified empirical regularities, never assertions of causal effects.
"""

import json
import re

from .knowledge import Claim, Dependency, Episode, Layer, Prediction, Scope, utc_now
from .learning import Learner, claim_identity
from .store import KnowledgeStore

SEMANTIC_KINDS = frozenset({"phone", "email", "url", "identifier", "date", "name"})
PHONE_FIELDS = frozenset({"phone", "mobile", "tel", "telephone", "fax"})
NAME_FIELDS = frozenset({"name", "firstname", "lastname", "fullname", "displayname"})
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_EMAIL_EXACT = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_NAME_SPAN = re.compile(
    r"\b[A-Z][a-z]+(?:['’][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:['’][A-Z][a-z]+)?){1,2}\b"
)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?$")
_URL = re.compile(r"^https?://\S+$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def value_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def value_kind(value, field_name: str | None = None) -> str:
    """Classify a value by its structural type or format, independent of its literal instance."""
    kind = value_type(value)
    if kind != "string":
        return kind
    text = value.strip()
    if _EMAIL_EXACT.fullmatch(text):
        return "email"
    if _DATE.fullmatch(text):
        return "date"
    if _URL.fullmatch(text) and "{" not in text:
        return "url"
    digits = re.sub(r"\D", "", text)
    phone_shaped = (
        not re.search(r"[A-Za-z]", text)
        and 7 <= len(digits) <= 15
        and (text.startswith("+") or re.search(r"[\s().-]", text) is not None)
    )
    if phone_shaped:
        return "phone"
    if _UUID.fullmatch(text) or (
        len(text) in (15, 18) and text.isalnum() and re.search(r"[A-Za-z]", text)
    ):
        return "identifier"
    # Array indexes are path segments, not field names, and carry no slot evidence.
    name = field_name.rsplit(".", 1)[-1].casefold() if isinstance(field_name, str) else ""
    if (
        name in PHONE_FIELDS
        and not re.search(r"[A-Za-z]", text)
        and 7 <= len(digits) <= 15
        and re.fullmatch(r"[+\d\s().-]+", text)
    ):
        return "phone"
    if _person_name(text, name):
        return "name"
    return "string"


def _person_name(text: str, field_name: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    if not 1 <= len(tokens) <= 3 or re.search(r"\d", text):
        return False
    if len(tokens) >= 2 and _NAME_SPAN.fullmatch(text):
        return True
    return field_name in NAME_FIELDS and all(token[:1].isalpha() for token in tokens)


def value_format(value, field_name: str | None = None) -> str | None:
    """Normalized slot format. The literal value is not part of the format."""
    kind = value_kind(value, field_name)
    if kind in SEMANTIC_KINDS:
        return kind.upper()
    return None


def text_formats(text: str) -> set[str]:
    """Same formats as value_format, read from unstructured text rather than a scalar field."""
    found = set()
    if not isinstance(text, str) or not text.strip():
        return found
    if _EMAIL.search(text):
        found.add("EMAIL")
    if _NAME_SPAN.search(text):
        found.add("NAME")
    if re.search(r"\d{4}-\d{2}-\d{2}", text):
        found.add("DATE")
    if re.search(r"https?://\S+", text):
        found.add("URL")
    for chunk in re.findall(r"\+?\d[\d\s().-]{6,}\d", text):
        if value_kind(chunk) == "phone":
            found.add("PHONE")
    if _UUID.search(text) or re.search(
        r"\b(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{15}(?:[A-Za-z0-9]{3})?\b", text
    ):
        found.add("IDENTIFIER")
    return found


_CHANGED_SLOT_NOUNS = (
    ("PHONE", re.compile(r"\b(?:phone|mobile|telephone|fax)\b", re.IGNORECASE)),
    ("EMAIL", re.compile(r"\be-?mail\s+address(?:es)?\b", re.IGNORECASE)),
    ("URL", re.compile(r"\b(?:url|website|webpage|homepage)\b", re.IGNORECASE)),
    ("DATE", re.compile(r"\b(?:date|deadline|birthdate|birthday|due\s+date)\b", re.IGNORECASE)),
)


def requested_change_formats(text: str) -> set[str]:
    """Slot formats a public task names as the target attribute type.

    Conservative lexical detection of typed attribute nouns (as distinct from
    the communication channel carrying the request, such as a bare 'email').
    """
    if not isinstance(text, str):
        return set()
    return {label for label, pattern in _CHANGED_SLOT_NOUNS if pattern.search(text)}


def name_tokens(value) -> int:
    if not isinstance(value, str):
        return 0
    return len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", value))


def request_shape(arguments: dict) -> tuple[str, ...]:
    """Keep names and value kinds only; never promote record IDs, email text, or values."""
    fields = []
    for section in ("params", "body"):
        raw = arguments.get(section)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                fields.append(section + ":string")
                continue
        if isinstance(raw, dict):
            fields.extend(f"{section}.{key}:{value_kind(value, key)}" for key, value in raw.items())
    return tuple(sorted(fields))


def route_key(payload: dict) -> tuple[str, str]:
    return (str(payload.get("method", "")).upper(), str(payload.get("url", "")))


def contract_signature(payload: dict) -> str:
    """Aliases are not schema changes; all other documented differences matter."""
    return json.dumps({k: v for k, v in payload.items() if k != "id"}, sort_keys=True)


def request_target(url: str) -> str:
    return url.split("?", 1)[0].split("#", 1)[0]


def matching_contracts(store: KnowledgeStore, scope: Scope, method: str, url: str) -> list[Claim]:
    target = request_target(url)
    matches = []
    for claim in store.claims(scope):
        if (
            claim.relation != "endpoint_contract"
            or claim.payload.get("method", "").upper() != method.upper()
        ):
            continue
        parts = re.split(r"(\{[^{}]+\})", claim.payload.get("url", ""))
        pattern = "".join("[^/?]+" if part.startswith("{") else re.escape(part) for part in parts)
        if re.fullmatch(pattern, target):
            matches.append(claim)
    return matches


def _fresh(claim: Claim) -> bool:
    return (
        claim.status == "supported"
        and not claim.needs_review
        and (claim.valid_until is None or claim.valid_until > utc_now())
    )


def preferred_contract(matches: list[Claim]) -> Claim | None:
    """Same method and URL template are one route. Distinct templates stay unresolved
    unless all are parameterized templates and one has strictly fewer placeholders.
    """
    groups: dict[tuple[str, str], list[Claim]] = {}
    for claim in matches:
        groups.setdefault(route_key(claim.payload), []).append(claim)
    if len(groups) > 1:
        urls = [url for _, url in groups]
        if all("{" in url for url in urls):
            ranked = sorted(
                groups.items(),
                key=lambda item: (len(re.findall(r"\{[^{}]+\}", item[0][1])), -len(item[0][1])),
            )
            best_count = len(re.findall(r"\{[^{}]+\}", ranked[0][0][1]))
            second_count = len(re.findall(r"\{[^{}]+\}", ranked[1][0][1]))
            if best_count < second_count:
                groups = {ranked[0][0]: ranked[0][1]}
    if len(groups) != 1:
        return None
    group = next(iter(groups.values()))
    if len({contract_signature(claim.payload) for claim in group}) != 1:
        return None
    fresh = [claim for claim in group if _fresh(claim)]
    pool = fresh or group
    return max(pool, key=lambda claim: (claim.version, claim.claim_id))


def endpoint_contract(store: KnowledgeStore, scope: Scope, arguments: dict) -> Claim | None:
    chosen = preferred_contract(
        matching_contracts(
            store, scope, arguments.get("method", ""), arguments.get("url", "")
        )
    )
    if chosen is None or not _fresh(chosen):
        return None
    return chosen


def predict(store: KnowledgeStore, scope: Scope, arguments: dict) -> tuple[Prediction, ...]:
    contract = endpoint_contract(store, scope, arguments)
    if contract is None:
        return ()
    shape = request_shape(arguments)
    predictions = []
    for claim in store.claims(scope):
        if (
            claim.relation != "observed_output_type"
            or claim.status != "supported"
            or claim.needs_review
            or (claim.valid_until is not None and claim.valid_until <= utc_now())
            or claim.payload["endpoint"] != contract.payload["id"]
            or tuple(claim.payload["request_shape"]) != shape
        ):
            continue
        if any(
            d.mode == "requires" and not store.dependency_compatible(d) for d in claim.dependencies
        ):
            continue
        predictions.append(
            Prediction(
                claim_id=claim.claim_id,
                claim_version=claim.version,
                path=claim.payload["field"],
                expected=claim.payload["type"],
                predicate="type",
            )
        )
    return tuple(predictions)


class BehaviorLearner:
    def __init__(self, learner: Learner):
        self.learner = learner
        self.store = learner.store

    def _retain(
        self,
        episode: Episode,
        contract: Claim,
        relation: str,
        payload: dict,
        evidence_ids: tuple[str, ...],
        *,
        promotable: bool,
    ) -> Claim:
        scope = Scope(
            environment=episode.scope.environment,
            version=episode.scope.version,
            role=episode.scope.role,
        )
        stable = (
            {key: value for key, value in payload.items() if key not in {"type", "conflicting_type"}}
            if relation == "observed_output_type"
            else payload
        )
        signature = json.dumps(stable, sort_keys=True)
        participants = (contract.payload["id"], relation, signature)
        identity = claim_identity(scope, relation, participants)
        try:
            previous = self.store.claim(identity)
        except KeyError:
            previous = None
        ids = tuple(
            dict.fromkeys((*(previous.supporting_episodes if previous else ()), *evidence_ids))
        )
        if previous and ids == previous.supporting_episodes:
            return previous
        # Multiple logs of one event or repetitions of one task/tenant cannot
        # promote an environment-wide observed regularity.
        evidence = [self.store.episode(i) for i in ids]
        independent = {(e.scope.tenant, e.task_id) for e in evidence}
        tenants = {e.scope.tenant for e in evidence if e.scope.tenant is not None}
        supported = promotable and len(independent) >= 2 and len(tenants) >= 2
        status = "supported" if supported else "hypothesis"
        contradicting = previous.contradicting_episodes if previous else ()
        if (
            previous
            and relation == "observed_output_type"
            and previous.payload.get("type") != payload.get("type")
        ):
            # Another kind in the same slot disputes that slot. The literal value is not stored.
            contradicting = tuple(dict.fromkeys((*contradicting, *evidence_ids)))
            ids = previous.supporting_episodes
            payload = {**previous.payload, "conflicting_type": payload["type"]}
            status = "disputed"
        elif previous and previous.status in ("disputed", "retired"):
            status = previous.status
        claim = Claim(
            claim_id=identity,
            version=previous.version + 1 if previous else 1,
            layer=Layer.BEHAVIOR,
            relation=relation,
            participants=participants,
            statement=f"Observed {relation} for {contract.payload['id']}; request shape is a condition, not proof of causation.",
            scope=scope,
            payload=payload,
            origin="inferred",
            status=status,
            supporting_episodes=ids,
            contradicting_episodes=contradicting,
            dependencies=(Dependency(claim_id=contract.claim_id, version=contract.version),),
            needs_review=status == "disputed" or (previous.needs_review if previous else False),
            last_verified_at=episode.timestamp,
        )
        self.store.put_claim(claim)
        return claim

    def observe(self, episode: Episode) -> list[Claim]:
        if episode.operation != "api_fetch" or episode.outcome in ("unknown", "blocked"):
            return []
        contract = endpoint_contract(self.store, episode.scope, episode.arguments)
        if contract is None:
            return []
        common = {
            "endpoint": contract.payload["id"],
            "request_shape": request_shape(episode.arguments),
        }
        claims = []
        if episode.outcome == "success" and isinstance(episode.response, dict):
            for field in episode.coverage:
                if field not in episode.response:
                    continue
                claims.append(
                    self._retain(
                        episode,
                        contract,
                        "observed_output_type",
                        common
                        | {
                            "field": field,
                            "type": value_kind(episode.response[field], field),
                        },
                        (episode.episode_id,),
                        promotable=True,
                    )
                )
        if episode.outcome == "error":
            error = episode.response.get("error", {}) if isinstance(episode.response, dict) else {}
            code = error.get("code") if isinstance(error, dict) else None
            # Error messages may contain live identifiers; retain only the code.
            if isinstance(code, (int, str)) and str(code).isdigit():
                claims.append(
                    self._retain(
                        episode,
                        contract,
                        "error_observed",
                        common | {"error_code": str(code)},
                        (episode.episode_id,),
                        promotable=False,
                    )
                )
        if episode.outcome == "success":
            previous = self.store.episodes(episode.scope)
            for failure in reversed(previous):
                if failure.episode_id == episode.episode_id or failure.operation != "api_fetch":
                    continue
                # Do not attribute a recovery across unrelated records or tasks.
                if failure.arguments.get("url") != episode.arguments.get(
                    "url"
                ) or failure.arguments.get("method") != episode.arguments.get("method"):
                    continue
                if failure.outcome == "error" and request_shape(failure.arguments) != request_shape(
                    episode.arguments
                ):
                    claims.append(
                        self._retain(
                            episode,
                            contract,
                            "recovery_sequence",
                            common | {"previous_request_shape": request_shape(failure.arguments)},
                            (failure.episode_id, episode.episode_id),
                            promotable=False,
                        )
                    )
                break
        return claims
