"""Surface variants of AutomationBench tasks: same workflow, different literals.

A variant renames, consistently across the prompt, the initial state and the grader
assertions, every surface identifier of a task:
- ids (values with a digit or an underscore, such as ss_badges or BDG-3001);
- email addresses (local part and domain) and phone numbers;
- person names, from first/last-name pairs and person fields.
Titles, companies, statuses, amounts and dates are kept, so the rules the task
depends on are unchanged and the assertions still describe a correct solution.
A variant is the "same query, surface changed" case of world-model-v3.md.
"""

import copy
import json
import random
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"\+\d{11,14}")
ID = re.compile(r"(?=[\w-]*[\d_])[A-Za-z][\w-]*[\w]|\d{5,}")
PERSON = re.compile(r"[A-Z][a-z]+(?: [A-Z][a-z'-]+){1,2}")
FIRST_KEYS = ("first_name", "firstname", "firstName", "FirstName", "given_name")
LAST_KEYS = ("last_name", "lastname", "lastName", "LastName", "family_name")
PERSON_KEYS = {"employee", "manager", "contact_name", "user_name", "owner_name", "real_name", "assignee_name",
               "full_name", "requester_name", "rep", "sales_rep", "owner", "approver", "reviewer"}
ROLE = re.compile(
    r"contractor|candidate|technician|employee|manager|owner|assignee|contact|attendee|applicant|referr|"
    r"member|driver|tenant|guest|volunteer|instructor|student|patient|rep\b|recruiter|interviewer|approver",
    re.IGNORECASE,
)
# Columns whose two-word values are titles, organisations or categories, not people.
NOT_PERSON = re.compile(
    r"title|role|position|department|company|account|industry|stage|vendor|product|category|type|"
    r"location|building|team|status|subject|rule|plan|tier|channel|source|segment",
    re.IGNORECASE,
)
ID_KEYS = re.compile(r"(^|_)id$|Id$|^key$")
# Codes such as BDG-3001 or INC-4921, wherever they appear.
CODE = re.compile(r"[A-Z]{2,6}-\d{2,}")
FIRST_NAMES = ["Avery", "Bianca", "Caleb", "Dalia", "Emeka", "Farah", "Gideon", "Hana", "Ilya", "Jun", "Keira",
               "Lorenzo", "Mirela", "Nikhil", "Oona", "Pavel", "Quinn", "Rosalind", "Soren", "Talia", "Umar",
               "Vesna", "Wren", "Ximena", "Yusuf", "Zora"]
LAST_NAMES = ["Abara", "Bellweather", "Castellano", "Dubois", "Eriksen", "Fairbanks", "Gallo", "Halvorsen",
              "Ivanova", "Jaramillo", "Kowalczyk", "Lindqvist", "Marchetti", "Nakashima", "Okonkwo", "Petrakis",
              "Quintero", "Rasmussen", "Sorensen", "Thibault", "Underhill", "Valdivia", "Whitlock", "Yamada"]
# Values that look like ids but are part of the API or the platform, not the task's data.
KEEP = re.compile(r"^(v\d+(\.\d+)?|[a-z]+_[a-z]+_[a-z_]+)$")


def _walk(node, key=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, key)
    elif isinstance(node, str):
        yield key, node


def _objects(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def surface_literals(info: dict, first_names: frozenset[str] = frozenset()) -> dict[str, set[str]]:
    """The task's surface identifiers, by kind, found in its initial state.

    `first_names` (see `known_first_names`) lets two-word values such as a sheet's
    "Wayne Ellis" be read as person names in any field.
    """
    state = info["initial_state"]
    found = {"email": set(), "phone": set(), "id": set(), "person": set()}
    for key, value in _walk(state):
        found["email"] |= set(EMAIL.findall(value))
        found["phone"] |= set(PHONE.findall(value))
        if ID_KEYS.search(key) and ID.fullmatch(value) and not KEEP.match(value):
            found["id"].add(value)
        found["id"] |= set(CODE.findall(value))
        if key.lower() in PERSON_KEYS and PERSON.fullmatch(value):
            found["person"].add(value)
    # A column of names: named for a person role, or mostly two-word names of which at
    # least two start with a known first name.
    columns: dict[str, list[str]] = {}
    for key, value in _walk(state):
        columns.setdefault(key, []).append(value)
    for key, values in columns.items():
        if NOT_PERSON.search(key):
            continue
        names = [v for v in values if PERSON.fullmatch(v)]
        known = len({v for v in names if v.split(" ", 1)[0] in first_names}) >= 2
        if names and (ROLE.search(key) or (len(names) >= 0.6 * len(values) and known)):
            found["person"] |= set(names)
    for obj in _objects(state):
        first = next((obj[k] for k in FIRST_KEYS if isinstance(obj.get(k), str)), None)
        last = next((obj[k] for k in LAST_KEYS if isinstance(obj.get(k), str)), None)
        if first and last and PERSON.fullmatch(f"{first} {last}"):
            found["person"].add(f"{first} {last}")
    return found


def known_first_names(rows) -> frozenset[str]:
    """First names that appear in first-name fields anywhere in the dataset."""
    names = set()
    for row in rows:
        for obj in _objects(json.loads(row["info"])["initial_state"]):
            names |= {obj[k] for k in FIRST_KEYS if isinstance(obj.get(k), str) and obj[k].isalpha()}
    return frozenset(names)


def _new_id(value: str, rng: random.Random) -> str:
    def digits(match):
        return "".join(rng.choice("0123456789") for _ in match.group())

    renamed = re.sub(r"\d+", digits, value)
    if renamed == value:  # no digits: keep the prefix, replace the rest
        head, _, _ = value.partition("_")
        renamed = f"{head}_{''.join(rng.choice('abcdefghjkmnpqrstuvwxyz') for _ in range(6))}"
    return renamed


def rename_map(info: dict, seed: int, first_names: frozenset[str] = frozenset()) -> dict[str, str]:
    """Old literal -> new literal, deterministic for a seed, with no collisions."""
    rng, found, mapping = random.Random(seed), surface_literals(info, first_names), {}
    taken = set().union(*found.values())  # a new value must not equal any old one

    def put(old, new):
        while new in taken or new == old:
            new = new + rng.choice("xyz")
        mapping[old] = new
        taken.add(new)

    text = json.dumps(info)
    firsts = [n for n in FIRST_NAMES if n not in text]
    lasts = [n for n in LAST_NAMES if n not in text]
    rng.shuffle(firsts)
    rng.shuffle(lasts)
    for i, person in enumerate(sorted(found["person"])):
        first, *rest = person.split(" ")
        # Distinct first names while the pool lasts, so people stay distinguishable.
        new_first = mapping.get(first) or firsts[i % len(firsts)]
        new_last = mapping.get(rest[-1]) if rest else None
        new_last = new_last or lasts[i % len(lasts)]
        put(person, f"{new_first} {new_last}")
        # Tasks also refer to people by first or last name alone.
        if first not in mapping and len(first) >= 3:
            mapping[first] = new_first
        if rest and rest[-1] not in mapping and len(rest[-1]) >= 3:
            mapping[rest[-1]] = new_last
    domains = {}
    for email in sorted(found["email"]):
        local, domain = email.split("@", 1)
        if domain not in domains:
            domains[domain] = f"{''.join(rng.choice('abcdefghkmnprstvwz') for _ in range(7))}.example"
        new_local = ".".join(mapping.get(part.capitalize(), part).lower() for part in local.split("."))
        if new_local == local:
            new_local = f"{local[0]}{''.join(rng.choice('abcdefghkmnprstvwz') for _ in range(5))}"
        put(email, f"{new_local}@{domains[domain]}")
    for phone in sorted(found["phone"]):
        put(phone, phone[:5] + "".join(rng.choice("0123456789") for _ in phone[5:]))
    for value in sorted(found["id"]):
        put(value, _new_id(value, rng))
    return mapping


def apply(text: str, mapping: dict[str, str]) -> str:
    """Replace whole literals only (not inside longer words or ids), longest first."""
    if not mapping:
        return text
    pattern = re.compile(
        r"(?<![\w.+-])(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?![\w@-])"
    )
    return pattern.sub(lambda m: mapping[m.group(1)], text)


def make_variant(row: dict, seed: int = 0, first_names: frozenset[str] = frozenset()) -> tuple[dict, dict[str, str]]:
    """A renamed copy of a dataset row, and the rename map that produced it."""
    info = json.loads(row["info"])
    mapping = rename_map(info, seed, first_names)
    variant = copy.deepcopy(row)
    tools = info.pop("zapier_tools", None)
    name = info.pop("task_name")
    renamed = json.loads(apply(json.dumps(info), mapping))
    renamed.update({"zapier_tools": tools, "task_name": f"{name}~v{seed}"})
    variant["info"] = json.dumps(renamed)
    variant["prompt"] = [
        {**m, "content": apply(m["content"], mapping) if m.get("role") == "user" else m["content"]}
        for m in row["prompt"]
    ]
    return variant, mapping


def leaks(variant: dict, mapping: dict[str, str]) -> list[str]:
    """Old literals still present in the variant's user prompt, state or assertions."""
    users = [m for m in variant["prompt"] if m.get("role") == "user"]
    text = json.dumps(json.loads(variant["info"])) + json.dumps(users)
    return [old for old in mapping if re.search(r"(?<![\w.+-])" + re.escape(old) + r"(?![\w@-])", text)]
