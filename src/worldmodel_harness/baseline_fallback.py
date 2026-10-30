"""Hand the episode back to the plain agent when the operational layers trap it.

The world model and its guards are advice. When the decisions they shape stop
making progress (the same action keeps recurring, or the gate keeps blocking),
the rest of the episode runs exactly like the baseline: raw tools, plain finish,
no injected memory. Observations still reach the learner through the adapter.
"""

import json
from collections import Counter

TRAP_WINDOW = 6
TRAP_REPEATS = 3
TRAP_BLOCKED_STREAK = 3

NOTICE = (
    "\n[Operational layer disengaged: {reason}. Continue the task on your own judgment. "
    "From now on call the tools directly (api_search, api_fetch, base64_encode, finish); "
    "execute_decision, lookup_knowledge and check_effect are no longer available.]"
)


def _action_key(row: dict) -> str:
    action = (row.get("decision") or {}).get("action") or {}
    return json.dumps([action.get("tool"), action.get("arguments")], sort_keys=True, default=str)


def detect_trap(audit: list[dict]) -> dict | None:
    """Return the trap signal over the recent decision audit, or None."""
    streak = 0
    for row in reversed(audit):
        if (row.get("gate") or {}).get("allowed", True):
            break
        streak += 1
    if streak >= TRAP_BLOCKED_STREAK:
        return {"reason": "blocked_streak", "count": streak}
    recent = audit[-TRAP_WINDOW:]
    if not recent:
        return None
    key, count = Counter(_action_key(row) for row in recent).most_common(1)[0]
    if count >= TRAP_REPEATS:
        tool, arguments = json.loads(key)
        return {"reason": "repeated_action", "count": count, "tool": tool, "arguments": arguments}
    return None


class BaselineFallback:
    def __init__(self, enabled: bool, baseline_tools=(), finish_tool=None):
        self.enabled = enabled
        self.baseline_tools = list(baseline_tools)
        self.finish_tool = finish_tool
        self.active = False
        self.trigger = None
        self._installed = False

    def observe(self, audit: list[dict]) -> str:
        """Check the audit after a decision; return a notice when this call engages fallback."""
        if not self.enabled or self.active:
            return ""
        trap = detect_trap(audit)
        if trap is None:
            return ""
        self.active = True
        self.trigger = {**trap, "decision_index": len(audit)}
        if trap["reason"] == "blocked_streak":
            reason = f"{trap['count']} consecutive actions were blocked"
        else:
            reason = f"the same {trap['tool']} action was proposed {trap['count']} times"
        return NOTICE.format(reason=reason)

    def install(self, active_tools: dict) -> None:
        """Swap the agent's live tool map, in place, for the baseline tool set."""
        if self._installed:
            return
        active_tools.clear()
        active_tools.update({tool.name: tool for tool in self.baseline_tools})
        if self.finish_tool is not None:
            active_tools[self.finish_tool.name] = self.finish_tool
        self._installed = True

    def report(self) -> dict:
        return {"enabled": self.enabled, "engaged": self.active, "trigger": self.trigger}
