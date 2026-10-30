"""Environment-call ledger with an optional ceiling."""

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    max_environment_calls: int | None = None
    environment_calls: int = 0
    by_purpose: dict[str, int] = field(default_factory=dict)
    compute_seconds: float = 0.0
    utility_calls: int = 0

    def charge_environment(self, purpose: str) -> None:
        if (
            self.max_environment_calls is not None
            and self.environment_calls >= self.max_environment_calls
        ):
            raise BudgetExceeded("Environment budget exhausted")
        self.environment_calls += 1
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1
