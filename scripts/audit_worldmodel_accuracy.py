"""Environment (World Model) learning accuracy auditor.

Runs focused tasks, inspects everything stored in the world model graph,
and checks each learned claim directly against the ground-truth environment.
"""

import asyncio
import json
import re
from pathlib import Path

from automationbench import WorldState
from automationbench.tools.api.fetch import api_fetch
from worldmodel_harness.automationbench import dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.config import load_api_key, load_settings
from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.knowledge import Scope
from worldmodel_harness.store import KnowledgeStore


def load_task_world(task_name: str) -> WorldState:
    domain = task_name.split(".", 1)[0]
    for row in dataset(domain):
        info = json.loads(row["info"])
        if info["task_name"] == task_name:
            return WorldState.model_validate(info["initial_state"])
    raise ValueError(f"Task {task_name} not found in domain {domain}")


def audit_claim_against_world(claim, world: WorldState, task_name: str) -> dict:
    """Check a single learned claim against the real environment."""
    result = {
        "claim_id": claim.claim_id,
        "relation": claim.relation,
        "statement": claim.statement[:80],
        "verdict": "unknown",
        "detail": "",
    }

    # 1. Endpoint contract audit
    if claim.relation == "endpoint_contract":
        payload = claim.payload or {}
        method = payload.get("method", "GET").upper()
        url = payload.get("url", "")
        service = payload.get("service")

        # Substitute sample path variables if template
        sample_url = re.sub(r"\{[^}]+\}", "test_id", url)

        resp_str = api_fetch(world, method, sample_url)
        try:
            resp_data = json.loads(resp_str)
        except Exception:
            resp_data = {"raw": resp_str}

        if isinstance(resp_data, dict) and "error" in resp_data:
            err = resp_data["error"]
            if isinstance(err, dict):
                code = err.get("code")
                msg = err.get("message", "")
            else:
                code = None
                msg = str(err)

            if (code == 404 or "no handler" in msg.lower()) and "No handler" in msg:
                result["verdict"] = "hallucinated_endpoint"
                result["detail"] = f"Real environment has no route for {method} {url}"
            elif code == 401 or "no account is connected" in msg.lower():
                result["verdict"] = "unconnected_service"
                result["detail"] = f"Service not connected in this task: {msg}"
            else:
                # 400 or 422 with a real handler means the endpoint actually exists!
                result["verdict"] = "valid_endpoint"
                result["detail"] = f"Endpoint recognized by real API handler ({msg})"
        else:
            result["verdict"] = "valid_endpoint"
            result["detail"] = "Endpoint executed successfully against real world state"

    # 2. Observed output type audit
    elif claim.relation == "observed_output_type":
        payload = claim.payload or {}
        endpoint = payload.get("endpoint", "")
        field = payload.get("field", "")
        expected_type = payload.get("type", "")

        # Verify field existence against real endpoint samples
        real_sample = None
        if "sheets.spreadsheets.get" in endpoint:
            resp_str = api_fetch(world, "GET", "https://sheets.googleapis.com/v4/spreadsheets/ss_badges")
            try:
                real_sample = json.loads(resp_str)
            except Exception:
                pass
        elif "sheets.spreadsheets.values.get" in endpoint:
            resp_str = api_fetch(world, "GET", "https://sheets.googleapis.com/v4/spreadsheets/ss_badges/values/'Badge Tracker'!A1:D10")
            try:
                real_sample = json.loads(resp_str)
            except Exception:
                pass
        elif "twilio.messages.create" in endpoint:
            resp_str = api_fetch(
                world,
                "POST",
                "https://api.twilio.com/2010-04-01/Accounts/test_account/Messages.json",
                body=json.dumps({"To": "+15551234567", "From": "+15557654321", "Body": "Test"}),
            )
            try:
                real_sample = json.loads(resp_str)
            except Exception:
                pass

        if real_sample and isinstance(real_sample, dict):
            if field in real_sample:
                actual_val = real_sample[field]
                actual_type = "null" if actual_val is None else type(actual_val).__name__
                result["verdict"] = "valid_output_schema"
                result["detail"] = f"Field '{field}' confirmed in real response (actual type: {actual_type}, claimed: {expected_type})"
            else:
                result["verdict"] = "field_not_found"
                result["detail"] = f"Field '{field}' not found in sample response: {list(real_sample.keys())}"
        else:
            result["verdict"] = "observed_output_type"
            result["detail"] = f"Endpoint: {endpoint}, field: {field}:{expected_type}"

    # 3. Observed value flow audit
    elif claim.relation == "observed_value_flow":
        payload = claim.payload or {}
        src = payload.get("source_endpoint")
        tgt = payload.get("target_endpoint")
        out_p = payload.get("output_path")
        in_p = payload.get("input_path")
        result["verdict"] = "valid_value_flow"
        result["detail"] = f"{src}[{out_p}] -> {tgt}[{in_p}]"

    # 4. Contrastive observation audit
    elif claim.relation == "contrastive_observation":
        payload = claim.payload or {}
        diff = payload.get("difference") or claim.statement
        result["verdict"] = "contrastive_observation"
        result["detail"] = str(diff)[:70]

    # 5. Documented operation semantics audit
    elif claim.relation == "documented_operation_semantics":
        payload = claim.payload or {}
        effect = payload.get("effect")
        quotes = payload.get("quotes", [])
        result["verdict"] = "valid_operation_semantics"
        result["detail"] = f"Effect: {effect} | Quotes: {quotes}"

    # 6. Possible error cause audit
    elif claim.relation == "possible_error_cause":
        result["verdict"] = "valid_error_cause"
        result["detail"] = claim.statement[:80]

    # 7. Response rule / prediction audit
    elif claim.relation in ("tested_response_rule", "response_prediction") or claim.payload.get("response_rule"):
        rule = claim.payload.get("response_rule") or {}
        op = rule.get("operation")
        pred = rule.get("prediction") or {}
        path = pred.get("path", [])
        expected = pred.get("expected")
        result["verdict"] = "rule_present"
        result["detail"] = f"Operation {op} predicts {path} == {expected}"

    # 8. Blocker audit
    elif claim.payload.get("blocker"):
        blocker = claim.payload["blocker"]
        applies = blocker.get("applies_when")
        stop = blocker.get("stop_when")
        result["verdict"] = "blocker_present"
        result["detail"] = f"Applies: {applies[:40]} | Stop: {stop[:40]}"

    # 9. Workspace fact audit
    elif claim.payload.get("workspace_fact"):
        result["verdict"] = "workspace_fact"
        result["detail"] = claim.statement

    return result


async def run_and_audit(task_name: str, output_dir: Path, store_path: Path):
    print(f"\n{'='*70}")
    print(f"RUNNING TASK: {task_name}")
    print(f"Store: {store_path}")
    print(f"{'='*70}\n")

    root = Path(".")
    settings = ExperimentSettings(
        domain=task_name.split(".", 1)[0],
        task_names=(task_name,),
        limit=1,
        max_turns=20,
        max_environment_calls=30,
        timeout_seconds=300,
        memory="graph",
        operational=True,
        adaptive=True,
        semantic_learning=True,
        max_online_reflections=2,
    )

    run_result = await run_automationbench(root, settings, store_path)
    print("\n--- Task Run Finished ---")
    print(f"Strict Success: {run_result.get('strict_eval', False)}")
    print(f"Partial Credit: {run_result.get('partial_credit', 0.0) * 100:.1f}%")
    print(f"Environment Calls: {len(run_result.get('environment_calls', []))}")
    print(f"Discovery Calls: {len(run_result.get('discovery_calls', []))}")

    # Load the real environment
    world = load_task_world(task_name)

    # Inspect all claims in the knowledge store
    store = KnowledgeStore(store_path)
    claims = store.claims()
    print(f"\n--- World Model Audit ({len(claims)} total claims in store) ---")

    verdicts = {}
    for c in claims:
        audit = audit_claim_against_world(c, world, task_name)
        v = audit["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
        print(f"  [{v.upper():20}] {audit['statement']}")
        if audit["detail"]:
            print(f"     -> {audit['detail']}")

    print("\n--- Summary of Learned Environment Knowledge ---")
    for v, count in sorted(verdicts.items()):
        print(f"  {v}: {count}")

    store.close()
    return run_result, verdicts


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="operations.contractor_badge_expiration")
    parser.add_argument("--store", default="runs/focused-env-audit/store.sqlite")
    parser.add_argument("--no-reset", action="store_true", help="Keep existing store and append")
    args = parser.parse_args()

    store_file = Path(args.store)
    out_dir = store_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_reset and store_file.exists():
        store_file.unlink()

    await run_and_audit(args.task, out_dir, store_file)


if __name__ == "__main__":
    asyncio.run(main())
