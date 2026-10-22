"""Run a small single-task test with live Gemma model and display injected context."""

import asyncio
import json
import os
import sys
from pathlib import Path

from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.knowledge import Episode
from worldmodel_harness.store import KnowledgeStore


async def run_small_test(task_name: str = "operations.asana_fire_drill", max_turns: int = 5):
    os.environ["WORLDMODEL_MODEL_CONFIG"] = "configs/model-gemma.toml"
    root = Path.cwd()
    store_path = root / "runs/iterative-eval-adapted-v2/stores/worldmodel_carried.sqlite"
    if not store_path.exists():
        store_path = root / "runs/iterative-worldmodel-eval/stores/worldmodel_carried.sqlite"

    settings = ExperimentSettings(
        domain="operations",
        task_names=(task_name,),
        limit=1,
        max_turns=max_turns,
        max_environment_calls=max_turns,
        timeout_seconds=240,
        memory="graph",
        operational=True,
        resolve_execution_frontier=True,
        successful_transition_graph=True,
        phase="test",
        semantic_learning=False,
        adaptive=True,
    )

    print(f"\n========================================================")
    print(f"LAUNCHING SMALL TEST: {task_name} (Max Turns: {max_turns})")
    print(f"Model: Gemma-4-31B-it via ElementAI")
    print(f"Carried Store: {store_path}")
    print(f"Frontier Resolution: ENABLED | Entity Tracker: ENABLED")
    print(f"========================================================\n")

    report = await run_automationbench(root, settings, store_path)
    run_id = report.get("run_id")
    res = report["results"][0] if report.get("results") else {}

    print(f"\n========================================================")
    print(f"RUN FINISHED: {run_id}")
    print(f"Strict Completed: {res.get('task_completed_correctly')}")
    print(f"Partial Credit: {res.get('partial_credit')}")
    print(f"Turns Taken: {res.get('turns_taken')}")
    print(f"Error Type: {res.get('error_type')}")
    print(f"========================================================\n")

    # Extract turn-by-turn injected information
    retrievals = res.get("retrievals", [])
    decision_audit = res.get("decision_audit", [])
    evidence_path = root / "runs" / run_id / "evidence.sqlite"
    evidence_store = KnowledgeStore(evidence_path, frozen=True) if evidence_path.exists() else None

    print(f"EXTRACTING INJECTIONS ACROSS {len(retrievals)} TURNS:\n")
    for turn_idx, ret in enumerate(retrievals):
        print(f"--------------------------------------------------------")
        print(f"TURN {turn_idx}:")
        print(f"--------------------------------------------------------")

        # 1. WHERE
        print(f"  [WHERE]: System operational context prefix at head of conversation messages.")

        # 2. HOW & WHY
        cur_node = ret.get("current_node")
        comp_nodes = ret.get("completed_nodes", [])
        focus_ents = ret.get("focus_entities", [])
        subgoal = ret.get("subgoal", "")

        print(f"  [HOW]:")
        print(f"     Resolved Current Node: {cur_node}")
        print(f"     Completed Operations: {comp_nodes}")
        print(f"     Focus Entities: {focus_ents}")
        print(f"     Subgoal: \"{subgoal}\"")

        print(f"  [WHY]:")
        if turn_idx == 0:
            print(f"     Initial cold start: planner identifies root observation endpoint from task instruction.")
        else:
            prev_action = decision_audit[turn_idx - 1]["decision"]["action"] if turn_idx - 1 < len(decision_audit) else {}
            print(f"     Prior executed action: {prev_action.get('tool')} {prev_action.get('arguments', {}).get('method', '')} {prev_action.get('arguments', {}).get('url', '')}")
            print(f"     Advance frontier updated state from evidence -> conditioned next step.")

        # 3. WHAT
        print(f"  [WHAT INJECTED]:")
        plan_steps = ret.get("workflow_plan_steps", 0)
        active_step = ret.get("workflow_plan_active_step", 0)
        plan_goal = ret.get("workflow_plan_goal")
        injected_claims = ret.get("injected_claims", [])

        if plan_steps > 0:
            print(f"     Workflow Plan: {plan_goal} (Total Steps: {plan_steps}, Active Step: {active_step})")
        else:
            print(f"     Workflow Plan: Not active")

        print(f"     Injected Claims ({len(injected_claims)}):")
        for cid in injected_claims[:5]:
            try:
                cl = evidence_store.claim(cid) if evidence_store else None
            except Exception:
                cl = None
            if cl and isinstance(cl.payload, dict):
                m = cl.payload.get("method", "")
                u = cl.payload.get("url") or cl.payload.get("path", "")
                print(f"       * {cl.payload.get('id', cid)}: {m} {u}")
            else:
                print(f"       * {cid}")
        if len(injected_claims) > 5:
            print(f"       ... and {len(injected_claims) - 5} more claims")

        # Check action taken at this turn
        if turn_idx < len(decision_audit):
            da = decision_audit[turn_idx]
            dec_act = da["decision"]["action"]
            gate = da.get("gate", {})
            print(f"\n  [MODEL ACTION TAKEN IN RESPONSE]:")
            print(f"     Goal: {da['decision'].get('goal')}")
            print(f"     Action: {dec_act.get('tool')} args: {dec_act.get('arguments')}")
            print(f"     Gate Allowed: {gate.get('allowed')}, Gaps: {gate.get('gaps')}")
            if not gate.get("allowed"):
                rep = da.get("repair_details", {})
                print(f"     Gate Block Diagnostic: reason={rep.get('reason')}, next_step={rep.get('instruction') or da.get('repair_details')}")

        print()

    if evidence_store:
        evidence_store.close()


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "operations.asana_fire_drill"
    asyncio.run(run_small_test(task, max_turns=5))
