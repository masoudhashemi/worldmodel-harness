"""Record public tool interactions without treating local feedback as world state.

Environment episodes remain authoritative observations of the environment.
Harness episodes link tool schemas, rejected calls, verification feedback and
successful calls to those observations. No correction or recovery is prescribed.
"""

import json
from contextvars import ContextVar
from uuid import uuid4

from .knowledge import Episode

CURRENT_INTERACTION: ContextVar[str | None] = ContextVar("interaction", default=None)


class InteractionRecorder:
    def __init__(self, adapter):
        self.adapter = adapter
        self.previous_id = None

    def retain(self, episode):
        adapter = self.adapter
        adapter.evidence.append_episode(episode)
        persistent = adapter.learner is not None or adapter.retain_interactions
        if persistent and adapter.knowledge is not adapter.evidence:
            adapter.knowledge.append_episode(episode)
        if adapter.learner is not None:
            from .rule_tests import observe_rules

            observe_rules(adapter.knowledge, episode)

    def register_tools(self, tools):
        """Retain the exact public schemas, never evaluator state or hidden tools."""
        self.adapter.public_tools = {tool.name: tool for tool in tools}
        for tool in tools:
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_json_schema(),
            }
            episode = Episode(
                task_id=self.adapter.scope.task_id,
                scope=self.adapter.scope,
                operation="tool_schema",
                arguments={"name": tool.name},
                response=schema,
                outcome="success",
                coverage=tuple(schema),
                purpose="discovery",
                origin="harness",
                environment_called=False,
            )
            self.retain(episode)

    def begin(self):
        interaction_id = uuid4().hex
        return interaction_id, CURRENT_INTERACTION.set(interaction_id)

    def end(self, interaction_id, token, call, result=None, error_type=None):
        CURRENT_INTERACTION.reset(token)
        try:
            arguments = json.loads(call.arguments)
        except (ValueError, TypeError):
            arguments = {"raw_arguments": call.arguments}
        if not isinstance(arguments, dict):
            arguments = {"raw_arguments": arguments}
        related = tuple(
            episode.episode_id
            for episode in self.adapter.evidence.episodes(self.adapter.scope)
            if episode.interaction_id == interaction_id and episode.origin == "environment"
        )
        response = {"error_type": error_type} if result is None else {
            "content": result.content,
            "success": result.success,
            "args_was_valid": result.args_was_valid,
        }
        if result is not None:
            # Keep original text alongside its structured portion. The suffix
            # supplies controller-owned evidence references, not API fields.
            structured = result.content.split("\nTask evidence references:", 1)[0]
            try:
                response["result"] = json.loads(structured)
            except (ValueError, TypeError):
                pass
        episode = Episode(
            task_id=self.adapter.scope.task_id,
            scope=self.adapter.scope,
            operation="tool_call:" + call.name,
            arguments=arguments,
            response=response,
            outcome="unknown" if result is None else "success" if result.success else "error",
            coverage=tuple(response),
            origin="harness",
            interaction_id=interaction_id,
            previous_interaction_id=self.previous_id,
            related_episode_ids=related,
            environment_called=bool(related),
        )
        self.retain(episode)
        self.previous_id = interaction_id
        self.adapter.last_interaction = episode
        return episode
