from typing import Any

from agent import MahjongAgent
from helpers import (
    process_streaming_agent_response,
    update_task_with_agent_response,
)
from typing_extensions import override

from a2a.server.agent_execution import BaseAgentExecutor
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TextPart,   
)
from a2a.utils import create_task_obj
from langchain_mcp_adapters.client import MultiServerMCPClient

class MahjongAgentExecutor(BaseAgentExecutor):
    """Mahjong AgentExecutor Example."""

    def __init__(self, mcp_client: MultiServerMCPClient):
        self.agent = MahjongAgent(mcp_client=mcp_client)

    @override
    async def on_message_send(
        self,
        request: SendMessageRequest,
        event_queue: EventQueue,
        task: Task | None,
    ) -> None:
        """Handler for 'message/send' requests."""
        params: MessageSendParams = request.params
        query = self._get_user_query(params)

        if not task:
            task = create_task_obj(params)

        # invoke the underlying agent
        agent_response: dict[str, Any] = self.agent.invoke(
            query, task.contextId
        )
        update_task_with_agent_response(task, agent_response)
        event_queue.enqueue_event(task)

    @override
    async def on_message_stream(
        self,
        request: SendStreamingMessageRequest,
        event_queue: EventQueue,
        task: Task | None,
    ) -> None:
        """Handler for 'message/stream' requests."""
        params: MessageSendParams = request.params
        query = self._get_user_query(params)
        player_id = int(self._get_param(params, 'player_id', 999))
        persona = self._get_param(params, 'persona', '女性玩家')

        if not task:
            task = create_task_obj(params)
            # emit the initial task so it is persisted to TaskStore
            event_queue.enqueue_event(task)

        # kickoff the streaming agent and process responses
        async for item in self.agent.stream(query, task.contextId, player_id, persona):
            task_artifact_update_event, task_status_event = (
                process_streaming_agent_response(task, item)
            )

            if task_artifact_update_event:
                event_queue.enqueue_event(task_artifact_update_event)

            event_queue.enqueue_event(task_status_event)

    def _get_user_query(self, task_send_params: MessageSendParams) -> str:
        """Helper to get user query from task send params."""
        part = task_send_params.message.parts[0].root
        if not isinstance(part, TextPart):
            raise ValueError('Only text parts are supported')
        return part.text

    def _get_param(self, task_send_params: MessageSendParams, key: str, default=None):
        for part in task_send_params.message.parts:
            root = getattr(part, 'root', None)
            if root and getattr(root, 'type', None) == 'text':
                txt = getattr(root, 'text', '')
                if txt.startswith('[PARAM]') and f'{key}=' in txt:
                    try:
                        return (txt.split(f'{key}=')[1].split()[0])
                    except Exception:
                        pass
        return default
