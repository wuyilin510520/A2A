import logging

from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import (
    RunnableConfig,
)
from langchain_core.tools import tool  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent  # type: ignore
from pydantic import BaseModel


logger = logging.getLogger(__name__)

memory = MemorySaver()


@tool
def get_mahjong_suggestion(
    hand: list[str] = None,
    wall_count: int = 0,
    is_current_player: bool = False,
):
    """麻将AI出牌建议工具。

    Args:
        hand: 当前手牌（如["1M", "2M", ...]）。
        wall_count: 剩余牌墙数量。
        is_current_player: 是否当前玩家。

    Returns:
        建议打出的牌，或错误信息。
    """
    if not is_current_player:
        return {"error": "当前不是你的回合，无法出牌。"}
    if not hand:
        return {"error": "手牌为空。"}
    # 简单策略：打出第一张
    return {"suggestion": hand[0], "message": f"建议打出: {hand[0]}"}


class MahjongResponseFormat(BaseModel):
    """麻将Agent响应格式。"""
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class MahjongAgent:
    """麻将AI Agent 示例。"""

    SYSTEM_INSTRUCTION = (
        '你是一个麻将AI助手，只能用 get_mahjong_suggestion 工具为用户提供出牌建议。'
        '如果用户问与麻将无关的问题，请礼貌拒绝。'
    )

    RESPONSE_FORMAT_INSTRUCTION: str = (
        'Select status as completed if the request is complete'
        'Select status as input_required if the input is a question to the user'
        'Set response status to error if the input indicates an error'
    )

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        self.tools = [get_mahjong_suggestion]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, MahjongResponseFormat),
        )

    def invoke(self, query: str, sessionId: str) -> dict[str, Any]:
        config: RunnableConfig = {'configurable': {'thread_id': sessionId}}
        self.graph.invoke({'messages': [('user', query)]}, config)
        return self.get_agent_response(config)

    async def stream(
        self, query: str, sessionId: str
    ) -> AsyncIterable[dict[str, Any]]:
        inputs: dict[str, Any] = {'messages': [('user', query)]}
        config: RunnableConfig = {'configurable': {'thread_id': sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '正在分析出牌建议...'
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '正在处理出牌建议...'
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config: RunnableConfig) -> dict[str, Any]:
        current_state = self.graph.get_state(config)

        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, MahjongResponseFormat
        ):
            if structured_response.status in {'input_required', 'error'}:
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': '暂时无法处理你的请求，请稍后再试。',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
