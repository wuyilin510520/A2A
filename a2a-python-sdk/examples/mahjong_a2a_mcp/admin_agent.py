import logging
from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse
from langchain_core.messages import AnyMessage

logger = logging.getLogger(__name__)

class AdminResponseFormat(BaseModel):
    """管理员Agent响应格式。"""
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str

class CustomState(AgentStateWithStructuredResponse):
    game_state: str

def admin_prompt(state, config):
    system_msg = (
        "你是麻将游戏的管理员，负责协调游戏流程，处理异常情况。"
        "你的职责包括："
        "1. 监控游戏状态"
        "2. 确保玩家按正确顺序行动"
        "3. 处理游戏中的异常情况"
        "4. 切换玩家视角"
        "5. 维护游戏秩序"
        "当发生异常时，你需要："
        "1. 检查游戏状态"
        "2. 确定当前应该行动的玩家"
        "3. 切换到正确的玩家视角"
        "4. 确保游戏继续进行"
        "请用专业、冷静的语气处理问题。"
    )
    return [{"role": "system", "content": system_msg}] + state["messages"]

class MahjongAdminAgent:
    """麻将游戏管理员Agent。"""
    RESPONSE_FORMAT_INSTRUCTION: str = (
        'Select status as completed if the request is complete'
        'Select status as input_required if the input is a question to the user'
        'Set response status to error if the input indicates an error'
    )

    def __init__(self, mcp_client: MultiServerMCPClient):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        self.mcp_client = mcp_client
        self.tools = self.mcp_client.get_tools()
        self.memory = MemorySaver()
        self.admin_graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=admin_prompt,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, AdminResponseFormat),
            state_schema=CustomState,
        )

    async def stream(
        self, query: str, sessionId: str
    ) -> AsyncIterable[dict[str, Any]]:
        try:
            # 获取全局游戏状态
            resources = await self.mcp_client.get_resources("mahjong-game", "resource://admin/game_state")
            game_state = resources[0].as_string()
            
            admin_inputs = {
                'messages': [('user', query)],
                'game_state': game_state
            }
            config: RunnableConfig = {'configurable': {'thread_id': sessionId}}

            async for item in self.admin_graph.astream(admin_inputs, config, stream_mode='values'):
                message = item['messages'][-1]
                if isinstance(message, AIMessage):
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': '管理员正在分析游戏状态...'
                    }
                elif isinstance(message, ToolMessage):
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': '管理员正在执行操作...'
                    }

            yield self.get_admin_response(config)
            
        except Exception as e:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f'管理员处理异常失败：{str(e)}，请重试。'
            }

    def get_admin_response(self, config: RunnableConfig) -> dict[str, Any]:
        current_state = self.admin_graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(structured_response, AdminResponseFormat):
            return {
                'is_task_complete': structured_response.status == 'completed',
                'require_user_input': structured_response.status == 'input_required',
                'content': structured_response.message,
            }
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': '管理员处理失败，请重试。',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain'] 