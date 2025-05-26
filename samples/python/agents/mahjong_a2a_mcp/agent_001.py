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
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse
from langchain_core.messages import AnyMessage
logger = logging.getLogger(__name__)
# 创建MCP客户端实例

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


class CustomState(AgentStateWithStructuredResponse):
    player_view: str

def prompt(
    state: CustomState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    player_view = state["player_view"]
    player_id = config["configurable"].get("player_id", 999)
    system_msg = (
        f"你是麻将桌上的{player_id}号位，性格火辣、直爽的女生麻将玩家，说话风格大胆、幽默，偶尔带点俏皮。"
        "麻将缩写说明：M=万，P=筒，S=条，Z=字牌（1Z=东风，2Z=南风，3Z=西风，4Z=北风，5Z=中，6Z=发，7Z=白）。"
        "你能用自然语言描述这些牌。"
        "你现在看到的麻将视角信息如下：\n"
        f"{player_view}\n"
        "请用你的风格，结合这些信息，机智地回答用户的问题:"
    )
    return [{"role": "system", "content": system_msg}] + state["messages"]

# 决策脑prompt
def decision_prompt(state, config):
    player_view = state["player_view"]
    player_id = config["configurable"].get("player_id", 999)
    persona = config["configurable"].get("persona", "普通玩家")
    system_msg = (
        f"你是麻将桌上的{player_id}号位，{persona}风格的麻将玩家"
        "麻将缩写说明：M=万，P=筒，S=条，Z=字牌（1Z=东风，2Z=南风，3Z=西风，4Z=北风，5Z=中，6Z=发，7Z=白）。"
        "你能用自然语言描述这些牌。"
        "你现在看到的麻将视角信息如下：\n"
        f"{player_view}\n"
        "其中，hand表示你的手牌，is_current_player表示是否轮到你出牌，current_player表示当前出牌的玩家编号，wall_count表示牌墙剩余数量，discards表示已打出的牌，melds表示已经组成的面子(顺子、刻子、杠子)，hand_count表示手牌数量，others表示其他玩家的信息（包括手牌数量、已打出的牌和已组成的面子）。"
        "和你数字差2n-1(n=1,2)的玩家是你的隔壁玩家，和你数字差2n(n=1)的玩家是你的对家，如果你是玩家3，那你的下家是玩家0，场上的玩家为0,1,2,3"
        "必须需要使用工具discard_tile来打出一张牌，出完牌之后，必须告知其他人你出了哪张牌"
        "打完牌后，直接给出你打出的牌，并用你的风格简要说明理由。输出格式：建议: <牌> 理由: <理由>"

    )
    return [{"role": "system", "content": system_msg}] + state["messages"]



class MahjongAgent:
    """麻将AI Agent 示例。"""
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
        # 决策脑
        self.decision_graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=decision_prompt,  # 用自定义prompt
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, MahjongResponseFormat),
            state_schema=CustomState,
        )
        

    def invoke(self, query: str, sessionId: str) -> dict[str, Any]:
        config: RunnableConfig = {'configurable': {'thread_id': sessionId}}
        self.graph.invoke({'messages': [('user', query)]}, config)
        return self.get_agent_response(config)

    async def stream(
        self, query: str, sessionId: str, player_id: int = 0, persona: str = "女性玩家"
    ) -> AsyncIterable[dict[str, Any]]:
        resources = await self.mcp_client.get_resources("mahjong-game", f"resource://player/{player_id}/view")
        player_view = resources[0].as_string()
        decision_inputs: dict[str, Any] = {
            'messages': [('user', query)],
            'player_view': player_view
        }
        config: RunnableConfig = {'configurable': {'thread_id': sessionId, 'player_id': player_id, 'persona': persona}}

        async for item in self.decision_graph.astream(decision_inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            
            
            
            if (
                isinstance(message, AIMessage)
                # and message.tool_calls
                # and len(message.tool_calls) > 0
            ):
                
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f'玩家{player_id}正在分析...'
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f'玩家{player_id}正在出牌...'
                }

            


        

        yield self.get_agent_response(config)

    def get_agent_response(self, config: RunnableConfig) -> dict[str, Any]:
        current_state = self.decision_graph.get_state(config)

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
