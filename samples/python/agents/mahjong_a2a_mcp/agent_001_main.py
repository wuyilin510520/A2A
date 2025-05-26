import os
import sys
import asyncio
import click
import uvicorn

from agent_001 import MahjongAgent
from agent_001_executor import MahjongAgentExecutor
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from a2a.server import A2AServer
from a2a.server.request_handlers import DefaultA2ARequestHandler
from a2a.types import (
    AgentAuthentication,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

load_dotenv()

async def async_main(host: str, port: int):
    if not os.getenv('GOOGLE_API_KEY'):
        print('GOOGLE_API_KEY environment variable not set.')
        sys.exit(1)

    # 手动管理client生命周期
    mcp_client = MultiServerMCPClient({
        "mahjong-game": {
            "url": "http://localhost:5000/sse",
            "transport": "sse",
        }
    })
    await mcp_client.__aenter__()
    print("MCP服务器连接成功")
    try:
        request_handler = DefaultA2ARequestHandler(
            agent_executor=MahjongAgentExecutor(mcp_client=mcp_client)
        )

        server = A2AServer(
            agent_card=get_agent_card(host, port), 
            request_handler=request_handler
        )
        config = uvicorn.Config(server.app(), host=host, port=port)
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()
    finally:
        await mcp_client.__aexit__(None, None, None)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host: str, port: int):
    asyncio.run(async_main(host, port))

def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Mahjong Agent."""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id='mahjong_play',
        name='Mahjong Play Tool',
        description='AI麻将玩家1号位，能自动决策打牌',
        tags=['mahjong', 'game', 'AI'],
        examples=['请帮我出一张牌'],
    )
    return AgentCard(
        name='Mahjong Agent 001',
        description='AI麻将玩家1号位，能自动决策打牌',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=MahjongAgent.SUPPORTED_CONTENT_TYPES,        
        defaultOutputModes=MahjongAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
        authentication=AgentAuthentication(schemes=['public']),
    )

if __name__ == '__main__':
    main()
