import os
import sys

import click

from agent import MahjongAgent
from agent_executor import MahjongAgentExecutor
from dotenv import load_dotenv

from a2a.server import A2AServer
from a2a.server.request_handlers import DefaultA2ARequestHandler
from a2a.types import (
    AgentAuthentication,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)


load_dotenv()


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host: str, port: int):
    if not os.getenv('GOOGLE_API_KEY'):
        print('GOOGLE_API_KEY environment variable not set.')
        sys.exit(1)

    request_handler = DefaultA2ARequestHandler(
        agent_executor=MahjongAgentExecutor()
    )

    server = A2AServer(
        agent_card=get_agent_card(host, port), request_handler=request_handler
    )
    server.start(host=host, port=port)


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Mahjong Agent."""
    capabilities = AgentCapabilities(streaming=False, pushNotifications=True)
    skill = AgentSkill(
        id='mahjong_play',
        name='Mahjong Play Tool',
        description='AI麻将玩家，能自动决策打牌',
        tags=['mahjong', 'game', 'AI'],
        examples=['请帮我出一张牌'],
    )
    return AgentCard(
        name='Mahjong Agent',
        description='AI麻将玩家，能自动决策打牌',
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
