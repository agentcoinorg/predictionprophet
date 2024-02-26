import typer
import getpass
import typing as t
from git import Repo
from datetime import datetime
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.deploy.agent import MonitorConfig
from prediction_market_agent_tooling.deploy.gcp.utils import gcp_get_secret_value
from evo_researcher.deployment.models import DeployableAgentER, DeployableAgentER_EvoGPT3, DeployableAgentER_OlasEmbeddingOA
from prediction_market_agent_tooling.markets.manifold.api import get_authenticated_user

DEPLOYABLE_AGENTS = [
    DeployableAgentER_EvoGPT3,
    DeployableAgentER_OlasEmbeddingOA
]


def deploy(
    market_type: MarketType,
    deployable_agent_name: str = typer.Option(),
    manifold_api_key_secret_name: str = typer.Option(),
    openai_api_key_secret_name: str = typer.Option(),
    tavity_api_key_secret_name: str = typer.Option(),
    bet_from_address: str = typer.Option(),
    bet_from_private_key_secret_name: str = typer.Option(),
    start_time: datetime | None = None,
) -> None:
    """
    Script to execute locally to deploy the agent to GCP.
    """
    if deployable_agent_name not in [
        agent.__name__ for agent in DEPLOYABLE_AGENTS
    ]:
        raise ValueError(f"Invalid agent name: {deployable_agent_name}")
    
    chosen_agent_class: t.Type[DeployableAgentER] = [agent for agent in DEPLOYABLE_AGENTS if agent.__name__ == deployable_agent_name][0]
    
    chosen_agent_class().deploy_gcp(
        repository=f"git+https://github.com/polywrap/evo.researcher.git@{Repo('.').active_branch.name}",
        market_type=market_type,
        memory=1024,
        labels={
            "owner": getpass.getuser().lower(),
        },
        env_vars={
            "BET_FROM_ADDRESS": bet_from_address,
        },
        secrets={
            "MANIFOLD_API_KEY": f"{manifold_api_key_secret_name}:latest",
            "OPENAI_API_KEY": f"{openai_api_key_secret_name}:latest",
            "TAVILY_API_KEY": f"{tavity_api_key_secret_name}:latest",
            "BET_FROM_PRIVATE_KEY": f"{bet_from_private_key_secret_name}:latest",
        },
        cron_schedule="0 */2 * * *",
        monitor_config=MonitorConfig(
            start_time=start_time or datetime.utcnow(),
            manifold_user_id=get_authenticated_user(gcp_get_secret_value(manifold_api_key_secret_name)).id,
            omen_public_key=bet_from_address,
        ),
    )


if __name__ == "__main__":
    typer.run(deploy)
