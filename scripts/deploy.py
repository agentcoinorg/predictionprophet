import typer
import getpass
import typing as t
from datetime import datetime
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_prophet.deployment.models import DeployableAgentER, DeployableAgentER_PredictionProphetGPT3, DeployableAgentER_PredictionProphetGPT4, DeployableAgentER_OlasEmbeddingOA
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.tools.utils import get_current_git_commit_sha, get_current_git_url
from prediction_market_agent_tooling.gtypes import private_key_type, DatetimeWithTimezone
from pydantic.types import SecretStr


DEPLOYABLE_AGENTS = [
    DeployableAgentER_PredictionProphetGPT3,
    DeployableAgentER_PredictionProphetGPT4,
    DeployableAgentER_OlasEmbeddingOA
]


def deploy(
    market_type: MarketType,
    deployable_agent_name: str = typer.Option(),
    manifold_api_key_secret_name: str = typer.Option(),
    openai_api_key_secret_name: str = typer.Option(),
    tavity_api_key_secret_name: str = typer.Option(),
    google_search_api_key_name: str = typer.Option(),
    google_search_engine_id_name: str = typer.Option(),
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
        repository=f"git+{get_current_git_url()}@{get_current_git_commit_sha()}",
        market_type=market_type,
        api_keys=APIKeys(
            MANIFOLD_API_KEY=SecretStr(f"{manifold_api_key_secret_name}:latest"),
            BET_FROM_PRIVATE_KEY=private_key_type(f"{bet_from_private_key_secret_name}:latest"),
            OPENAI_API_KEY=SecretStr(f"{openai_api_key_secret_name}:latest"),
            GOOGLE_SEARCH_API_KEY=SecretStr(f"{google_search_api_key_name}:latest"),
            GOOGLE_SEARCH_ENGINE_ID=SecretStr(f"{google_search_engine_id_name}:latest"),
        ),
        memory=2048,
        labels={
            "owner": getpass.getuser().lower(),
        },
        secrets={
            "TAVILY_API_KEY": f"{tavity_api_key_secret_name}:latest",
        },
        cron_schedule="0 */2 * * *",
        start_time=DatetimeWithTimezone(start_time) if start_time else None,
    )


if __name__ == "__main__":
    typer.run(deploy)
