import typer
import getpass
import typing as t
from git import Repo
from datetime import datetime
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_prophet.deployment.models import DeployableAgentER, DeployableAgentER_PredictionProphetGPT3, DeployableAgentER_OlasEmbeddingOA
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import private_key_type, DatetimeWithTimezone
from prediction_market_agent_tooling.tools.web3_utils import verify_address
from pydantic.types import SecretStr


DEPLOYABLE_AGENTS = [
    DeployableAgentER_PredictionProphetGPT3,
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
        repository=f"git+https://github.com/polywrap/predictionprophet.git@{Repo('.').active_branch.name}",
        market_type=market_type,
        api_keys=APIKeys(
            BET_FROM_ADDRESS=verify_address(bet_from_address),
            MANIFOLD_API_KEY=SecretStr(f"{manifold_api_key_secret_name}:latest"),
            BET_FROM_PRIVATE_KEY=private_key_type(f"{bet_from_private_key_secret_name}:latest"),
        ),
        memory=2048,
        labels={
            "owner": getpass.getuser().lower(),
        },
        secrets={
            "OPENAI_API_KEY": f"{openai_api_key_secret_name}:latest",
            "TAVILY_API_KEY": f"{tavity_api_key_secret_name}:latest",
            "GOOGLE_SEARCH_API_KEY": f"{google_search_api_key_name}:latest",
            "GOOGLE_SEARCH_ENGINE_ID": f"{google_search_engine_id_name}:latest",
        },
        cron_schedule="0 */2 * * *",
        start_time=DatetimeWithTimezone(start_time) if start_time else None,
    )


if __name__ == "__main__":
    typer.run(deploy)
