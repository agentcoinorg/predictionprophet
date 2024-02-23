import typer
import getpass
import typing as t
from prediction_market_agent_tooling.markets.markets import MarketType
from evo_researcher.deployment.models import DeployableAgentER, DeployableAgentER_EvoGPT3, DeployableAgentER_OlasEmbeddingOA


DEPLOYABLE_AGENTS = [
    DeployableAgentER_EvoGPT3,
    DeployableAgentER_OlasEmbeddingOA
]


def deploy(
    deployable_agent_name: str,
    manifold_api_key_secret_name: str,
    openai_api_key_secret_name: str,
    tavity_api_key_secret_name: str,
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
        repository="git+https://github.com/polywrap/evo.researcher.git@peter/new-deployment",
        market_type=MarketType.MANIFOLD,
        memory=1024,
        labels={
            "owner": getpass.getuser().lower(),
            "deployable_agent_name": deployable_agent_name.lower(),
        },
        secrets={
            "MANIFOLD_API_KEY": f"{manifold_api_key_secret_name}:latest",
            "OPENAI_API_KEY": f"{openai_api_key_secret_name}:latest",
            "TAVILY_API_KEY": f"{tavity_api_key_secret_name}:latest",
        },
    )


if __name__ == "__main__":
    typer.run(deploy)
