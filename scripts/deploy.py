import os
import typer
import typing as t
from prediction_market_agent_tooling.deploy.gcp.deploy import (
    deploy_to_gcp,
    run_deployed_gcp_function,
    schedule_deployed_gcp_function,
)
from prediction_market_agent_tooling.deploy.gcp.utils import gcp_function_is_active
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.config import APIKeys
from flask.wrappers import Request
import functions_framework
import random
from evo_researcher.benchmark.agents import EvoAgent, OlasAgent
from pydantic import ConfigDict
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.markets.data_models import AgentMarket
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType


class DeployableAgentER(DeployableAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # TODO: Remove once DeployableAgent isn't BaseModel anymore.
    agent: AbstractBenchmarkedAgent

    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        return random.sample(markets, 1)

    def answer_binary_market(self, market: AgentMarket) -> bool:
        prediciton = self.agent.evaluate_research_predict(market.question)
        if prediciton.outcome_prediction is None:
            raise ValueError(f"Missing prediction: {prediciton}")
        binary_answer: bool = prediciton.outcome_prediction.binary_answer
        return binary_answer


DEPLOYABLE_AGENTS: dict[str, DeployableAgentER] = {
    "DeployableAgentER_OlasReference": DeployableAgentER(
        agent=OlasAgent(model="gpt-3.5-turbo", temperature=0.7),
    ),
    "DeployableAgentER_OlasCheaperZeroTemp": DeployableAgentER(
        agent=OlasAgent(model="gpt-3.5-turbo-0125", agent_name="olas_gpt-3.5-turbo-0125"),
    ),
    "DeployableAgentER_EvoSummary": DeployableAgentER(
        agent=EvoAgent(model="gpt-3.5-turbo-0125", use_summaries=True),
    ),
}


@functions_framework.http
def main(request: Request) -> str:
    """
    Entrypoint for the deployed function.
    """
    agent_name = os.getenv("DEPLOYABLE_AGENT_NAME")
    if agent_name not in DEPLOYABLE_AGENTS:
        raise ValueError(f"Invalid agent name: {agent_name}")
    DEPLOYABLE_AGENTS[agent_name].run(market_type=MarketType.MANIFOLD)
    return "Success"


def deploy(
    deployable_agent_name: str,
) -> None:
    """
    Script to execute locally to deploy the agent to GCP.
    """
    if deployable_agent_name not in DEPLOYABLE_AGENTS:
        raise ValueError(f"Invalid agent name: {deployable_agent_name}")

    fname = deploy_to_gcp(
        requirements_file=None,
        extra_deps=[
            "git+https://github.com/polywrap/evo.researcher.git@peter/pmat"
        ],
        function_file=os.path.abspath(__file__),
        market_type=MarketType.MANIFOLD,
        api_keys={
            "MANIFOLD_API_KEY": APIKeys().manifold_api_key,
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "DEPLOYABLE_AGENT_NAME": deployable_agent_name,
        },
        memory=512,
    )

    # Check that the function is deployed
    assert gcp_function_is_active(fname), "Failed to deploy the function"

    # Run the function
    response = run_deployed_gcp_function(fname)
    assert response.ok, "Failed to run the deployed function"

    # Schedule the function
    schedule_deployed_gcp_function(fname, cron_schedule="0 */2 * * *")


if __name__ == "__main__":
    typer.run(deploy)
