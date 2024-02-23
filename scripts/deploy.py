import typer
import getpass
import typing as t
from prediction_market_agent_tooling.markets.markets import MarketType
from evo_researcher.benchmark.agents import EvoAgent, OlasAgent, EmbeddingModel
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType


class DeployableAgentER(DeployableAgent):
    agent: AbstractBenchmarkedAgent

    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        """
        Testing mode: Pick only one predictable market or nothing.
        """
        for market in markets:
            print(f"Verifying market predictability for '{market.question}'.")
            if self.agent.is_predictable(market.question):
                print(f"Market '{market.question}' is predictable.")
                return [market]
        return []

    def answer_binary_market(self, market: AgentMarket) -> bool:
        prediciton = self.agent.predict(market.question)  # Already checked in the `pick_markets`.
        if prediciton.outcome_prediction is None:
            raise ValueError(f"Missing prediction: {prediciton}")
        binary_answer: bool = prediciton.outcome_prediction.p_yes > 0.5
        print(f"Answering '{market.question}' with '{binary_answer}'.")
        return binary_answer
    

class DeployableAgentER_EvoGPT3(DeployableAgentER):
    agent = EvoAgent(model="gpt-3.5-turbo-0125")


class DeployableAgentER_OlasEmbeddingOA(DeployableAgentER):
    agent = OlasAgent(model="gpt-3.5-turbo-0125", embedding_model=EmbeddingModel.openai)


DEPLOYABLE_AGENTS = [
    DeployableAgentER_EvoGPT3,
    DeployableAgentER_OlasEmbeddingOA
]


def deploy(
    deployable_agent_name: str,
    manifold_api_key_secret_name: str,
    openai_api_key_secret_name: str,
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
        },
    )


if __name__ == "__main__":
    typer.run(deploy)
