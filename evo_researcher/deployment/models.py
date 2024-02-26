from evo_researcher.benchmark.agents import EvoAgent, OlasAgent, EmbeddingModel
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.deploy.agent import DeployableAgent


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
