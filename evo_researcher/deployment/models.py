from decimal import Decimal
from evo_researcher.benchmark.agents import EvoAgent, OlasAgent, EmbeddingModel
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.manifold.manifold import ManifoldAgentMarket
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.deploy.agent import DeployableAgent, BetAmount
from prediction_market_agent_tooling.markets.betting_strategies import minimum_bet_to_win


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
    
    def calculate_bet_amount(self, answer: bool, market: AgentMarket) -> BetAmount:
        amount: Decimal
        max_bet_amount: float
        if isinstance(market, ManifoldAgentMarket) :
            # Manifold won't give us fractional Mana, so bet the minimum amount to win at least 1 Mana.
            amount = market.get_minimum_bet_to_win(answer, amount_to_win=1) 
            max_bet_amount = 10.0
        else:
            # Otherwise, bet to win at least 0.001 (of something), unless the bet would be less than the tiny bet.
            amount = max(
                Decimal(minimum_bet_to_win(answer, amount_to_win=0.001, market=market)), 
                market.get_tiny_bet_amount().amount,
            )
            max_bet_amount = 0.1
        if amount > max_bet_amount:
            print(f"Would need at least {amount} {market.currency} to be profitable, betting only {market.get_tiny_bet_amount()} for benchmark purposes.")
            amount = market.get_tiny_bet_amount().amount
        return BetAmount(amount=amount, currency=market.currency)

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
