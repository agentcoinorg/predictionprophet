from decimal import Decimal
from loguru import logger
from datetime import timedelta
from prediction_prophet.benchmark.agents import PredictionProphetAgent, OlasAgent, EmbeddingModel
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.manifold.manifold import ManifoldAgentMarket
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_contracts import WrappedxDaiContract
from prediction_market_agent_tooling.deploy.agent import DeployableAgent, BetAmount, MarketType
from prediction_market_agent_tooling.tools.betting_strategies.stretch_bet_between import stretch_bet_between
from prediction_market_agent_tooling.markets.manifold.api import get_manifold_bets, get_authenticated_user, get_manifold_market
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler
from prediction_market_agent_tooling.tools.utils import should_not_happen, utcnow, prob_uncertainty
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import Probability


class DeployableAgentER(DeployableAgent):
    agent: AbstractBenchmarkedAgent
    max_markets_per_run = 5

    def recently_betted(self, market: AgentMarket) -> bool:
        start_time = utcnow() - timedelta(hours=24)
        keys = APIKeys()
        recently_betted_questions = set(get_manifold_market(b.contractId).question for b in get_manifold_bets(
            user_id=get_authenticated_user(keys.manifold_api_key.get_secret_value()).id,
            start_time=start_time,
            end_time=None,
        )) if isinstance(market, ManifoldAgentMarket) else set(b.title for b in OmenSubgraphHandler().get_bets(
            better_address=keys.bet_from_address,
            start_time=start_time,
        )) if isinstance(market, OmenAgentMarket) else should_not_happen(f"Uknown market: {market}")
        return market.question in recently_betted_questions

    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        picked_markets: list[AgentMarket] = []
        for market in markets:
            logger.info(f"Looking if we recently bet on '{market.question}'.")
            if self.recently_betted(market):
                logger.info("Recently betted, skipping.")
                continue
            logger.info(f"Verifying market predictability for '{market.question}'.")
            if self.agent.is_predictable(market.question):
                logger.info(f"Market '{market.question}' is predictable.")
                picked_markets.append(market)
            if len(picked_markets) >= self.max_markets_per_run:
                break
        return picked_markets
    
    def calculate_bet_amount(self, answer: bool, market: AgentMarket) -> BetAmount:
        amount: Decimal
        max_bet_amount: float
        if isinstance(market, ManifoldAgentMarket) :
            # Manifold won't give us fractional Mana, so bet the minimum amount to win at least 1 Mana.
            amount = market.get_minimum_bet_to_win(answer, amount_to_win=1) 
            max_bet_amount = 10.0
        elif isinstance(market, OmenAgentMarket):
            # TODO: After https://github.com/gnosis/prediction-market-agent-tooling/issues/161 is done,
            # use agent's probability to calculate the amount.
            market_liquidity = market.get_liquidity_in_xdai()
            amount = Decimal(stretch_bet_between(
                Probability(prob_uncertainty(market.p_yes)),  # Not a probability, but it's a value between 0 and 1, so it's fine.
                min_bet=0.5, 
                max_bet=1.0,
            ))
            if answer == (market.p_yes > 0.5):
                amount = amount * Decimal(0.75)
            else:
                amount = amount * Decimal(1.25)
            max_bet_amount = 2.0 if market_liquidity > 5 else 0.1 if market_liquidity > 1 else 0
        else:
            should_not_happen(f"Unknown market type: {market}")
        if amount > max_bet_amount:
            logger.warning(f"Calculated amount {amount} {market.currency} is exceeding our limit {max_bet_amount=}, betting only {market.get_tiny_bet_amount()} for benchmark purposes.")
            amount = market.get_tiny_bet_amount().amount
        return BetAmount(amount=amount, currency=market.currency)

    def answer_binary_market(self, market: AgentMarket) -> bool | None:
        prediciton = self.agent.predict(market.question)  # Already checked in the `pick_markets`.
        if prediciton.outcome_prediction is None:
            logger.error(f"Prediction failed for {market.question}.")
            return None
        binary_answer: bool = prediciton.outcome_prediction.p_yes > 0.5
        logger.info(f"Answering '{market.question}' with '{binary_answer}'.")
        return binary_answer

    def before(self, market_type: MarketType) -> None:
        keys = APIKeys()
        wxdai = WrappedxDaiContract()
        current_wxdai_balance = 0

        if market_type == MarketType.OMEN:
            current_wxdai_balance = wxdai.balanceOf(keys.bet_from_address)
            logger.info(f"My current wxDai balance is {current_wxdai_balance} Wei")

        super().before(market_type)

        if market_type == MarketType.OMEN:
            new_wxdai_balance = wxdai.balanceOf(keys.bet_from_address)
            logger.info(f"My wxDai balance after redeeming is {new_wxdai_balance} Wei, so {new_wxdai_balance - current_wxdai_balance} Wei was redeemed.")

class DeployableAgentER_PredictionProphetGPT3(DeployableAgentER):
    agent = PredictionProphetAgent(model="gpt-3.5-turbo-0125")


class DAPredictionProphetGPT4(DeployableAgentER):
    agent = PredictionProphetAgent(model="gpt-4-0125-preview")
    # Limit to just 1, because so far it seems that 20x higher costs aren't justified by the prediction performance.
    max_markets_per_run = 1

class DeployableAgentER_OlasEmbeddingOA(DeployableAgentER):
    agent = OlasAgent(model="gpt-3.5-turbo-0125", embedding_model=EmbeddingModel.openai)
