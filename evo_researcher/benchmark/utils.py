import requests
import typing as t


class Market:
    def __init__(self, market_json: dict):
        self.question = market_json["question"]
        self.url = market_json["url"]
        self.p_yes = market_json["probability"]
        self.volume = market_json["volume"]
        self.is_resolved = market_json["isResolved"]

    def __repr__(self):
        return f"Market: {self.question}, p_yes:{self.p_yes}"


class PredictionResult:
    def __init__(
        self,
        p_yes: float,
        confidence: float,
        info_utility: float,
        time: t.Optional[float] = None,
        cost: t.Optional[float] = None,
    ):
        self.p_yes = p_yes
        self.confidence = confidence
        self.info_utility = info_utility
        self.time = time
        self.cost = cost

    def __repr__(self):
        return f"PredictionResult: p_yes:{self.p_yes}, confidence:{self.confidence}, info_utility:{self.info_utility}, time:{self.time}, cost:{self.cost}"


def get_manifold_markets(number: int = 100) -> t.List[Market]:
    url = "https://api.manifold.markets/v0/search-markets"
    params = {
        "term": "",
        "sort": "liquidity",
        "filter": "open",
        "limit": f"{number}",
        "contractType": "BINARY",  # TODO support CATEGORICAL markets
    }
    response = requests.get(url, params=params)

    response.raise_for_status()
    markets_json = response.json()
    markets = [Market(m) for m in markets_json]
    markets = [m for m in markets if not m.is_resolved]
    assert len(markets) == number
    return markets


def get_llm_api_call_cost(model: str, prompt_tokens: int, completion_tokens) -> float:
    """
    In older versions of langchain, the cost calculation doesn't work for
    newer models. This is a temporary workaround to get the cost.

    See:
    https://github.com/langchain-ai/langchain/issues/12994

    Costs are in USD, per 1000 tokens.
    """
    model_costs = {
        "gpt-4-1106-preview": {
            "prompt_tokens": 0.01,
            "completion_tokens": 0.03,
        },
    }
    if model not in model_costs:
        raise ValueError(f"Unknown model: {model}")

    model_cost = model_costs[model]["prompt_tokens"] * prompt_tokens
    model_cost += model_costs[model]["completion_tokens"] * completion_tokens
    model_cost /= 1000
    return model_cost
