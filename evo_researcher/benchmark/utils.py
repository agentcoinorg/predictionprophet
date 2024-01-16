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
    def __init__(self, p_yes: float, confidence: float, info_utility: float):
        self.p_yes = p_yes
        self.confidence = confidence
        self.info_utility = info_utility

    def __repr__(self):
        return f"PredictionResult: p_yes:{self.p_yes}, confidence:{self.confidence}, info_utility:{self.info_utility}"


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
