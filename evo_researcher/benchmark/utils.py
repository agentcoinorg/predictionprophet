import json
import os
import requests
import typing as t
from pydantic import BaseModel


class Market(BaseModel):
    question: str
    url: str
    p_yes: float
    volume: float
    is_resolved: bool


class PredictionResult(BaseModel):
    p_yes: float
    confidence: float
    info_utility: float
    time: t.Optional[float] = None
    cost: t.Optional[float] = None


AgentPredictions = t.Dict[str, PredictionResult]
Predictions = t.Dict[str, AgentPredictions]


class PredictionsCache(BaseModel):
    predictions: Predictions

    def get_prediction(self, agent_name: str, question: str) -> PredictionResult:
        return self.predictions[agent_name][question]

    def has_market(self, agent_name: str, question: str) -> bool:
        return (
            agent_name in self.predictions and question in self.predictions[agent_name]
        )

    def add_prediction(
        self, agent_name: str, question: str, prediction: PredictionResult
    ):
        if agent_name not in self.predictions:
            self.predictions[agent_name] = {}
        assert question not in self.predictions[agent_name]
        self.predictions[agent_name][question] = prediction

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "PredictionsCache":
        with open(path, "r") as f:
            return PredictionsCache.parse_obj(json.load(f))


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

    # Map JSON fields to Market fields
    fields_map = {
        "probability": "p_yes",
        "isResolved": "is_resolved",
    }

    def _map_fields(old: dict, mapping: dict) -> dict:
        return {mapping.get(k, k): v for k, v in old.items()}

    markets = [Market.parse_obj(_map_fields(m, fields_map)) for m in markets_json]
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
