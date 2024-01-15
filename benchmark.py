"""
Take a list of markets from Manifold

For each market
    - generate evo report, make prediction
    - generate olas report, make prediction
    - record manifold market prediction

Track metrics:
    - Mean cost
    - Mean time
    - MSE for `p_yes` of prediction, vs manifold
    - MSE (normalized for 'confidence') of prediction, vs manifold
    - Mean confidence
    - Mean info_utility

Generate markdown comparison report
"""

import dotenv
import json
import os
import pandas as pd
import requests
import typing as t

from evo_researcher.functions.research import research as research_evo
from evo_researcher.autonolas.research import (
    make_prediction,
    research as research_autonolas,
)


class Market:
    def __init__(self, market_json: dict):
        self.question = market_json["question"]
        self.url = market_json["url"]
        self.p_yes = market_json["probability"]  # TODO support categorical markets
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


class ComparisonReport:
    def __init__(self):
        self.manifold_markets: t.List[Market] = []
        self.predictions: t.Dict[str, t.List[PredictionResult]] = {
            "olas": [],
            "evo": [],
        }
        self.metrics = {}

    def add_evo_prediction(self, prediction: PredictionResult):
        self.predictions["evo"].append(prediction)

    def add_olas_prediction(self, prediction: PredictionResult):
        self.predictions["olas"].append(prediction)

    def add_manifold_market(self, market: Market):
        self.manifold_markets.append(market)

    def generate_summary_statistics(self) -> t.Dict[str, t.List[t.Any]]:
        summary_statistics = {}
        summary_statistics["Agents"] = list(self.predictions.keys())

        summary_statistics["MSE for `p_yes`"] = []
        for agent in self.predictions.keys():
            mse = sum(
                [
                    (p.p_yes - m.p_yes) ** 2
                    for p, m in zip(self.predictions[agent], self.manifold_markets)
                ]
            )
            mse /= len(self.predictions[agent])
            summary_statistics["MSE for `p_yes`"].append(mse)

        summary_statistics["Normalized MSE for `p_yes`"] = []
        for agent in self.predictions.keys():
            mse = sum(
                [
                    ((p.p_yes - m.p_yes) * p.confidence) ** 2
                    for p, m in zip(self.predictions[agent], self.manifold_markets)
                ]
            )
            mse /= len(self.predictions[agent])
            summary_statistics["Normalized MSE for `p_yes`"].append(mse)

        summary_statistics["Mean confidence"] = []
        for agent in self.predictions.keys():
            mean_confidence = sum(
                [p.confidence for p in self.predictions[agent]]
            ) / len(self.predictions[agent])
            summary_statistics["Mean confidence"].append(mean_confidence)

        summary_statistics["Mean info_utility"] = []
        for agent in self.predictions.keys():
            mean_info_utility = sum(
                [p.info_utility for p in self.predictions[agent]]
            ) / len(self.predictions[agent])
            summary_statistics["Mean info_utility"].append(mean_info_utility)

        # TODO add mean cost and mean time

        return summary_statistics

    def generate_markets_summary(self) -> t.Dict[str, t.List[str]]:
        market_questions = [q.question for q in self.manifold_markets]
        urls = [q.url for q in self.manifold_markets]
        markets_summary = {
            "Market Question": [
                f"[{question}]({url})" for question, url in zip(market_questions, urls)
            ],
        }
        for model_type in self.predictions.keys():
            markets_summary[f"{model_type} p_yes"] = [
                p.p_yes for p in self.predictions[model_type]
            ]
        markets_summary["manifold p_yes"] = [m.p_yes for m in self.manifold_markets]
        return markets_summary

    def generate_markdown_report(self):
        md = "# Comparison Report\n\n"
        md += "## Summary Statistics\n\n"
        md += pd.DataFrame(self.generate_summary_statistics()).to_markdown(index=False)
        md += "\n\n"
        md += "## Markets\n\n"
        md += pd.DataFrame(self.generate_markets_summary()).to_markdown(index=False)
        return md


def get_manifold_markets() -> t.List[Market]:
    url = "https://api.manifold.markets/v0/search-markets"
    params = {
        "term": "",
        "sort": "liquidity",
        "filter": "open",
        "limit": "100",
        "contractType": "BINARY",  # TODO support CATEGORICAL markets
    }
    response = requests.get(url, params=params)

    response.raise_for_status()
    markets_json = response.json()
    markets = [Market(m) for m in markets_json]
    markets = [m for m in markets if not m.is_resolved]
    assert len(markets) == 100
    return markets


def parse_prediction_str(prediction: str) -> PredictionResult:
    """
    Parse a prediction string of the form:

    ```json
    {
        "p_yes": 0.6,
        "p_no": 0.4,
        "confidence": 0.8,
        "info_utility": 0.9
    }
    ```

    into a PredictionResult object
    """
    start_index = prediction.find("{")
    end_index = prediction.rfind("}")
    prediction = prediction[start_index : end_index + 1]
    prediction_json = json.loads(prediction)
    return PredictionResult(
        p_yes=prediction_json["p_yes"],
        confidence=prediction_json["confidence"],
        info_utility=prediction_json["info_utility"],
    )


def _make_prediction(
    market_question: str, additional_information: str
) -> PredictionResult:
    # prediction: str = make_prediction(
    #     prompt=market_question, additional_information=additional_information
    # )
    prediction = (
        "```json\n"
        "{\n"
        '  "p_yes": 0.6,\n'
        '  "p_no": 0.4,\n'
        '  "confidence": 0.8,\n'
        '  "info_utility": 0.9\n'
        "}\n"
        "```\n"
    )
    prediction: PredictionResult = parse_prediction_str(prediction)
    return prediction


def research_and_predict_evo(market_question: str, model: str) -> PredictionResult:
    dotenv.load_dotenv()
    open_ai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    # report, _ = research_evo(
    #     goal=market_question,
    #     openai_key=open_ai_key,
    #     tavily_key=tavily_key,
    #     model=model,
    # )
    report = "TODO"
    return _make_prediction(
        market_question=market_question, additional_information=report
    )


def research_and_predict_olas(market_question: str, model: str) -> PredictionResult:
    # report = research_autonolas(
    #     prompt=market_question,
    #     engine=model,
    # )
    report = "TODO"
    return _make_prediction(
        market_question=market_question, additional_information=report
    )


if __name__ == "__main__":
    cr = ComparisonReport()
    # TODO register agents for comparison
    model = "gpt-4-1106-preview"
    for market in get_manifold_markets()[:2]:  # Pick first 2 markets for now
        result_evo = research_and_predict_evo(
            market_question=market.question, model=model
        )
        result_olas = research_and_predict_olas(
            market_question=market.question, model=model
        )
        cr.add_evo_prediction(result_evo)
        cr.add_olas_prediction(result_olas)
        cr.add_manifold_market(market)
    md = cr.generate_markdown_report()

    with open("/Users/evan/scratch/report.md", "w") as f:
        f.write(md)
