import argparse
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
    prediction: str = make_prediction(
        prompt=market_question, additional_information=additional_information
    )
    prediction: PredictionResult = parse_prediction_str(prediction)
    return prediction


class AbstractBenchmarkedAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def research_and_predict(self, market_question: str) -> PredictionResult:
        raise NotImplementedError


class OlasAgent(AbstractBenchmarkedAgent):
    def __init__(self, model: str):
        super().__init__(agent_name="olas")
        self.model = model

    def research_and_predict(self, market_question: str) -> PredictionResult:
        report = research_autonolas(
            prompt=market_question,
            engine=self.model,
        )
        return _make_prediction(
            market_question=market_question, additional_information=report
        )


class EvoAgent(AbstractBenchmarkedAgent):
    def __init__(self, model: str):
        super().__init__(agent_name="evo")
        self.model = model

    def research_and_predict(self, market_question: str) -> PredictionResult:
        dotenv.load_dotenv()
        open_ai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        report, _ = research_evo(
            goal=market_question,
            openai_key=open_ai_key,
            tavily_key=tavily_key,
            model=self.model,
        )
        return _make_prediction(
            market_question=market_question, additional_information=report
        )


class DummyAgent(AbstractBenchmarkedAgent):
    def __init__(self):
        super().__init__(agent_name="dummy")

    def research_and_predict(self, market_question: str) -> PredictionResult:
        return PredictionResult(p_yes=0.5, confidence=0.5, info_utility=0.5)


class Benchmarker:
    def __init__(
        self,
        markets: t.List[Market],
        agents: t.List[AbstractBenchmarkedAgent],
        metric_fns: t.Dict[str, t.Callable] = {},
    ):
        self.markets: t.List[Market] = markets
        self.registered_agents: t.List[AbstractBenchmarkedAgent] = agents

        # Predictions
        self.predictions: t.Dict[str, t.List[PredictionResult]] = {}
        for agent in self.registered_agents:
            self.predictions[agent.agent_name] = []

        # Metrics
        self.metric_fns = metric_fns
        predefined_metric_fns = {
            "MSE for `p_yes`": self._compute_mse,
            "Mean confidence": self._compute_mean_confidence,
            "Mean info_utility": self._compute_mean_info_utility,
            # TODO add mean cost and mean time
            # TODO add 'normalized' mse to take into account confidence?
        }
        self.metric_fns.update(predefined_metric_fns)

    def add_prediction(
        self, agent: AbstractBenchmarkedAgent, prediction: PredictionResult
    ):
        self.predictions[agent.agent_name].append(prediction)

    def run_agents(self):
        for agent in self.registered_agents:
            for market in self.markets:
                prediction = agent.research_and_predict(market_question=market.question)
                self.add_prediction(agent=agent, prediction=prediction)

    def _compute_mse(
        self, predictions: t.List[PredictionResult], markets: t.List[Market]
    ):
        mse = sum([(p.p_yes - m.p_yes) ** 2 for p, m in zip(predictions, markets)])
        mse /= len(predictions)
        return mse

    def _compute_mean_confidence(
        self, predictions: t.List[PredictionResult], markets: t.List[Market]
    ):
        mean_confidence = sum([p.confidence for p in predictions]) / len(predictions)
        return mean_confidence

    def _compute_mean_info_utility(
        self, predictions: t.List[PredictionResult], markets: t.List[Market]
    ):
        mean_info_utility = sum([p.info_utility for p in predictions]) / len(
            predictions
        )
        return mean_info_utility

    def compute_metrics(self) -> t.Dict[str, t.List[t.Any]]:
        metrics = {}
        metrics["Agents"] = list(self.predictions.keys())

        for name, fn in self.metric_fns.items():
            metrics[name] = []
            for agent in self.predictions.keys():
                metrics[name].append(
                    fn(predictions=self.predictions[agent], markets=self.markets)
                )

        return metrics

    def get_markets_summary(self) -> t.Dict[str, t.List[str]]:
        market_questions = [q.question for q in self.markets]
        urls = [q.url for q in self.markets]
        markets_summary = {
            "Market Question": [
                f"[{question}]({url})" for question, url in zip(market_questions, urls)
            ],
        }
        for model_type in self.predictions.keys():
            markets_summary[f"{model_type} p_yes"] = [
                p.p_yes for p in self.predictions[model_type]
            ]
        markets_summary["manifold p_yes"] = [m.p_yes for m in self.markets]
        return markets_summary

    def generate_markdown_report(self):
        md = "# Comparison Report\n\n"
        md += "## Summary Statistics\n\n"
        md += pd.DataFrame(self.compute_metrics()).to_markdown(index=False)
        md += "\n\n"
        md += "## Markets\n\n"
        md += pd.DataFrame(self.get_markets_summary()).to_markdown(index=False)
        return md


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--output",
        type=str,
        default="./benchmark_report.md",
    )
    args = args.parse_args()

    model = "gpt-4-1106-preview"
    benchmarker = Benchmarker(
        markets=get_manifold_markets()[:1],  # Pick first 1 markets for now
        agents=[
            OlasAgent(model=model),
            EvoAgent(model=model),
        ],
    )
    benchmarker.run_agents()
    md = benchmarker.generate_markdown_report()

    with open(args.output, "w") as f:
        f.write(md)
