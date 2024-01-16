import argparse
import pandas as pd
import typing as t

from evo_researcher.benchmark.agents import (
    AbstractBenchmarkedAgent,
    EvoAgent,
    OlasAgent,
)
from evo_researcher.benchmark.utils import (
    Market,
    PredictionResult,
    get_manifold_markets,
)


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

    benchmarker = Benchmarker(
        markets=get_manifold_markets()[:3],  # Pick first 3 markets for now
        agents=[
            OlasAgent(model="gpt-3.5-turbo"),  # TODO use same models!
            EvoAgent(model="gpt-4-1106-preview"),
        ],
    )
    benchmarker.run_agents()
    md = benchmarker.generate_markdown_report()

    with open(args.output, "w") as f:
        print(f"Writing benchmark report to: {args.output}")
        f.write(md)
