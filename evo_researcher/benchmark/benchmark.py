import argparse
import concurrent.futures
import pandas as pd
import time
import typing as t

from langchain.callbacks import get_openai_callback

from evo_researcher.benchmark.agents import (
    AbstractBenchmarkedAgent,
    EvoAgent,
    OlasAgent,
)
from evo_researcher.benchmark.utils import (
    Market,
    PredictionResult,
    get_llm_api_call_cost,
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
        self.predictions: t.Dict[str, t.List[PredictionResult]] = {
            agent.agent_name: [] for agent in self.registered_agents
        }

        # Metrics
        self.metric_fns = metric_fns
        predefined_metric_fns = {
            "MSE for `p_yes`": self._compute_mse,
            "Mean confidence": self._compute_mean_confidence,
            "Mean info_utility": self._compute_mean_info_utility,
            "Mean cost ($)": self._compute_mean_cost,
            "Mean time (s)": self._compute_mean_time,
            # TODO add 'normalized' mse to take into account confidence?
        }
        self.metric_fns.update(predefined_metric_fns)

    def add_prediction(
        self, agent: AbstractBenchmarkedAgent, prediction: PredictionResult
    ):
        self.predictions[agent.agent_name].append(prediction)

    def run_agents(self):
        for agent in self.registered_agents:

            def get_prediction_result(market: Market):
                with get_openai_callback() as cb:
                    start = time.time()
                    prediction = agent.research_and_predict(
                        market_question=market.question
                    )
                    prediction.time = time.time() - start

                    if cb.total_tokens > 0 and cb.total_cost == 0:
                        # TODO: this is a hack to get the cost for an unsupported model
                        cb.total_cost = get_llm_api_call_cost(
                            model=agent.model,
                            prompt_tokens=cb.prompt_tokens,
                            completion_tokens=cb.completion_tokens,
                        )
                    prediction.cost = cb.total_cost
                return prediction

            # Run agents in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=agent.max_workers
            ) as executor:
                future_to_market = {
                    executor.submit(get_prediction_result, market): market
                    for market in self.markets
                }
                for future in concurrent.futures.as_completed(future_to_market):
                    self.add_prediction(agent=agent, prediction=future.result())

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

    def _compute_mean_cost(
        self, predictions: t.List[PredictionResult], markets: t.List[Market]
    ):
        # Note: costs are optional
        costs = [p.cost for p in predictions if p.cost]
        if costs:
            return sum(costs) / len(costs)
        else:
            return None

    def _compute_mean_time(
        self, predictions: t.List[PredictionResult], markets: t.List[Market]
    ):
        # Note: times are optional
        times = [p.time for p in predictions if p.time]
        if times:
            return sum(times) / len(times)
        else:
            return None

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
        markets=get_manifold_markets(number=3),
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
