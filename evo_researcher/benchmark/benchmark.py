import argparse
import concurrent.futures
import numpy as np
import os
import pandas as pd
import time
import typing as t

from langchain_community.callbacks import get_openai_callback

from evo_researcher.benchmark.agents import (
    AbstractBenchmarkedAgent,
    EvoAgent,
    OlasAgent,
)
from evo_researcher.benchmark.utils import (
    Market,
    MarketSource,
    Prediction,
    PredictionsCache,
    get_llm_api_call_cost,
    get_markets,
)


class Benchmarker:
    def __init__(
        self,
        markets: t.List[Market],
        agents: t.List[AbstractBenchmarkedAgent],
        metric_fns: t.Dict[str, t.Callable] = {},
        cache_path: t.Optional[str] = None,
    ):
        self.markets: t.List[Market] = markets
        self.registered_agents: t.List[AbstractBenchmarkedAgent] = agents

        # Predictions
        self.cache_path = cache_path
        if self.cache_path and os.path.exists(self.cache_path):
            self.predictions = PredictionsCache.load(path=self.cache_path)
        else:
            self.predictions = PredictionsCache(predictions={})

        # Metrics
        self.metric_fns = metric_fns
        predefined_metric_fns = {
            "MSE for `p_yes`": self._compute_mse,
            "Mean confidence": self._compute_mean_confidence,
            "% within +-0.05": lambda predictions, markets: self._compute_percentage_within_range(
                predictions, markets, tolerance=0.05
            ),
            "% within +-0.1": lambda predictions, markets: self._compute_percentage_within_range(
                predictions, markets, tolerance=0.1
            ),
            "% within +-0.2": lambda predictions, markets: self._compute_percentage_within_range(
                predictions, markets, tolerance=0.2
            ),
            "% correct outcome": self._compute_correct_outcome_percentage,
            "confidence/p_yes error correlation": self._compute_confidence_p_yes_error_correlation,
            "Mean info_utility": self._compute_mean_info_utility,
            "Mean cost ($)": self._compute_mean_cost,
            "Mean time (s)": self._compute_mean_time,
        }
        self.metric_fns.update(predefined_metric_fns)

    def add_prediction(
        self,
        agent: AbstractBenchmarkedAgent,
        prediction: Prediction,
        market_question: str,
    ):
        self.predictions.add_prediction(
            agent_name=agent.agent_name,
            question=market_question,
            prediction=prediction,
        )

    def get_prediction(self, agent_name: str, question: str) -> Prediction:
        return self.predictions.get_prediction(agent_name=agent_name, question=question)

    def run_agents(self):
        for agent in self.registered_agents:
            # Filter out cached predictions
            markets_to_run = [
                m
                for m in self.markets
                if not self.predictions.has_market(
                    agent_name=agent.agent_name, question=m.question
                )
            ]

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
                return market.question, prediction

            # Run agents in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=agent.max_workers
            ) as executor:
                future_to_market = {
                    executor.submit(get_prediction_result, market): market
                    for market in markets_to_run
                }
                for future in concurrent.futures.as_completed(future_to_market):
                    market_question, prediction = future.result()
                    self.add_prediction(
                        agent=agent,
                        prediction=prediction,
                        market_question=market_question,
                    )
                if self.cache_path:
                    self.predictions.save(self.cache_path)

    def _compute_mse(self, predictions: t.List[Prediction], markets: t.List[Market]):
        mse = sum([(p.p_yes - m.p_yes) ** 2 for p, m in zip(predictions, markets)])
        mse /= len(predictions)
        return mse

    def _compute_mean_confidence(
        self, predictions: t.List[Prediction], markets: t.List[Market]
    ):
        mean_confidence = sum([p.confidence for p in predictions]) / len(predictions)
        return mean_confidence

    def _compute_mean_info_utility(
        self, predictions: t.List[Prediction], markets: t.List[Market]
    ):
        mean_info_utility = sum([p.info_utility for p in predictions]) / len(
            predictions
        )
        return mean_info_utility

    def _compute_percentage_within_range(
        self,
        predictions: t.List[Prediction],
        markets: t.List[Market],
        tolerance: float = 0.05,
    ):
        within_range_count = 0
        for p, m in zip(predictions, markets):
            if abs(p.p_yes - m.p_yes) <= tolerance:
                within_range_count += 1

        return (100 * within_range_count) / len(predictions)

    def _compute_correct_outcome_percentage(
        self, predictions: t.List[Prediction], markets: t.List[Market]
    ):
        correct_outcome_count = 0
        for p, m in zip(predictions, markets):
            if (p.p_yes > 0.5 and m.p_yes > 0.5) or (p.p_yes < 0.5 and m.p_yes < 0.5):
                correct_outcome_count += 1

        return (100 * correct_outcome_count) / len(predictions)

    def _compute_confidence_p_yes_error_correlation(
        self, predictions: t.List[Prediction], markets: t.List[Market]
    ):
        p_yes_errors = [abs(p.p_yes - m.p_yes) for p, m in zip(predictions, markets)]
        confidences = [p.confidence for p in predictions]
        return np.corrcoef(confidences, p_yes_errors)[0, 1]

    def _compute_mean_cost(
        self, predictions: t.List[Prediction], markets: t.List[Market]
    ):
        # Note: costs are optional
        costs = [p.cost for p in predictions if p.cost]
        if costs:
            return sum(costs) / len(costs)
        else:
            return None

    def _compute_mean_time(
        self, predictions: t.List[Prediction], markets: t.List[Market]
    ):
        # Note: times are optional
        times = [p.time for p in predictions if p.time]
        if times:
            return sum(times) / len(times)
        else:
            return None

    def compute_metrics(self) -> t.Dict[str, t.List[t.Any]]:
        metrics = {}
        agents = [a.agent_name for a in self.registered_agents]
        metrics["Agents"] = agents

        for name, fn in self.metric_fns.items():
            metrics[name] = []
            for agent in agents:
                ordered_predictions = [
                    self.get_prediction(question=market.question, agent_name=agent)
                    for market in self.markets
                ]
                metrics[name].append(
                    fn(predictions=ordered_predictions, markets=self.markets)
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

        for agent in [a.agent_name for a in self.registered_agents]:
            markets_summary[f"{agent} p_yes"] = [
                self.get_prediction(agent_name=agent, question=q).p_yes
                for q in market_questions
            ]
        markets_summary[f"reference p_yes"] = [m.p_yes for m in self.markets]
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
    args.add_argument(
        "--reference",
        type=str,
        choices=[ms.value for ms in MarketSource],
        default="manifold",
    )
    args = args.parse_args()

    benchmarker = Benchmarker(
        markets=get_markets(number=3, source=MarketSource(args.reference)),
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
