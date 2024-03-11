import typing as t

import typer
from prediction_market_agent_tooling.benchmark.agents import FixedAgent, RandomAgent
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.benchmark.utils import MarketSource, get_markets, MarketFilter, MarketSort

from evo_researcher.autonolas.research import EmbeddingModel
from evo_researcher.benchmark.agents import EvoAgent, OlasAgent, QuestionOnlyAgent
from evo_researcher.functions.cache import ENABLE_CACHE


def main(
    n: int = 10,
    output: str = "./benchmark_report.md",
    reference: MarketSource = MarketSource.MANIFOLD,
    filter: MarketFilter = MarketFilter.open,
    sort: MarketSort | None = None,
    max_workers: int = 1,
    cache_path: t.Optional[str] = "predictions_cache.json",
    only_cached: bool = False,
) -> None:
    """
    Polymarket usually contains higher quality questions, 
    but on Manifold, additionally to filtering by MarketFilter.resolved, you can sort by MarketSort.newest.
    """
    markets = get_markets(number=n, source=reference, filter_=filter, sort=sort)
    markets_deduplicated = list(({m.question: m for m in markets}.values()))
    if len(markets) != len(markets_deduplicated):
        print(
            f"Warning: Deduplicated markets from {len(markets)} to {len(markets_deduplicated)}."
        )

    print(f"Found {len(markets_deduplicated)} markets.")

    benchmarker = Benchmarker(
        markets=markets_deduplicated,
        agents=[
            RandomAgent(agent_name="random", max_workers=max_workers),
            FixedAgent(
                fixed_answer=False, agent_name="fixed-no", max_workers=max_workers
            ),
            FixedAgent(
                fixed_answer=True, agent_name="fixed-yes", max_workers=max_workers
            ),
            QuestionOnlyAgent(
                model="gpt-3.5-turbo-0125",
                agent_name="question-only_gpt-3.5-turbo-0125",
                max_workers=max_workers,
            ),
            OlasAgent(
                model="gpt-3.5-turbo-0125",
                max_workers=max_workers,
                agent_name="olas_gpt-3.5-turbo-0125",
            ),
            OlasAgent(
                model="gpt-3.5-turbo-0125",
                max_workers=max_workers,
                agent_name="olas_gpt-3.5-turbo-0125_openai-embeddings",
                embedding_model=EmbeddingModel.openai,
            ),
            EvoAgent(
                model="gpt-3.5-turbo-0125",
                max_workers=max_workers,
                agent_name="evo_gpt-3.5-turbo-0125_summary",
                use_summaries=True,
            ),
            EvoAgent(
                model="gpt-3.5-turbo-0125",
                max_workers=max_workers,
                agent_name="evo_gpt-3.5-turbo-0125",
            ),
            EvoAgent(
                model="gpt-3.5-turbo-0125",
                max_workers=max_workers,
                agent_name="evo_gpt-3.5-turbo-0125_summary_tavilyrawcontent",
                use_summaries=True,
                use_tavily_raw_content=True,
            ),
            EvoAgent(
                model="gpt-3.5-turbo-0125",
                max_workers=max_workers,
                agent_name="evo_gpt-3.5-turbo-0125_tavilyrawcontent",
                use_tavily_raw_content=True,
            ),
            # EvoAgent(model="gpt-4-0125-preview", max_workers=max_workers, agent_name="evo_gpt-4-0125-preview"),  # Too expensive to be enabled by default.
        ],
        cache_path=cache_path,
        only_cached=only_cached,
    )

    benchmarker.run_agents(
        enable_timing=not ENABLE_CACHE
    )  # Caching of search etc. can distort timings
    md = benchmarker.generate_markdown_report()

    with open(output, "w") as f:
        print(f"Writing benchmark report to: {output}")
        f.write(md)


if __name__ == "__main__":
    typer.run(main)
