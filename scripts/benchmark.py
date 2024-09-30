import os
import typing as t

import typer
from prediction_market_agent_tooling.benchmark.agents import RandomAgent
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import (
    FilterBy,
    MarketType,
    SortBy,
    get_binary_markets,
)
from prediction_market_agent_tooling.tools.is_predictable import (
    is_predictable_binary,
    is_predictable_without_description,
)

from prediction_prophet.benchmark.agents import PredictionProphetAgent
from prediction_prophet.functions.cache import ENABLE_CACHE
from prediction_prophet.functions.research import ResearchCache

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(
    n: int = 10,
    output: str = "./benchmark_report.md",
    reference: MarketType = MarketType.MANIFOLD,
    filter: FilterBy = FilterBy.OPEN,
    sort: SortBy = SortBy.NONE,
    max_workers: int = 1,
    cache_path: t.Optional[str] = "predictions_cache.json",
    only_cached: bool = False,
) -> None:
    """
    Polymarket usually contains higher quality questions,
    but on Manifold, additionally to filtering by MarketFilter.resolved, you can sort by MarketSort.newest.
    """
    markets = get_binary_markets(n, reference, filter_by=filter, sort_by=sort)
    markets_deduplicated = list(({m.question: m for m in markets}.values()))
    if len(markets) != len(markets_deduplicated):
        print(
            f"Warning: Deduplicated markets from {len(markets)} to {len(markets_deduplicated)}."
        )

    skip_questions = [
        "Will China launch a full-scale invasion of Taiwan by the end of September?",
        'Will Manifold "hit 15k DAU (7-day average) by Sept 30th" 2024? (lol)',
    ]
    final_markets: list[AgentMarket] = []
    strict_filters = False
    for m in markets_deduplicated:
        if m.volume < 2000:
            print(f"Skipping `{m.question}` because it has low volume.")
            continue
        if m.question in skip_questions:
            print(f"Skipping `{m.question}` because it is in the skip list.")
            continue

        if strict_filters:
            if not is_predictable_binary(m.question):
                print(
                    f"Skipping `{m.question}` because it seems to not be predictable."
                )
                continue

            if m.description and not is_predictable_without_description(
                m.question, m.description
            ):
                print(
                    f"Skipping `{m.question}` because it seems to not be predictable without the description `{m.description}`."
                )
                continue

        final_markets.append(m)

    print(f"Found {len(final_markets)} markets.")
    for m in final_markets:
        print(f" - {m.question}, {m.volume}")

    # Load research reports from cache
    research_cache_path = "research_cache.json"
    if os.path.exists(research_cache_path):
        research_cache = ResearchCache.load(path=research_cache_path)
    else:
        research_cache = ResearchCache(researches={})

    # Perform research for new markets
    failed_research_markets = []
    for m in final_markets:
        if m.question in research_cache.researches:
            print(f"Skipping research for {m.question}, already in cache.")
            continue
        print(f"Performing research for {m.question}")
        research = PredictionProphetAgent(model="gpt-4o").research(m.question)

        if research is None:
            print(f"Research failed for {m.question}")
            failed_research_markets.append(m.id)
            continue
        research_cache.add_research(m.question, research)

        # Save research cache
        research_cache.save(path=research_cache_path)

    researches = research_cache.researches
    final_markets = [m for m in final_markets if m.id not in failed_research_markets]

    benchmarker = Benchmarker(
        markets=final_markets,
        agents=[
            RandomAgent(agent_name="random", max_workers=max_workers),
            PredictionProphetAgent(
                model="gpt-4o",
                max_workers=max_workers,
                agent_name="prediction_prophet_gpt-4o-no-reasoning",
                include_reasoning=False,
                researches=researches,
            ),
            PredictionProphetAgent(
                model="gpt-4o",
                max_workers=max_workers,
                agent_name="prediction_prophet_gpt-4o-reasoning",
                include_reasoning=True,
                researches=researches,
            ),
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
    APP()
