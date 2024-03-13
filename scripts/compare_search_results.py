import json
import typer
import typing as t
import itertools as it
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
from collections import defaultdict
from prediction_market_agent_tooling.benchmark.utils import get_markets, MarketSource
from evo_prophet.functions.web_search import web_search
from evo_prophet.autonolas.research import safe_get_urls_from_query

ENGINES: dict[str, t.Callable[[str, int], list[str]]] = {
    "tavily": lambda q, limit: [x.url for x in web_search(q, max_results=limit)],
    "google": lambda q, limit: safe_get_urls_from_query(q, num=limit)
}


def main(
    n: int = 10,
    source: MarketSource = MarketSource.MANIFOLD,
    max_results_per_query: int = 10,
    output: str = "results",
) -> None:
    markets = get_markets(n, source)

    summary = defaultdict(list)
    links: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for market in tqdm(markets):
        results = {
            engine: func(market.question, max_results_per_query)
            for engine, func in ENGINES.items()
        }

        summary["question"].append(market.question)

        for a, b in it.combinations(ENGINES, 2):
            if a != b:
                summary[f"common links in {a} and {b}"].append(len(set(results[a]) & set(results[b])) / max(len(results[a]), len(results[b])))
                summary[f"common domains in {a} and {b}"].append(len(set(map(extract_domain_from_url, results[a])) & set(map(extract_domain_from_url, results[b]))) / max(len(results[a]), len(results[b])))

        for engine, urls in results.items():
            links[market.question][engine] = urls
    
    pd.DataFrame(summary).to_csv(f"{output}.csv", index=False)
    with open(f"{output}.json", "w") as f:
        json.dump(links, f, indent=4)


def extract_domain_from_url(url: str) -> str:
    if not url.startswith("http"):
        url = f"http://{url}"
    return urlparse(url).netloc.replace("www.", "")


if __name__ == "__main__":
    typer.run(main)
