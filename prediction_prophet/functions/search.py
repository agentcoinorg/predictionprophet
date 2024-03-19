import requests
import typing as t
from prediction_prophet.functions.web_search import WebSearchResult, web_search
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic.types import SecretStr


def safe_web_search(query: str, max_results: int = 5, tavily_api_key: SecretStr | None = None) -> t.Optional[list[WebSearchResult]]:
    try:
        return web_search(query, max_results, tavily_api_key)
    except requests.exceptions.HTTPError as e:
        print(f"Error in web_search: {e}")
        return None


def search(queries: list[str], filter: t.Callable[[WebSearchResult], bool] = lambda x: True, tavily_api_key: SecretStr | None = None) -> list[tuple[str, WebSearchResult]]:
    maybe_results: list[t.Optional[list[WebSearchResult]]] = []

    # Each result will have a query associated with it
    # We only want to keep the results that are unique
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(safe_web_search, query, 5, tavily_api_key) for query in queries}
        for future in as_completed(futures):
            maybe_results.append(future.result())

    results = [result for result in maybe_results if result is not None]
    if len(results) != len(maybe_results):
        print(f"Warning: {len(maybe_results) - len(results)} queries out of {len(maybe_results)} failed to return results.")

    results_with_queries: list[tuple[str, WebSearchResult]] = []

    for i in range(len(results)):
        for result in results[i]:
            if result.url not in [existing_result.url for (_, existing_result) in results_with_queries]:
                if filter(result):
                  results_with_queries.append((queries[i], result))

    return results_with_queries