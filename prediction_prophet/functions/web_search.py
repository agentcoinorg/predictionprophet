import tenacity
import typing as t
from tavily import TavilyClient
from pydantic.types import SecretStr

from prediction_market_agent_tooling.tools.utils import secret_str_from_env
from prediction_prophet.models.WebSearchResult import WebSearchResult
from prediction_prophet.functions.cache import persistent_inmemory_cache



def web_search(query: str, max_results: int = 5, tavily_api_key: SecretStr | None = None) -> list[WebSearchResult]:
    response = _web_search(query=query, max_results=max_results, tavily_api_key=tavily_api_key)

    transformed_results = [
        WebSearchResult(
            title=result['title'],
            url=result['url'],
            description=result['content'],
            raw_content=result['raw_content'],
            relevancy=result['score'],
            query=query
        )
        for result in response['results']
    ]

    return transformed_results


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1), reraise=True)
@persistent_inmemory_cache
def _web_search(query: str, max_results: int = 5, tavily_api_key: SecretStr | None = None) -> dict[str, t.Any]:
    if tavily_api_key == None:
        tavily_api_key = secret_str_from_env("TAVILY_API_KEY")
    
    tavily = TavilyClient(api_key=tavily_api_key.get_secret_value() if tavily_api_key else None)
    response: dict[str, t.Any] = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=True,
    )

    return response

