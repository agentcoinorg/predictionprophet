from pydantic.types import SecretStr

from prediction_prophet.models.WebSearchResult import WebSearchResult
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.tools.tavily_storage.tavily_models import TavilyStorage
from prediction_market_agent_tooling.tools.tavily_storage.tavily_storage import tavily_search



def web_search(query: str, max_results: int = 5, tavily_api_key: SecretStr | None = None, tavily_storage: TavilyStorage | None = None) -> list[WebSearchResult]:
    response = tavily_search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=True,
        tavily_storage=tavily_storage,
        api_keys=APIKeys(TAVILY_API_KEY=tavily_api_key) if tavily_api_key else None,
    )

    transformed_results = [
        WebSearchResult(
            title=result.title,
            url=result.url,
            description=result.content,
            raw_content=result.raw_content,
            relevancy=result.score,
            query=query,
        )
        for result in response.results
    ]

    return transformed_results
