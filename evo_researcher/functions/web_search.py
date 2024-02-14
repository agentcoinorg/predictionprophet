import os
import tenacity
from dotenv import load_dotenv
from tavily import TavilyClient

from evo_researcher.models.WebSearchResult import WebSearchResult
from evo_researcher.functions.cache import persistent_inmemory_cache


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1), reraise=True)
@persistent_inmemory_cache
def web_search(query: str, max_results: int = 5) -> list[WebSearchResult]:
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
    )

    transformed_results = [
        WebSearchResult(
            title=result['title'],
            url=result['url'],
            description=result['content'],
            relevancy=result['score'],
            query=query
        )
        for result in response['results']
    ]

    return transformed_results
