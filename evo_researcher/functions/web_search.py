import os
from dotenv import load_dotenv
from tavily import TavilyClient

from evo_researcher.models.WebSearchResult import WebSearchResult

def web_search(query: str, api_key: str, max_results=5) -> list[WebSearchResult]:
    tavily = TavilyClient(api_key=api_key)
    print(f"-- Searching the web for {query} --")
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