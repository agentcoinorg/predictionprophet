import os
from dotenv import load_dotenv
from tavily import TavilyClient

from evo_researcher.models.WebSearchResult import WebSearchResult

def web_search(query: str, api_key: str, max_results=5, max_retries=3) -> list[WebSearchResult]:
    tavily = TavilyClient(api_key=api_key)
    print(f"-- Searching the web for {query} --")
    for i in range(max_retries):
        try:
            response = tavily.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            )
            break
        except Exception as e:
            if i == max_retries - 1:
                raise e
            print(f"Error searching the web for {query}, retrying...")
            continue

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