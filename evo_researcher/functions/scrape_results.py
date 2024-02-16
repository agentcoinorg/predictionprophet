from evo_researcher.models.WebScrapeResult import WebScrapeResult
from evo_researcher.functions.web_search import WebSearchResult
from evo_researcher.functions.web_scrape import web_scrape

from concurrent.futures import ThreadPoolExecutor, as_completed

def scrape_results(results: list[WebSearchResult]) -> list[WebScrapeResult]:
    scraped: list[WebScrapeResult] = []
    results_by_url = {result.url: result for result in results}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(web_scrape, result.url) for result in results}
        for future in as_completed(futures):
            (scraped_content, url) = future.result()
            websearch_result = results_by_url[url]
            result = WebScrapeResult(
                query=websearch_result.query,
                url=websearch_result.url,
                title=websearch_result.title,
                content=scraped_content
            )
            
            scraped.append(result)

    return scraped