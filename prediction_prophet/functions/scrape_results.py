from prediction_prophet.models.WebScrapeResult import WebScrapeResult
from prediction_prophet.functions.web_search import WebSearchResult
from prediction_prophet.functions.web_scrape import web_scrape
from prediction_prophet.functions.parallelism import par_map


def scrape_results(results: list[WebSearchResult]) -> list[WebScrapeResult]:
    scraped: list[WebScrapeResult] = par_map(results, lambda result: WebScrapeResult(
        query=result.query,
        url=result.url,
        title=result.title,
        content=web_scrape(result.url),
    ))

    return scraped
