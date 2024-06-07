from prediction_prophet.models.WebScrapeResult import WebScrapeResult
from prediction_prophet.functions.web_search import WebSearchResult
from prediction_prophet.functions.web_scrape import web_scrape
from prediction_prophet.functions.parallelism import par_map
from firecrawl import FirecrawlApp


def scrape_results(results: list[WebSearchResult]) -> list[WebScrapeResult]:
    scraped: list[WebScrapeResult] = par_map(results, lambda result: WebScrapeResult(
        query=result.query,
        url=result.url,
        title=result.title,
        content=web_scrape(result.url),
    ))

    return scraped


def scrape_results_firescrap(results: list[WebSearchResult]) -> list[WebScrapeResult]:
    app = FirecrawlApp(api_key="fc-72e8afa08d3046b2b458a33b9e671839")
    scraped: list[WebScrapeResult] = par_map(results, lambda result: WebScrapeResult(
        query=result.query,
        url=result.url,
        title=result.title,
        content=app.scrape_url(result.url)["markdown"],
    ))

    return scraped
