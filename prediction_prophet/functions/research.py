import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from prediction_prophet.functions.create_embeddings_from_results import create_embeddings_from_results
from prediction_prophet.functions.generate_subqueries import generate_subqueries
from prediction_prophet.functions.prepare_report import prepare_report, prepare_summary
from prediction_prophet.models.WebScrapeResult import WebScrapeResult
from prediction_prophet.functions.rerank_subqueries import rerank_subqueries
from prediction_prophet.functions.scrape_results import scrape_results
from prediction_prophet.functions.search import search
from pydantic.types import SecretStr

def research(
    goal: str,
    use_summaries: bool,
    model: str = "gpt-4-0125-preview",
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    scrape_content_split_chunk_size: int = 800,
    scrape_content_split_chunk_overlap: int = 225,
    top_k_per_query: int = 8,
    use_tavily_raw_content: bool = False,
    openai_api_key: SecretStr | None = None,
    tavily_api_key: SecretStr | None = None,
    logger: logging.Logger = logging.getLogger()
) -> str:
    logger.info("Started subqueries generation")
    queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, model=model, api_key=openai_api_key)
    
    stringified_queries = '\n- ' + '\n- '.join(queries)
    logger.info(f"Generated subqueries: {stringified_queries}")
    
    logger.info("Started subqueries reranking")
    queries = rerank_subqueries(queries=queries, goal=goal, model=model, api_key=openai_api_key)[:subqueries_limit] if initial_subqueries_limit > subqueries_limit else queries

    stringified_queries = '\n- ' + '\n- '.join(queries)
    logger.info(f"Reranked subqueries. Will use top {subqueries_limit}: {stringified_queries}")
    
    logger.info(f"Started web searching")
    search_results_with_queries = search(
        queries, 
        lambda result: not result.url.startswith("https://www.youtube"),
        tavily_api_key=tavily_api_key
    )

    if not search_results_with_queries:
        raise ValueError(f"No search results found for the goal {goal}.")

    scrape_args = [result for (_, result) in search_results_with_queries]
    websites_to_scrape = set([result.url for result in scrape_args])
    
    stringified_websites = '\n- ' + '\n- '.join(websites_to_scrape)
    logger.info(f"Found the following relevant results: {stringified_websites}")
    
    logger.info(f"Started scraping of web results")
    scraped = scrape_results(scrape_args) if not use_tavily_raw_content else [WebScrapeResult(
        query=result.query,
        url=result.url,
        title=result.title,
        content=result.raw_content,
    ) for result in scrape_args if result.raw_content]
    scraped = [result for result in scraped if result.content != ""]
    
    logger.info(f"Scraped content from {len(scraped)} websites")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "  "],
        chunk_size=scrape_content_split_chunk_size,
        chunk_overlap=scrape_content_split_chunk_overlap
    )
    
    logger.info("Started embeddings creation")
    collection = create_embeddings_from_results(scraped, text_splitter, api_key=openai_api_key)
    logger.info("Embeddings created")

    vector_result_texts: list[str] = []
    url_to_content_deemed_most_useful: dict[str, str] = {}

    stringified_queries = '\n- ' + '\n- '.join(queries)
    logger.info(f"Started similarity searches for: {stringified_queries}")
    for query in queries:
        top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
        vector_result_texts += [result.page_content for result in top_k_per_query_results if result.page_content not in vector_result_texts]

        for x in top_k_per_query_results:
            # `x.metadata["content"]` holds the whole url's web page, so it's ok to overwrite the value of the same url.
            url_to_content_deemed_most_useful[x.metadata["url"]] = x.metadata["content"]
    
    stringified_urls = '\n- ' + '\n- '.join(url_to_content_deemed_most_useful.keys())
    logger.info(f"Found {len(vector_result_texts)} information chunks across the following sites: {stringified_urls}")

    if use_summaries:
        logger.info(f"Started summarizing information")
        vector_result_texts = [
            prepare_summary(
                goal,
                content,
                "gpt-3.5-turbo-0125",
                api_key=openai_api_key,
                trim_content_to_tokens=14_000
            )
            for content in url_to_content_deemed_most_useful.values()
        ]
        logger.info(f"Information summarized")

    logger.info(f"Started preparing report")
    report = prepare_report(goal, vector_result_texts, model=model, api_key=openai_api_key)
    logger.info(f"Report prepared")

    return report