import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from evo_researcher.functions.create_embeddings_from_results import create_embeddings_from_results
from evo_researcher.functions.generate_subqueries import generate_subqueries
from evo_researcher.functions.prepare_report import prepare_report, prepare_summary

from evo_researcher.functions.rerank_subqueries import rerank_subqueries
from evo_researcher.functions.scrape_results import scrape_results
from evo_researcher.functions.search import search

def research(
    goal: str,
    openai_key: str,
    tavily_key: str,
    use_summaries: bool,
    model: str = "gpt-4-1106-preview",
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    scrape_content_split_chunk_size: int = 800,
    scrape_content_split_chunk_overlap: int = 225,
    top_k_per_query: int = 8
) -> tuple[str, str]:    
    queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, api_key=openai_key)
    queries = rerank_subqueries(queries=queries, goal=goal, api_key=openai_key)[:subqueries_limit] if initial_subqueries_limit > subqueries_limit else queries

    search_results_with_queries = search(queries, tavily_key, lambda result: not result["url"].startswith("https://www.youtube"))

    if not search_results_with_queries:
        raise ValueError(f"No search results found for the goal {goal}.")

    scrape_args = [result for (_, result) in search_results_with_queries]
    scraped = scrape_results(scrape_args)
    scraped = [result for result in scraped if result.content != ""]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "  "],
        chunk_size=scrape_content_split_chunk_size,
        chunk_overlap=scrape_content_split_chunk_overlap
    )
    collection = create_embeddings_from_results(scraped, text_splitter, api_key=openai_key)

    vector_result_texts: list[str] = []
    url_to_content_deemed_most_useful: dict[str, str] = {}

    for query in queries:
        top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
        vector_result_texts += [result.page_content for result in top_k_per_query_results if result.page_content not in vector_result_texts]

        for x in top_k_per_query_results:
            # `x.metadata["content"]` holds the whole url's web page, so it's ok to overwrite the value of the same url.
            url_to_content_deemed_most_useful[x.metadata["url"]] = x.metadata["content"]

    vector_result_texts = [
        prepare_summary(goal, content, "gpt-3.5-turbo-0125")  # Hard code gpt-3.5-turbo-0125, because it would be very costly with gpt-4.
        for content in url_to_content_deemed_most_useful.values()
    ] if use_summaries else vector_result_texts
    
    chunks = ""
    for chunk in vector_result_texts:
        chunks += "- " + chunk + "\n\n"

    report = prepare_report(goal, vector_result_texts, api_key=openai_key, model=model)

    return (report, chunks)
