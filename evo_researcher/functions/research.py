import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from evo_researcher.functions.create_embeddings_from_results import create_embeddings_from_results
from evo_researcher.functions.generate_subqueries import generate_subqueries
from evo_researcher.functions.prepare_report import prepare_report

from evo_researcher.functions.rerank_subqueries import rerank_subqueries
from evo_researcher.functions.scrape_results import scrape_results
from evo_researcher.functions.search import search

def research(
    goal: str,
    openai_key: str,
    tavily_key: str,
    model: str = "gpt-4-1106-preview",
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    scrape_content_split_chunk_size: int = 800,
    scrape_content_split_chunk_overlap: int = 225,
    top_k_per_query: int = 8
) -> tuple[str, str]:    
    queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, api_key=openai_key)
    queries = rerank_subqueries(queries=queries, goal=goal, api_key=openai_key)[:subqueries_limit]

    search_results_with_queries = search(queries, tavily_key, lambda result: not result["url"].startswith("https://www.youtube"))

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

    for query in queries:
        top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
        vector_result_texts += [f"{result.page_content}" for result in top_k_per_query_results]
    
    chunks = ""
    for chunk in vector_result_texts:
        chunks += "- " + chunk + "\n\n"

    report = prepare_report(goal, vector_result_texts, api_key=openai_key, model=model)

    return (report, chunks)