import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from evo_researcher.functions.create_embeddings_from_results import create_embeddings_from_results
from evo_researcher.functions.generate_subqueries import generate_subqueries
from evo_researcher.functions.prepare_report import prepare_report

from evo_researcher.functions.rerank_subqueries import rerank_subqueries
from evo_researcher.functions.scrape_results import scrape_results
from evo_researcher.functions.search import search

def research(goal: str, **kwargs) -> tuple[str, str]:
    api_keys: dict[str, str] = kwargs["api_keys"]
    open_ai_key = api_keys.get('openai', os.getenv("OPENAI_API_KEY"))
    tavily_key = api_keys.get('tavily', os.getenv("TAVILY_API_KEY"))
    
    if open_ai_key is None:
        raise ValueError("OpenAI API key is required")
    
    if tavily_key is None:
        raise ValueError("Tavily API key is required")
    
    initial_subqueries_limit = kwargs.get('initial_subqueries_limit', 20)
    subqueries_limit = kwargs.get('subqueries_limit', 4)
    scrape_content_split_chunk_size = kwargs.get('scrape_content_split_chunk_size', 800)
    scrape_content_split_chunk_overlap = kwargs.get('scrape_content_split_chunk_overlap', 225)
    top_k_per_query = kwargs.get('top_k_per_query', 8)

    queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, api_key=open_ai_key)
    queries = rerank_subqueries(queries=queries, goal=goal, api_key=open_ai_key)[:subqueries_limit]

    search_results_with_queries = search(queries, tavily_key, lambda result: not result["url"].startswith("https://www.youtube"))

    scrape_args = [result for (_, result) in search_results_with_queries]
    scraped = scrape_results(scrape_args)
    scraped = [result for result in scraped if result.content != ""]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "  "],
        chunk_size=scrape_content_split_chunk_size,
        chunk_overlap=scrape_content_split_chunk_overlap
    )

    collection = create_embeddings_from_results(scraped, text_splitter, api_key=open_ai_key)

    vector_result_texts: list[str] = []

    for query in queries:
        top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
        vector_result_texts += [f"{result.page_content}" for result in top_k_per_query_results]
    
    chunks = ""
    for chunk in vector_result_texts:
        chunks += "- " + chunk + "\n\n"

    report = prepare_report(goal, vector_result_texts, api_key=open_ai_key)

    return (report, chunks)