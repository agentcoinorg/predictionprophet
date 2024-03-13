import os
from typing import cast
from evo_researcher.benchmark.agents import _make_prediction
from evo_researcher.functions.evaluate_question import is_predictable as evaluate_if_predictable
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction
)
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from evo_researcher.functions.create_embeddings_from_results import create_embeddings_from_results
from evo_researcher.functions.generate_subqueries import generate_subqueries
from evo_researcher.functions.prepare_report import prepare_report
from evo_researcher.functions.rerank_subqueries import rerank_subqueries
from evo_researcher.functions.scrape_results import scrape_results
from evo_researcher.functions.search import search

def research(
    goal: str,
    openai_api_key: str,
    tavily_api_key: str,
    model: str = "gpt-4-0125-preview",
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    scrape_content_split_chunk_size: int = 800,
    scrape_content_split_chunk_overlap: int = 225,
    top_k_per_query: int = 8
) -> str:
    with st.status("Generating subqueries"):
        queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, model=model, api_key=openai_api_key)
    
        stringified_queries = '\n- ' + '\n- '.join(queries)
        st.write(f"Generated subqueries: {stringified_queries}")
        
    with st.status("Reranking subqueries"):
        queries = rerank_subqueries(queries=queries, goal=goal, model=model, api_key=openai_api_key)[:subqueries_limit] if initial_subqueries_limit > subqueries_limit else queries

        stringified_queries = '\n- ' + '\n- '.join(queries)
        st.write(f"Reranked subqueries. Will use top {subqueries_limit}: {stringified_queries}")
    
    with st.status("Searching the web"):
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
        st.write(f"Found the following relevant results: {stringified_websites}")
    
    with st.status(f"Scraping web results"):
        scraped = scrape_results(scrape_args)
        scraped = [result for result in scraped if result.content != ""]
        
        st.write(f"Scraped content from {len(scraped)} websites")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "  "],
        chunk_size=scrape_content_split_chunk_size,
        chunk_overlap=scrape_content_split_chunk_overlap
    )
    
    with st.status(f"Performing similarity searches"):
        collection = create_embeddings_from_results(scraped, text_splitter, api_key=openai_api_key)
        st.write("Created embeddings")

        vector_result_texts: list[str] = []
        url_to_content_deemed_most_useful: dict[str, str] = {}
        
        for query in queries:
            top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
            vector_result_texts += [result.page_content for result in top_k_per_query_results if result.page_content not in vector_result_texts]

            for x in top_k_per_query_results:
                url_to_content_deemed_most_useful[x.metadata["url"]] = x.metadata["content"]
        
            st.write(f"Similarity searched for: {query}")

        st.write(f"Found {len(vector_result_texts)} relevant information chunks")

    with st.status(f"Preparing report"):
        report = prepare_report(goal, vector_result_texts, model=model, api_key=openai_api_key)
        st.markdown(report)

    return report

tavily_api_key = os.environ.get('TAVILY_API_KEY')

if tavily_api_key == None:
    try:
        tavily_api_key = st.secrets['TAVILY_API_KEY']
    except:
        st.container().error("No Tavily API Key provided")
        st.stop()

st.set_page_config(layout="wide")
st.title("Evo Prophet")
st.write('Ask any question about a future outcome')

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="open_ai_key")

if question := st.chat_input():
    st.chat_message("user").write(question)
    
    with st.chat_message("assistant"):
        st.write(f"I will evaluate the proability of '{question}' happening")
        
        with st.status("Evaluating question") as status:
            (is_predictable, reasoning) = evaluate_if_predictable(question=question, api_key=openai_api_key) 
            if not is_predictable:
                st.container().error(f"The agent thinks this question is not predictable: \n\n{reasoning}")
                status.update(label="Error evaluating question", state="error", expanded=True)
                st.stop()
        
        report = research(
            goal=question,
            subqueries_limit=6,
            top_k_per_query=15,
            openai_api_key=openai_api_key,
            tavily_api_key=tavily_api_key,
        )
                
        with st.status("Making prediction"):
            prediction = _make_prediction(market_question=question, additional_information=report, engine="gpt-4-0125-preview", temperature=0.0, api_key=openai_api_key)

            if prediction.outcome_prediction == None:
                st.container().error("The agent failed to generate a prediction")
                st.stop()
            
            outcome_prediction = cast(OutcomePrediction, prediction.outcome_prediction)
        
            st.write(f"Probability: {outcome_prediction.p_yes * 100}%. Confidence: {outcome_prediction.confidence * 100}%")
            if not prediction:
                st.container().error("No prediction was generated.")
                st.stop()
                
        st.write(f"With {outcome_prediction.confidence * 100}% confidence, I'd say '{question}' has a {outcome_prediction.p_yes * 100}% probability of happening")