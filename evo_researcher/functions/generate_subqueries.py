import os
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import ChatPromptTemplate

subquery_generation_template = """
You are a professional researcher. Your goal is to prepare a research plan for {query}.

The plan will consist of multiple web searches separated by commas.
Return ONLY the web searches, separated by commas and without quotes.

Limit your searches to {search_limit}.
"""
def generate_subqueries(query: str, limit: int, model: str, api_key: str | None = None) -> list[str]:
    if api_key == None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
            
    subquery_generation_prompt = ChatPromptTemplate.from_template(template=subquery_generation_template)

    subquery_generation_chain = (
        subquery_generation_prompt |
        ChatOpenAI(model=model, api_key=api_key) |
        CommaSeparatedListOutputParser()
    )

    subqueries = subquery_generation_chain.invoke({
        "query": query,
        "search_limit": limit
    })

    return [query] + [subquery.strip('\"') for subquery in subqueries]