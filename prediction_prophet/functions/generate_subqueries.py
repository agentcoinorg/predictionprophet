import os
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic.types import SecretStr
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import secretstr_to_v1_secretstr
from prediction_market_agent_tooling.tools.langfuse_ import get_langfuse_langchain_config, observe


subquery_generation_template = """
You are a professional researcher. Your goal is to prepare a research plan for {query}.

The plan will consist of multiple web searches separated by commas.
Return ONLY the web searches, separated by commas and without quotes.

Limit your searches to {search_limit}.
"""
@observe()
def generate_subqueries(query: str, limit: int, model: str, temperature: float, api_key: SecretStr | None = None) -> list[str]:
    if limit == 0:
        return [query]

    if api_key == None:
        api_key = APIKeys().openai_api_key
            
    subquery_generation_prompt = ChatPromptTemplate.from_template(template=subquery_generation_template)

    subquery_generation_chain = (
        subquery_generation_prompt |
        ChatOpenAI(model=model, temperature=temperature, api_key=secretstr_to_v1_secretstr(api_key)) |
        CommaSeparatedListOutputParser()
    )

    subqueries = subquery_generation_chain.invoke({
        "query": query,
        "search_limit": limit
    }, config=get_langfuse_langchain_config())

    return [query] + [subquery.strip('\"') for subquery in subqueries]