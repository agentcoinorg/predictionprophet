import os
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic.types import SecretStr
from prediction_market_agent_tooling.tools.utils import secret_str_from_env
from prediction_market_agent_tooling.gtypes import secretstr_to_v1_secretstr
from langfuse.decorators import langfuse_context
from langchain_core.runnables.config import RunnableConfig


subquery_generation_template = """
You are a professional researcher. Your goal is to prepare a research plan for {query}.

The plan will consist of multiple web searches separated by commas.
Return ONLY the web searches, separated by commas and without quotes.

Limit your searches to {search_limit}.
"""
def generate_subqueries(query: str, limit: int, model: str, api_key: SecretStr | None = None, add_langfuse_callback: bool = False) -> list[str]:
    if limit == 0:
        return [query]

    if api_key == None:
        api_key = secret_str_from_env("OPENAI_API_KEY")
            
    subquery_generation_prompt = ChatPromptTemplate.from_template(template=subquery_generation_template)

    subquery_generation_chain = (
        subquery_generation_prompt |
        ChatOpenAI(model=model, api_key=secretstr_to_v1_secretstr(api_key)) |
        CommaSeparatedListOutputParser()
    )

    config: RunnableConfig = {}
    if add_langfuse_callback:
        config["callbacks"] = [langfuse_context.get_current_langchain_handler()]

    subqueries = subquery_generation_chain.invoke({
        "query": query,
        "search_limit": limit
    })

    return [query] + [subquery.strip('\"') for subquery in subqueries]