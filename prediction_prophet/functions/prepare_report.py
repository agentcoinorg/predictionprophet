import os
import tiktoken
import typing as t
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from prediction_prophet.functions.cache import persistent_inmemory_cache
from prediction_prophet.functions.utils import trim_to_n_tokens
from prediction_market_agent_tooling.tools.utils import secret_str_from_env
from pydantic.types import SecretStr


@persistent_inmemory_cache
def prepare_summary(goal: str, content: str, model: str, api_key: SecretStr | None = None, trim_content_to_tokens: t.Optional[int] = None) -> str:
    if api_key == None:
        api_key = secret_str_from_env("OPENAI_API_KEY")
    
    prompt_template = """Write comprehensive summary of the following web content, that provides relevant information to answer the question: '{goal}'.
But cut the fluff and keep it up to the point.
Write in bullet points.
    
Content:

{content}
"""
    content = trim_to_n_tokens(content, trim_content_to_tokens, model) if trim_content_to_tokens else content
    evaluation_prompt = ChatPromptTemplate.from_template(template=prompt_template)

    research_evaluation_chain = (
        evaluation_prompt |
        ChatOpenAI(model=model, api_key=api_key.get_secret_value() if api_key else None) |
        StrOutputParser()
    )

    response: str = research_evaluation_chain.invoke({
        "goal": goal,
        "content": content
    })

    return response


def prepare_report(goal: str, scraped: list[str], model: str, api_key: SecretStr | None = None) -> str:
    if api_key == None:
        api_key = secret_str_from_env("OPENAI_API_KEY")
        
    evaluation_prompt_template = """
    You are a professional researcher. Your goal is to provide a relevant information report
    in order to make an informed prediction for the question: '{goal}'.
    
    Here are the results of relevant web searches:
    
    {search_results}
    
    Prepare a full comprehensive report that provides relevant information to answer the aforementioned question.
    If that is not possible, state why.
    You will structure your report in the following sections:
    
    - Introduction
    - Background
    - Findings and Analysis
    - Conclusion
    - Caveats
    
    Don't limit yourself to just stating each finding; provide a thorough, full and comprehensive analysis of each finding.
    Use markdown syntax. Include as much relevant information as possible and try not to summarize.
    """
    evaluation_prompt = ChatPromptTemplate.from_template(template=evaluation_prompt_template)

    research_evaluation_chain = (
        evaluation_prompt |
        ChatOpenAI(model=model, api_key=api_key.get_secret_value() if api_key else None) |
        StrOutputParser()
    )

    response: str = research_evaluation_chain.invoke({
        "search_results": scraped,
        "goal": goal
    })

    return response