from langchain.prompts import ChatPromptTemplate
from openai import OpenAI

subquery_generation_template = """
Your goal is to prepare a research plan for {query}.

The plan will consist of multiple web searches separated by commas.
Return ONLY the web searches, separated by commas and without quotes.

Limit your searches to {search_limit}.
"""
def generate_subqueries(query: str, limit: int, api_key: str, model: str) -> list[str]:
    client = OpenAI(api_key=api_key)
    subquery_generation_prompt = (
        ChatPromptTemplate
            .from_template(template=subquery_generation_template)
            .format(query=query, search_limit=limit)
    )
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional researcher"
            },
            {
                "role": "user",
                "content": subquery_generation_prompt,
            }
        ],
        model=model,
        n=1,
        timeout=90,
        stop=None,
    )

    subqueries_str = str(response.choices[0].message.content)

    return [query] + [subquery.strip('\"').strip() for subquery in subqueries_str.split(',')]