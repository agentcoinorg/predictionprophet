from openai import OpenAI
from langchain.prompts import ChatPromptTemplate

rerank_queries_template = """
I will present you with a list of queries to search the web for, for answers to the question: {goal}.

The queries are divided by '---query---'

Evaluate the queries in order that will provide the best data to answer the question. Do not modify the queries.
Return them, in order of relevance, as a comma separated list of strings with no quotes.

Queries: {queries}
"""
def rerank_subqueries(queries: list[str], goal: str, api_key: str, model: str) -> list[str]:
    client = OpenAI(api_key=api_key)
    rerank_results_prompt = (
        ChatPromptTemplate
            .from_template(template=rerank_queries_template)
            .format(goal=goal, queries="\n---query---\n".join(queries))
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional researcher"
            },
            {
                "role": "user",
                "content": rerank_results_prompt,
            }
        ],
        model=model,
        n=1,
        timeout=90,
        stop=None,
    )

    subqueries_str = str(response.choices[0].message.content)

    return [subquery.strip('\"').strip() for subquery in subqueries_str.split(',')]