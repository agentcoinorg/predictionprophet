from langchain.prompts import ChatPromptTemplate
from openai import OpenAI

def prepare_report(goal: str, scraped: list[str], model: str, api_key: str):
    evaluation_prompt_template = """
    Your goal is to provide a relevant information report
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
    evaluation_prompt = (
        ChatPromptTemplate
            .from_template(template=evaluation_prompt_template)
            .format(search_results=scraped, goal=goal)
    )
    
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional researcher"
            },
            {
                "role": "user",
                "content": evaluation_prompt,
            }
        ],
        model=model,
        n=1,
        timeout=90,
        stop=None,
    )

    return str(response.choices[0].message.content)