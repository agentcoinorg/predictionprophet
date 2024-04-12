import json
from prediction_prophet.autonolas.research import clean_completion_json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from prediction_prophet.functions.cache import persistent_inmemory_cache
from pydantic.types import SecretStr
from prediction_market_agent_tooling.tools.utils import secret_str_from_env



# I tried to make it return a JSON, but it didn't work well in combo with asking it to do chain of thought.
QUESTION_EVALUATE_PROMPT = """Main signs about an answerable question (sometimes reffered to as a "market"):
- The question needs to be specific, without use of pronouns.
- The question needs to have a clear future event.
- The question needs to have a clear time frame.
- The answer is probably Google-able, after the event happened.
- The question can not be about itself.
- The question needs to be a yes or no question.

Follow a chain of thought to evaluate if the question is answerable:

First, write the parts of the following question:

"{question}"

Then, write down what is the future event of the question, what it reffers to and when that event will happen if the question contains it.

Then, give your final decision about whether the question is answerable.

Return a JSON object with the following structure:

{{
    "is_predictable": bool,
    "reasoning": string
}}

Output only the JSON object in your response. Do not include any other contents in your response.
"""



@persistent_inmemory_cache
def is_predictable_and_binary(
    question: str,
    engine: str = "gpt-4-0125-preview",
    prompt_template: str = QUESTION_EVALUATE_PROMPT,
    api_key: SecretStr | None = None
) -> tuple[bool, str]:
    """
    Evaluate if the question is actually answerable.
    """
    
    if api_key == None:
        api_key = secret_str_from_env("OPENAI_API_KEY")
    llm = ChatOpenAI(model=engine, temperature=0.0, api_key=api_key.get_secret_value() if api_key else None)

    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    messages = prompt.format_messages(question=question)
    completion = str(llm(messages, max_tokens=256).content)
    response = json.loads(clean_completion_json(completion))

    return (response["is_predictable"], response["reasoning"])
