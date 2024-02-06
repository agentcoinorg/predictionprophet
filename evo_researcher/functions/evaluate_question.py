from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from evo_researcher.functions.cache import persisted_inmemory_cache


# I tried to make it return a JSON, but it didn't work well in combo with asking it to do chain of thought.
QUESTION_EVALUATE_PROMPT = """Main signs about an answerable question (sometimes reffered to as a "market"):
- The question needs to be specific, without use of pronouns.
- The question needs to have a clear future event.
- The question needs to have a clear time frame.
- The answer is probably Google-able, after the event happened.
- The question can not be about itself.

Follow a chain of thought to evaluate if the question is answerable:

First, write the parts of the following question:

"{question}"

Then, write down what is the future event of the question, what it reffers to and when that event will happen if the question contains it.

Then, give your final decision, write either "yes" or "no" about whether the question is answerable.
"""


class IsPredictable(BaseModel):
    answer: bool
    prompt: str
    completion: str


class EvalautedQuestion(BaseModel):
    question: str
    is_predictable: IsPredictable


@persisted_inmemory_cache
def evaluate_question(
    question: str,
    engine: str = "gpt-4-1106-preview",
    prompt_template: str = QUESTION_EVALUATE_PROMPT,
) -> EvalautedQuestion:
    """
    Evaluate if the question is actually answerable.
    """
    llm = ChatOpenAI(model=engine, temperature=0.0)

    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    messages = prompt.format_messages(question=question)
    completion = llm(messages, max_tokens=256).content

    if "yes" in completion.lower():
        is_predictable = True
    elif "no" in completion.lower():
        is_predictable = False
    else:
        raise ValueError(f"Error in evaluate_question for `{question}`: {completion}")

    return EvalautedQuestion(
        question=question, 
        is_predictable=IsPredictable(
            answer=is_predictable,
            prompt=QUESTION_EVALUATE_PROMPT,
            completion=completion,
        )
    )
