import pytest
from evo_researcher.functions.evaluate_question import evaluate_question


@pytest.mark.parametrize("question, answerable", [
    ("Will there be an AI language model that surpasses ChatGPT and other OpenAI models before the end of 2024?", True),
    ("Will Vladimir Putin be the President of Russia at the end of 2024?", True),
    ("This market resolves YES when an artificial agent is appointed to the board of directors of a S&P500 company, meanwhile every day I will bet M25 in NO.", False),
    ("Will there be a >0 value liquidity event for me, a former Consensys Software Inc. employee, on my shares of the company?", False),
    ("Will this market have an odd number of traders by the end of 2024?", False),
    ("Did COVID-19 come from a laboratory?", False),
])
def test_evaluate_question(question: str, answerable: bool) -> None:
    eval = evaluate_question(question=question)
    assert (
        eval.is_predictable == answerable, 
        f"Question is not evaluated correctly, see the completion: {eval.is_predictable.completion}"
    )
