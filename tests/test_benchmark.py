import json

import prediction_market_agent_tooling.benchmark.benchmark as bm
from evo_researcher.autonolas.research import clean_completion_json
from evo_researcher.benchmark.agents import completion_prediction_json_to_pydantic_model


def test_parse_result_str_to_json() -> None:
    prediction_str = (
        "```json\n"
        "{\n"
        '  "decision": "y",\n'
        '  "decision_token_prob": 0.6,\n'
        '  "p_yes": 0.6,\n'
        '  "p_no": 0.4,\n'
        '  "confidence": 0.8,\n'
        '  "info_utility": 0.9\n'
        "}\n"
        "```\n"
    )
    prediction: bm.Prediction = completion_prediction_json_to_pydantic_model(
        json.loads(clean_completion_json(prediction_str))
    )
    assert prediction.outcome_prediction is not None
    assert prediction.outcome_prediction.p_yes == 0.6
    assert prediction.outcome_prediction.confidence == 0.8
    assert prediction.outcome_prediction.info_utility == 0.9
