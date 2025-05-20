import json

import prediction_market_agent_tooling.benchmark.benchmark as bm
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer

from prediction_prophet.autonolas.research import clean_completion_json


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
    prediction: ProbabilisticAnswer = ProbabilisticAnswer.model_validate(
        json.loads(clean_completion_json(prediction_str))
    )
    assert prediction is not None
    assert prediction.p_yes == 0.6
    assert prediction.confidence == 0.8

