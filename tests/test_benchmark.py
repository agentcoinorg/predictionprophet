import pytest

import evo_researcher.benchmark.benchmark as bm


@pytest.fixture
def dummy_agent():
    class DummyAgent(bm.AbstractBenchmarkedAgent):
        def __init__(self):
            super().__init__(agent_name="dummy")

        def research_and_predict(self, market_question: str) -> bm.PredictionResult:
            return bm.PredictionResult(p_yes=0.6, confidence=0.8, info_utility=0.9)

    return DummyAgent()


def test_agent_prediction(dummy_agent):
    prediction = dummy_agent.research_and_predict(market_question="Will GNO go up?")
    assert prediction.p_yes == 0.6
    assert prediction.confidence == 0.8
    assert prediction.info_utility == 0.9


def test_benchmark_run(dummy_agent):
    benchmarker = bm.Benchmarker(
        markets=bm.get_manifold_markets(number=1),
        agents=[dummy_agent],
    )
    benchmarker.run_agents()
    benchmarker.generate_markdown_report()


def test_parse_result_str_to_json():
    prediction = (
        "```json\n"
        "{\n"
        '  "p_yes": 0.6,\n'
        '  "p_no": 0.4,\n'
        '  "confidence": 0.8,\n'
        '  "info_utility": 0.9\n'
        "}\n"
        "```\n"
    )
    prediction: bm.PredictionResult = bm.parse_prediction_str(prediction)
    assert prediction.p_yes == 0.6
    assert prediction.confidence == 0.8
    assert prediction.info_utility == 0.9
