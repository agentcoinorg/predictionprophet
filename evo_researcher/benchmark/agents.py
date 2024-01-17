import dotenv
import json
import os
import typing as t

from evo_researcher.functions.research import research as research_evo
from evo_researcher.autonolas.research import (
    make_prediction,
    research as research_autonolas,
)
from evo_researcher.benchmark.utils import PredictionResult


def parse_prediction_str(prediction: str) -> PredictionResult:
    """
    Parse a prediction string of the form:

    ```json
    {
        "p_yes": 0.6,
        "p_no": 0.4,
        "confidence": 0.8,
        "info_utility": 0.9
    }
    ```

    into a PredictionResult object
    """
    start_index = prediction.find("{")
    end_index = prediction.rfind("}")
    prediction = prediction[start_index : end_index + 1]
    prediction_json = json.loads(prediction)
    return PredictionResult(
        p_yes=prediction_json["p_yes"],
        confidence=prediction_json["confidence"],
        info_utility=prediction_json["info_utility"],
    )


def _make_prediction(
    market_question: str, additional_information: str
) -> PredictionResult:
    prediction: str = make_prediction(
        prompt=market_question, additional_information=additional_information
    )
    prediction: PredictionResult = parse_prediction_str(prediction)
    return prediction


class AbstractBenchmarkedAgent:
    def __init__(self, agent_name: str, max_workers: t.Optional[int] = None):
        self.agent_name = agent_name
        self.max_workers = max_workers  # Limit the number of workers that can run this worker in parallel threads

    def research_and_predict(self, market_question: str) -> PredictionResult:
        raise NotImplementedError


class OlasAgent(AbstractBenchmarkedAgent):
    def __init__(self, model: str):
        super().__init__(agent_name="olas")
        self.model = model

    def research_and_predict(self, market_question: str) -> PredictionResult:
        report = research_autonolas(
            prompt=market_question,
            engine=self.model,
        )
        return _make_prediction(
            market_question=market_question, additional_information=report
        )


class EvoAgent(AbstractBenchmarkedAgent):
    def __init__(self, model: str):
        super().__init__(agent_name="evo")
        self.model = model

    def research_and_predict(self, market_question: str) -> PredictionResult:
        dotenv.load_dotenv()
        open_ai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        report, _ = research_evo(
            goal=market_question,
            openai_key=open_ai_key,
            tavily_key=tavily_key,
            model=self.model,
        )
        return _make_prediction(
            market_question=market_question, additional_information=report
        )
