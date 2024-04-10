import typing as t

from prediction_market_agent_tooling.benchmark.agents import (
    AbstractBenchmarkedAgent,
)
from prediction_market_agent_tooling.benchmark.utils import (
    Prediction,
)
from datetime import datetime
from prediction_prophet.autonolas.research import EmbeddingModel
from prediction_prophet.autonolas.research import make_prediction, get_urls_from_queries
from prediction_prophet.autonolas.research import research as research_autonolas
from prediction_prophet.functions.evaluate_question import is_predictable
from prediction_prophet.functions.rephrase_question import rephrase_question
from prediction_prophet.functions.research import research as prophet_research
from prediction_prophet.functions.search import search
from prediction_prophet.functions.utils import url_is_older_than
from prediction_prophet.models.WebSearchResult import WebSearchResult
from unittest.mock import patch
from prediction_prophet.functions.search import search
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction,
    Prediction,
)
from pydantic.types import SecretStr
from prediction_prophet.autonolas.research import Prediction as LLMCompletionPredictionDict

def _make_prediction(
    market_question: str,
    additional_information: str,
    engine: str,
    temperature: float,
    api_key: SecretStr | None = None
) -> Prediction:
    """
    We prompt model to output a simple flat JSON and convert it to a more structured pydantic model here.
    """
    prediction = make_prediction(
        prompt=market_question,
        additional_information=additional_information,
        engine=engine,
        temperature=temperature,
        api_key=api_key
    )
    return completion_prediction_json_to_pydantic_model(
        prediction
    )


def completion_prediction_json_to_pydantic_model(
    completion_prediction: LLMCompletionPredictionDict,
) -> Prediction:
    return Prediction(
        outcome_prediction=OutcomePrediction(
            p_yes=completion_prediction["p_yes"],
            confidence=completion_prediction["confidence"],
            info_utility=completion_prediction["info_utility"],
        ),
    )


class QuestionOnlyAgent(AbstractBenchmarkedAgent):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        agent_name: str = "question-only",
        max_workers: t.Optional[int] = None,
    ):
        super().__init__(agent_name=agent_name, max_workers=max_workers)
        self.model = model
        self.temperature = temperature

    def predict(
        self, market_question: str
    ) -> Prediction:
        try:
            return _make_prediction(
                market_question=market_question,
                additional_information="",
                engine=self.model,
                temperature=self.temperature,
            )
        except ValueError as e:
            print(f"Error in QuestionOnlyAgent's predict: {e}")
            return Prediction()
        
    def predict_restricted(
        self, market_question: str, time_restriction_up_to: datetime
    ) -> Prediction:
        return self.predict(market_question)


class OlasAgent(AbstractBenchmarkedAgent):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        agent_name: str = "olas",
        max_workers: t.Optional[int] = None,
        embedding_model: EmbeddingModel = EmbeddingModel.spacy,
    ):
        super().__init__(agent_name=agent_name, max_workers=max_workers)
        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model

    def is_predictable(self, market_question: str) -> bool:
        (result, _) = is_predictable(question=market_question)
        return result

    def is_predictable_restricted(self, market_question: str, time_restriction_up_to: datetime) -> bool:
        (result, _) = is_predictable(question=market_question)
        return result
    
    def research(self, market_question: str) -> str:
        return research_autonolas(
            prompt=market_question,
            engine=self.model,
            embedding_model=self.embedding_model,
        )

    def predict(self, market_question: str) -> Prediction:
        try:
            researched = self.research(market_question=market_question)
            return _make_prediction(
                market_question=market_question,
                additional_information=researched,
                engine=self.model,
                temperature=self.temperature,
            )
        except ValueError as e:
            print(f"Error in OlasAgent's predict: {e}")
            return Prediction()

    def predict_restricted(
        self, market_question: str, time_restriction_up_to: datetime
    ) -> Prediction:
        def side_effect(*args: t.Any, **kwargs: t.Any) -> list[str]:
            results: list[str] = get_urls_from_queries(*args, **kwargs)
            results_filtered = [
                url for url in results
                if url_is_older_than(url, time_restriction_up_to)
            ]
            return results_filtered
    
        with patch('prediction_prophet.autonolas.research.get_urls_from_queries', side_effect=side_effect, autospec=True):
            return self.predict(market_question)


class PredictionProphetAgent(AbstractBenchmarkedAgent):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        agent_name: str = "prediction_prophet",
        use_summaries: bool = False,
        use_tavily_raw_content: bool = False,
        max_workers: t.Optional[int] = None,
    ):
        super().__init__(agent_name=agent_name, max_workers=max_workers)
        self.model = model
        self.temperature = temperature
        self.use_summaries = use_summaries
        self.use_tavily_raw_content = use_tavily_raw_content

    def is_predictable(self, market_question: str) -> bool:
        (result, _) = is_predictable(question=market_question)
        return result

    def is_predictable_restricted(self, market_question: str, time_restriction_up_to: datetime) -> bool:
        (result, _) = is_predictable(question=market_question)
        return result
    
    def research(self, market_question: str) -> str:
        return prophet_research(
            goal=market_question,
            model=self.model,
            use_summaries=self.use_summaries,
            use_tavily_raw_content=self.use_tavily_raw_content,
        )
    
    def predict(self, market_question: str) -> Prediction:
        try:
            report = self.research(market_question)
            return _make_prediction(
                market_question=market_question,
                additional_information=report,
                engine=self.model,
                temperature=self.temperature,
            )
        except ValueError as e:
            print(f"Error in PredictionProphet's predict: {e}")
            return Prediction()

    def predict_restricted(
        self, market_question: str, time_restriction_up_to: datetime
    ) -> Prediction:
        def side_effect(*args: t.Any, **kwargs: t.Any) -> list[tuple[str, WebSearchResult]]:
            results: list[tuple[str, WebSearchResult]] = search(*args, **kwargs)
            results_filtered = [
                r for r in results
                if url_is_older_than(r[1].url, time_restriction_up_to)
            ]
            return results_filtered
    
        with patch('prediction_prophet.functions.research.search', side_effect=side_effect, autospec=True):
            return self.predict(market_question)

class RephrasingOlasAgent(OlasAgent):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        agent_name: str = "reph-olas",
        max_workers: t.Optional[int] = None,
        embedding_model: EmbeddingModel = EmbeddingModel.spacy,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            embedding_model=embedding_model,
            agent_name=agent_name,
            max_workers=max_workers,
        )

    def research(self, market_question: str) -> str:
        questions = rephrase_question(question=market_question)

        report_original = super().research(market_question=questions.original_question)
        report_negated = super().research(market_question=questions.negated_question)
        report_universal = super().research(market_question=questions.open_ended_question)

        report_concat = "\n\n---\n\n".join(
            [
                f"### {r_name}\n\n{r}"
                for r_name, r in [
                    ("Research based on the question", report_original),
                    ("Research based on the negated question", report_negated),
                    ("Research based on the universal search query", report_universal),
                ]
                if r is not None
            ]
        )

        return report_concat


AGENTS = [
    OlasAgent,
    RephrasingOlasAgent,
    PredictionProphetAgent,
    QuestionOnlyAgent,
]
