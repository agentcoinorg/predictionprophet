import typing as t

from prediction_market_agent_tooling.benchmark.agents import (
    AbstractBenchmarkedAgent,
    FixedAgent,
    RandomAgent,
)
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction,
    Prediction,
)
from datetime import datetime
from evo_researcher.autonolas.research import EmbeddingModel
from evo_researcher.autonolas.research import Prediction as LLMCompletionPredictionDict
from evo_researcher.autonolas.research import make_prediction
from evo_researcher.autonolas.research import research as research_autonolas
from evo_researcher.functions.evaluate_question import is_predictable
from evo_researcher.functions.rephrase_question import rephrase_question
from evo_researcher.functions.research import research as research_evo


def _make_prediction(
    market_question: str,
    additional_information: str,
    engine: str,
    temperature: float,
) -> Prediction:
    """
    We prompt model to output a simple flat JSON and convert it to a more structured pydantic model here.
    """
    prediction = make_prediction(
        prompt=market_question,
        additional_information=additional_information,
        engine=engine,
        temperature=temperature,
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
        return self.is_predictable_restricted(market_question, None)

    def is_predictable_restricted(self, market_question: str, time_restriction_up_to: datetime | None) -> bool:
        return is_predictable(question=market_question)

    def research_restricted(self, market_question: str, time_restriction_up_to: datetime | None) -> str:
        return research_autonolas(
                prompt=market_question,
                time_restriction_up_to=time_restriction_up_to,
                engine=self.model,
                embedding_model=self.embedding_model,
            )
    
    def predict(self, market_question: str) -> Prediction:
        return self.predict_restricted(market_question, None)

    def predict_restricted(
        self, market_question: str, time_restriction_up_to: datetime | None
    ) -> Prediction:
        try:
            researched = self.research_restricted(market_question=market_question, time_restriction_up_to=time_restriction_up_to)
            return _make_prediction(
                market_question=market_question,
                additional_information=researched,
                engine=self.model,
                temperature=self.temperature,
            )
        except ValueError as e:
            print(f"Error in OlasAgent's predict: {e}")
            return Prediction()


class EvoAgent(AbstractBenchmarkedAgent):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        agent_name: str = "evo",
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
        return self.is_predictable_restricted(market_question, None)

    def is_predictable_restricted(self, market_question: str, time_restriction_up_to: datetime | None) -> bool:
        return is_predictable(question=market_question)
    
    def predict(self, market_question: str) -> Prediction:
        return self.predict_restricted(market_question, None)

    def predict_restricted(
        self, market_question: str, time_restriction_up_to: datetime | None
    ) -> Prediction:
        try:
            report, _ = research_evo(
                goal=market_question,
                model=self.model,
                use_summaries=self.use_summaries,
                use_tavily_raw_content=self.use_tavily_raw_content,
                time_restriction_up_to=time_restriction_up_to,
            )
            return _make_prediction(
                market_question=market_question,
                additional_information=report,
                engine=self.model,
                temperature=self.temperature,
            )
        except ValueError as e:
            print(f"Error in EvoAgent's predict: {e}")
            return Prediction()


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

    def research_restricted(self, market_question: str, time_restriction_up_to: datetime | None) -> str:
        questions = rephrase_question(question=market_question)

        report_original = super().research_restricted(market_question=questions.original_question, time_restriction_up_to=time_restriction_up_to)
        report_negated = super().research_restricted(market_question=questions.negated_question, time_restriction_up_to=time_restriction_up_to)
        report_universal = super().research_restricted(market_question=questions.open_ended_question,time_restriction_up_to=time_restriction_up_to)

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
    EvoAgent,
    RandomAgent,
    QuestionOnlyAgent,
    FixedAgent,
]
