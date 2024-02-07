import dotenv
import json
import os
import typing as t

from evo_researcher.functions.evaluate_question import evaluate_question, EvalautedQuestion
from evo_researcher.functions.rephrase_question import rephrase_question
from evo_researcher.functions.research import research as research_evo
from evo_researcher.autonolas.research import (
    EmbeddingModel,
    make_prediction,
    Prediction as LLMCompletionPredictionDict,
    research as research_autonolas,
)
from evo_researcher.benchmark.utils import (
    Prediction, 
    OutcomePrediction, 
    EvalautedQuestion,
)


def _make_prediction(
    market_question: str, additional_information: str, evaluation_information: t.Optional[EvalautedQuestion], engine: str, temperature: float
) -> Prediction:
    """
    We prompt model to output a simple flat JSON and convert it to a more structured pydantic model here.
    """
    prediction = make_prediction(
        prompt=market_question, additional_information=additional_information, engine=engine, temperature=temperature
    )
    return completion_prediction_json_to_pydantic_model(prediction, evaluation_information)


def completion_prediction_json_to_pydantic_model(
    completion_prediction: LLMCompletionPredictionDict, 
    evaluation_information: t.Optional[EvalautedQuestion],
) -> Prediction:
    return Prediction(
        evaluation=evaluation_information,
        outcome_prediction=OutcomePrediction(
            p_yes=completion_prediction["p_yes"],
            confidence=completion_prediction["confidence"],
            info_utility=completion_prediction["info_utility"],
        ),
    )


class AbstractBenchmarkedAgent:
    def __init__(self, agent_name: str, max_workers: t.Optional[int] = None):
        self.agent_name = agent_name
        self.max_workers = max_workers  # Limit the number of workers that can run this worker in parallel threads

    def evaluate(self, market_question: str) -> EvalautedQuestion:
        raise NotImplementedError

    def research(self, market_question: str) -> t.Optional[str]:
        raise NotImplementedError

    def predict(self, market_question: str, researched: str, evaluated: EvalautedQuestion) -> Prediction:
        raise NotImplementedError

    def evaluate_research_predict(self, market_question: str) -> Prediction:
        eval = self.evaluate(market_question=market_question)
        if not eval.is_predictable:
            return Prediction(evaluation=eval)
        researched = self.research(market_question=market_question)
        if researched is None:
            return Prediction(evaluation=eval)
        return self.predict(
            market_question=market_question, 
            researched=researched,
            evaluated=eval,
        )

      
class OlasAgent(AbstractBenchmarkedAgent):
    def __init__(self, model: str, temperature: float, agent_name: str = "olas", max_workers: t.Optional[int] = None, embedding_model: EmbeddingModel = EmbeddingModel.spacy):
        super().__init__(agent_name=agent_name, max_workers=max_workers)
        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model

    def evaluate(self, market_question: str) -> EvalautedQuestion:
        return evaluate_question(question=market_question)

    def research(self, market_question: str) -> t.Optional[str]:
        try:
            return research_autonolas(
                prompt=market_question,
                engine=self.model,
                embedding_model=self.embedding_model,
            )
        except ValueError as e:
            print(f"Error in OlasAgent's research: {e}")
            return None
        
    def predict(self, market_question: str, researched: str, evaluated: EvalautedQuestion) -> Prediction:
        try:
            return _make_prediction(
                market_question=market_question,
                additional_information=researched,
                evaluation_information=evaluated,
                engine=self.model,
                temperature=self.temperature,
            )
        except ValueError as e:
            print(f"Error in OlasAgent's predict: {e}")
            return Prediction(evaluation=evaluated)

class EvoAgent(AbstractBenchmarkedAgent):
    def __init__(self, model: str, temperature: float, agent_name: str = "evo", max_workers: t.Optional[int] = None):
        super().__init__(agent_name=agent_name, max_workers=max_workers)
        self.model = model
        self.temperature = temperature

    def evaluate(self, market_question: str) -> EvalautedQuestion:
        return evaluate_question(question=market_question)

    def research(self, market_question: str) -> t.Optional[str]:
        dotenv.load_dotenv()
        open_ai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        try:
            report, _ = research_evo(
                goal=market_question,
                openai_key=open_ai_key,
                tavily_key=tavily_key,
                model=self.model,
            )
            return report
        except ValueError as e:
            print(f"Error in EvoAgent's research: {e}")
            return None

    def predict(self, market_question: str, researched: str, evaluated: EvalautedQuestion) -> Prediction:
        try:
            return _make_prediction(
                market_question=market_question, 
                additional_information=researched,
                evaluation_information=evaluated,
                engine=self.model,
                temperature=self.temperature,
            )
        except ValueError as e:
            print(f"Error in EvoAgent's predict: {e}")
            return Prediction(evaluation=evaluated)


class RephrasingOlasAgent(OlasAgent):
    def __init__(
        self,
        model: str,
        temperature: float,
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

    def research(self, market_question: str) -> t.Optional[str]:
        questions = rephrase_question(question=market_question)

        report_original = super().research(market_question=questions.original_question)
        report_negated = super().research(market_question=questions.negated_question)
        report_universal = super().research(market_question=questions.open_ended_question)

        report_concat = "\n\n---\n\n".join([
            f"### {r_name}\n\n{r}"
            for r_name, r in [
                ("Research based on the question", report_original), 
                ("Research based on the negated question", report_negated), 
                ("Research based on the universal search query", report_universal)
            ] 
            if r is not None
        ])

        return report_concat


AGENTS = [
    OlasAgent,
    RephrasingOlasAgent,
    EvoAgent,
]
