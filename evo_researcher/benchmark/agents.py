import typing as t
from evo_researcher.benchmark.logger import BaseLogger

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
from evo_researcher.autonolas.research import make_prediction, get_urls_from_queries
from evo_researcher.autonolas.research import research as research_autonolas
from evo_researcher.functions.evaluate_question import is_predictable
from evo_researcher.functions.rephrase_question import rephrase_question
from langchain.text_splitter import RecursiveCharacterTextSplitter
from evo_researcher.functions.create_embeddings_from_results import create_embeddings_from_results
from evo_researcher.functions.generate_subqueries import generate_subqueries
from evo_researcher.functions.prepare_report import prepare_report, prepare_summary
from evo_researcher.models.WebScrapeResult import WebScrapeResult
from evo_researcher.functions.rerank_subqueries import rerank_subqueries
from evo_researcher.functions.scrape_results import scrape_results
from evo_researcher.functions.search import search
from evo_researcher.functions.utils import url_is_older_than
from evo_researcher.models.WebSearchResult import WebSearchResult
from unittest.mock import patch
from evo_researcher.functions.search import search

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
        return is_predictable(question=market_question)

    def is_predictable_restricted(self, market_question: str, time_restriction_up_to: datetime) -> bool:
        return is_predictable(question=market_question)
    
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
    
        with patch('evo_researcher.autonolas.research.get_urls_from_queries', side_effect=side_effect, autospec=True):
            return self.predict(market_question)


class EvoAgent(AbstractBenchmarkedAgent):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        agent_name: str = "evo",
        use_summaries: bool = False,
        use_tavily_raw_content: bool = False,
        max_workers: t.Optional[int] = None,
        logger: BaseLogger = BaseLogger()
    ):
        super().__init__(agent_name=agent_name, max_workers=max_workers)
        self.model = model
        self.temperature = temperature
        self.use_summaries = use_summaries
        self.use_tavily_raw_content = use_tavily_raw_content
        self.logger = logger

    def is_predictable(self, market_question: str) -> bool:
        return is_predictable(question=market_question)

    def is_predictable_restricted(self, market_question: str, time_restriction_up_to: datetime) -> bool:
        return is_predictable(question=market_question)
    
    def predict(self, market_question: str) -> Prediction:
        try:
            report = self.research(
                goal=market_question,
                model=self.model,
                use_summaries=self.use_summaries,
                use_tavily_raw_content=self.use_tavily_raw_content,
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
        
    def predict_from_research(
        self, market_question: str, research_report: str
    ) -> Prediction:
        try:
            return _make_prediction(
                market_question=market_question,
                additional_information=research_report,
                engine=self.model,
                temperature=self.temperature,
            )
        except ValueError as e:
            print(f"Error in EvoAgent's predict: {e}")
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
    
        with patch('evo_researcher.functions.research.search', side_effect=side_effect, autospec=True):
            return self.predict(market_question)

    def research(
        self,
        goal: str,
        use_summaries: bool,
        model: str = "gpt-4-1106-preview",
        initial_subqueries_limit: int = 20,
        subqueries_limit: int = 4,
        scrape_content_split_chunk_size: int = 800,
        scrape_content_split_chunk_overlap: int = 225,
        top_k_per_query: int = 8,
        use_tavily_raw_content: bool = False,
    ) -> str:
        self.logger.info("Started subqueries generation")
        queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, model=model)
        
        stringified_queries = '\n- ' + '\n- '.join(queries)
        self.logger.info(f"Generated subqueries: {stringified_queries}")
        
        self.logger.info("Started subqueries reranking")
        queries = rerank_subqueries(queries=queries, goal=goal, model=model)[:subqueries_limit] if initial_subqueries_limit > subqueries_limit else queries

        stringified_queries = '\n- ' + '\n- '.join(queries)
        self.logger.info(f"Reranked subqueries. Will use top {subqueries_limit}: {stringified_queries}")
        
        self.logger.info(f"Started web searching")
        search_results_with_queries = search(
            queries, 
            lambda result: not result.url.startswith("https://www.youtube")
        )

        if not search_results_with_queries:
            raise ValueError(f"No search results found for the goal {goal}.")

        scrape_args = [result for (_, result) in search_results_with_queries]
        websites_to_scrape = set([result.url for result in scrape_args])
        
        stringified_websites = '\n- ' + '\n- '.join(websites_to_scrape)
        self.logger.info(f"Found the following relevant results: {stringified_websites}")
        
        self.logger.info(f"Started scraping of web results")
        scraped = scrape_results(scrape_args) if not use_tavily_raw_content else [WebScrapeResult(
            query=result.query,
            url=result.url,
            title=result.title,
            content=result.raw_content,
        ) for result in scrape_args if result.raw_content]
        scraped = [result for result in scraped if result.content != ""]
        
        self.logger.info(f"Scraped content from {len(scraped)} websites")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "  "],
            chunk_size=scrape_content_split_chunk_size,
            chunk_overlap=scrape_content_split_chunk_overlap
        )
        
        self.logger.info("Started embeddings creation")
        collection = create_embeddings_from_results(scraped, text_splitter)
        self.logger.info("Embeddings created")

        vector_result_texts: list[str] = []
        url_to_content_deemed_most_useful: dict[str, str] = {}

        stringified_queries = '\n- ' + '\n- '.join(queries)
        self.logger.info(f"Started similarity searches for: {stringified_queries}")
        for query in queries:
            top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
            vector_result_texts += [result.page_content for result in top_k_per_query_results if result.page_content not in vector_result_texts]

            for x in top_k_per_query_results:
                # `x.metadata["content"]` holds the whole url's web page, so it's ok to overwrite the value of the same url.
                url_to_content_deemed_most_useful[x.metadata["url"]] = x.metadata["content"]
        
        stringified_urls = '\n- ' + '\n- '.join(url_to_content_deemed_most_useful.keys())
        self.logger.info(f"Found {len(vector_result_texts)} information chunks across the following sites: {stringified_urls}")

        if use_summaries:
            self.logger.info(f"Started summarizing information")
            vector_result_texts = [
                prepare_summary(
                    goal,
                    content,
                    "gpt-3.5-turbo-0125",
                    trim_content_to_tokens=14_000
                )
                for content in url_to_content_deemed_most_useful.values()
            ]
            self.logger.info(f"Information summarized")

        self.logger.info(f"Started preparing report")
        report = prepare_report(goal, vector_result_texts, model=model)
        self.logger.info(f"Report prepared")

        return report

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
    EvoAgent,
    RandomAgent,
    QuestionOnlyAgent,
    FixedAgent,
]
