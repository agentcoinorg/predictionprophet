import os
from typing import cast
import discord
import re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prediction_prophet.benchmark.agents import _make_prediction
from prediction_prophet.functions.create_embeddings_from_results import (
    create_embeddings_from_results,
)
from prediction_prophet.functions.generate_subqueries import generate_subqueries
from prediction_prophet.functions.is_predictable_and_binary import (
    is_predictable_and_binary,
)
from prediction_prophet.functions.prepare_report import prepare_report
from prediction_prophet.functions.rerank_subqueries import rerank_subqueries
from prediction_prophet.functions.scrape_results import scrape_results
from prediction_prophet.functions.search import search
from prediction_market_agent_tooling.tools.utils import secret_str_from_env
from prediction_market_agent_tooling.benchmark.utils import OutcomePrediction

load_dotenv()


def remove_username_from_message_content(input_string: str):
    # Remove the user mention pattern `<@...>`
    cleaned = re.sub(r"<@\d+>", "", input_string)
    # Strip leading and trailing whitespace
    cleaned = cleaned.strip()
    # Replace multiple spaces with a single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


model: str = "gpt-4-0125-preview"
initial_subqueries_limit: int = 20
subqueries_limit: int = 4
scrape_content_split_chunk_size: int = 800
scrape_content_split_chunk_overlap: int = 225
top_k_per_query: int = 8

tavily_api_key = secret_str_from_env("TAVILY_API_KEY")
discord_bot_token = secret_str_from_env("DISCORD_BOT_TOKEN")

if tavily_api_key == None:
    raise Exception("No Tavily API Key provided")

if discord_bot_token == None:
    raise Exception("No discord bot token provided")


class ProphetClient(discord.Client):
    async def on_ready(self):
        print(f"Logged on as {self.user}!")

    async def on_message(self, message):
        """
        TODO:
            - Add channel ID
            - Add validation of 5 requets per user per day
            - 
        """
        if str(self.user.id) in message.content:
            goal = remove_username_from_message_content(message.content)
            channel = await message.create_thread(
                name=f'Assessing the likelihood of "{goal}" occurring'
            )
            message_prophet = "# Evaluating question..."
            message = await channel.send(message_prophet)
            (is_predictable, reasoning) = is_predictable_and_binary(goal)
            if not is_predictable:
                await message.edit(
                    content=f"The agent thinks this question is not predictable: \n{reasoning}"
                )
                return
            message_prophet = message_prophet + "\n### Generating subqueries..."
            await message.edit(content=message_prophet)
            queries = generate_subqueries(
                query=goal, limit=initial_subqueries_limit, model=model
            )
            message_prophet = message_prophet + "\n### Reranking subqueries..."
            await message.edit(content=message_prophet)
            queries = (
                rerank_subqueries(queries=queries, goal=goal, model=model)[
                    :subqueries_limit
                ]
                if initial_subqueries_limit > subqueries_limit
                else queries
            )

            message_prophet = message_prophet + "\n### Searching the web..."
            await message.edit(content=message_prophet)
            search_results_with_queries = search(
                queries,
                lambda result: not result.url.startswith("https://www.youtube"),
                tavily_api_key=tavily_api_key,
            )

            if not search_results_with_queries:
                message_prophet = message_prophet = (
                    "\n### No search results found for goal"
                )
                await message.edit(content=message_prophet)
                return

            message_prophet = message_prophet + "\n### Scraping web results..."
            await message.edit(content=message_prophet)
            scrape_args = [result for (_, result) in search_results_with_queries]

            scraped = scrape_results(scrape_args)
            scraped = [result for result in scraped if result.content != ""]

            message_prophet = (
                message_prophet + "\n### Performing similarity searches..."
            )
            await message.edit(content=message_prophet)

            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", "  "],
                chunk_size=scrape_content_split_chunk_size,
                chunk_overlap=scrape_content_split_chunk_overlap,
            )
            collection = create_embeddings_from_results(scraped, text_splitter)

            vector_result_texts: list[str] = []
            for query in queries:
                top_k_per_query_results = collection.similarity_search(
                    query, k=top_k_per_query
                )
                vector_result_texts += [
                    result.page_content
                    for result in top_k_per_query_results
                    if result.page_content not in vector_result_texts
                ]

            message_prophet = message_prophet + "\n### Preparing report..."
            await message.edit(content=message_prophet)

            report = prepare_report(goal, vector_result_texts, model=model)

            message_prophet = message_prophet + "\n### Making prediction..."
            await message.edit(content=message_prophet)

            prediction = _make_prediction(
                market_question=goal,
                additional_information=report,
                engine="gpt-4-0125-preview",
                temperature=0.0,
            )

            if prediction.outcome_prediction == None:
                message_prophet = (
                    message_prophet + "\n### The agent failed to generate a prediction"
                )
                await message.edit(content=message_prophet)
                return

            outcome_prediction = cast(OutcomePrediction, prediction.outcome_prediction)
            message_prophet = (
                message_prophet
                + f"\n# Probability: {outcome_prediction.p_yes * 100}%. Confidence: {outcome_prediction.confidence * 100}%"
            )
            await message.edit(content=message_prophet)


intents = discord.Intents(messages=True)
intents.message_content = True
client = ProphetClient(intents=intents)
client.run(discord_bot_token)
