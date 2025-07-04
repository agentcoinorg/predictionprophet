import os
import tenacity
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
from typing import Annotated, Literal, Any, Dict, Generator, List, Optional, Tuple, TypedDict, Sequence
from datetime import datetime, timezone
import json
from dotenv import load_dotenv
import re
import gc
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from concurrent.futures import Future
from itertools import groupby
from operator import itemgetter
from enum import Enum
from bs4 import BeautifulSoup, NavigableString
from googleapiclient.discovery import build
from prediction_prophet.functions.parallelism import THREADPOOL
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic import BaseModel, Field


import requests
from requests import Session
import spacy
import spacy.util
import spacy.cli
import tiktoken

from langchain_openai import OpenAIEmbeddings

from dateutil import parser
from prediction_prophet.functions.utils import check_not_none
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.tools.caches.db_cache import db_cache
from prediction_prophet.functions.parallelism import par_map
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.google_utils import search_google_gcp
from prediction_market_agent_tooling.logprobs_parser import LogprobsParser, FieldLogprobs
from prediction_market_agent_tooling.gtypes import Wei


load_dotenv()

NUM_URLS_EXTRACT = 5
MAX_TOTAL_TOKENS_CHAT_COMPLETION = 4000  # Set the limit for cost efficiency
WORDS_PER_TOKEN_FACTOR = 0.75
DEFAULT_OPENAI_SETTINGS = {
    "max_compl_tokens": 500,
    "temperature": 0,
}

ALLOWED_TOOLS = [
    "prediction-sentence-embedding-conservative",
    "prediction-sentence-embedding-bold",
]
TOOL_TO_ENGINE = {
    "prediction-sentence-embedding-conservative": "gpt-3.5-turbo",
    "prediction-sentence-embedding-bold": "gpt-4",
}


# * Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
# * If the information in "ADDITIONAL_INFORMATION" indicate without a doubt that the event has already happened, it is very likely that the outcome of the market question will be `Yes`.
# * If the information in "ADDITIONAL_INFORMATION" indicate that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
# * If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
# * If recent information indicates a status change in the future, pay close attention to the date of the status change and if it is before or after the closing date of the 'market question' and adjust your probability estimation accordingly, keeping the examples under "EXAMPLES" and their outcomes given the point in time of the status change in mind.
# * If there exist recent information indicating that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
# * Note that the sentences within the information items provided under "ADDITIONAL_INFORMATION" are a concatenation of the sentences from web pages that have the highest vector similarity to the 'market question'. Thus, the paragraphs do not represent the original context of the sentences and you should evaluate each sentence individually.


PREDICTION_PROMPT = """
INTRODUCTION:
You are a Large Language Model (LLM) within a multi-agent system. Your primary task is to accurately estimate the probabilities for the outcome of a 'market question', \
found in 'USER_PROMPT'. The market question is part of a prediction market, where users can place bets on the outcomes of market questions and earn rewards if the selected outcome occurrs. The 'market question' \
in this scenario has only two possible outcomes: `Yes` or `No`. Each market has a closing date at which the outcome is evaluated. This date is typically stated within the market question.  \
The closing date is considered to be 23:59:59 of the date provided in the market question. If the event specified in the market question has not occurred before the closing date, the market question's outcome is `No`. \
If the event has happened before the closing date, the market question's outcome is `Yes`. You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your probability estimation. You must adhere to the following 'INSTRUCTIONS'.  


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'market question'.
* If the 'market question' implies more than two outcomes, output the response "Error" and halt further processing.
* When the current time {timestamp} has passed the closing date of the market and the event specified in the market question has not happened, the market question's outcome is `No` and the user who placed a bet on `No` will receive a reward.
* When the current time {timestamp} has passed the closing date of the market and the event has happened before, the market question's final outcome is `Yes` and the user who placed a bet on `yes` will receive a reward.
* Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
* The probability estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user.
* Utilize your training data and the information provided under "ADDITIONAL_INFORMATION" to generate probability estimations for the outcomes of the 'market question'.
* Examine the itemized list under "ADDITIONAL_INFORMATION" thoroughly and use all the relevant information for your probability estimation. This data is sourced from a Google search engine query done a few seconds ago. 
* Use any relevant item in "ADDITIONAL_INFORMATION" in addition to your training data to make the probability estimation. You can assume that you have been provided with the most current and relevant information available on the internet. Still pay close attention on the release and modification timestamps provided in parentheses right before each information item. Some information might be outdated and not relevant anymore.
* More recent information indicated by the timestamps provided in parentheses right before each information item overrides older information within ADDITIONAL_INFORMATION and holds more weight for your probability estimation.
* If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
* Even if not all information might not be released today, you can assume that there haven't been publicly available updates in the meantime except for those inside ADDITIONAL_INFORMATION.
* If the information in "ADDITIONAL_INFORMATION" indicate without a doubt that the event has already happened, it is very likely that the outcome of the market question will be `Yes`.
* If the information in "ADDITIONAL_INFORMATION" indicate that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
* The closer the current time `{timestamp}` is to the closing time the higher the likelyhood that the outcome of the market question will be `No`, if recent information do not clearly indicate that the event will occur before the closing date.
* If there exist recent information indicating that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.


USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain {n_fields} fields: {fields_list}.
{fields_description}
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object in your response. Do not include any other contents in your response.
"""

PREDICTION_PROMPT_SCALAR = """
INTRODUCTION:
You are a Large Language Model (LLM) within a multi-agent system. Your primary task is to accurately estimate the 'scalar_value' for the outcome of a 'market question', \
found in 'USER_PROMPT'. The market question is part of a prediction market, where users can place bets on the outcomes of market questions and earn rewards if the predicted 'scalar_value' is close to the actual outcome.
Each market has {market_upper_bound} and {market_lower_bound} values use those values to calibrate your expectation about your prediction   .
Each market has a closing date at which the outcome is evaluated. This date is typically stated within the market question.  \
The closing date is considered to be 23:59:59 of the date provided in the market question. \
You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your 'scalar_value' estimation. You must adhere to the following 'INSTRUCTIONS'. 


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'market question'.
* Estimate the 'scalar_value' for the outcome of the market question.
* Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
* The 'scalar_value' estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user.
* Utilize your training data and the information provided under "ADDITIONAL_INFORMATION" to generate 'scalar_value' estimations for the outcomes of the 'market question'.
* Examine the itemized list under "ADDITIONAL_INFORMATION" thoroughly and use all the relevant information for your 'scalar_value' estimation. This data is sourced from a Google search engine query done a few seconds ago. 
* Use any relevant item in "ADDITIONAL_INFORMATION" in addition to your training data to make the 'scalar_value' estimation. You can assume that you have been provided with the most current and relevant information available on the internet. Still pay close attention on the release and modification timestamps provided in parentheses right before each information item. Some information might be outdated and not relevant anymore.
* More recent information indicated by the timestamps provided in parentheses right before each information item overrides older information within ADDITIONAL_INFORMATION and holds more weight for your 'scalar_value' estimation.
* If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
* Even if not all information might not be released today, you can assume that there haven't been publicly available updates in the meantime except for those inside ADDITIONAL_INFORMATION.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.


USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain {n_fields} fields: {fields_list}.
{fields_description}
* The 'scalar_value' is float number.
* Output only the JSON object in your response. Do not include any other contents in your response.
"""

PREDICTION_PROMPT_CATEGORICAL = """
INTRODUCTION:
You are a Large Language Model (LLM) within a multi-agent system. Your primary task is to accurately estimate the probabilities for the outcome of a 'market question', \
found in 'USER_PROMPT'. The market question is part of a prediction market, where users can place bets on the outcomes of market questions and earn rewards if the selected outcome occurrs.
Each market has a closing date at which the outcome is evaluated. This date is typically stated within the market question.  \
The closing date is considered to be 23:59:59 of the date provided in the market question. \
You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your probability estimation. You must adhere to the following 'INSTRUCTIONS'.  


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'market question'.
* Estimate probabilities for each possible outcome of the market question, which are provided in the 'POSSIBLE_OUTCOMES' section.
* Probabilities have to sum up to 1.
* Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
* The probability estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user.
* Utilize your training data and the information provided under "ADDITIONAL_INFORMATION" to generate probability estimations for the outcomes of the 'market question'.
* Examine the itemized list under "ADDITIONAL_INFORMATION" thoroughly and use all the relevant information for your probability estimation. This data is sourced from a Google search engine query done a few seconds ago. 
* Use any relevant item in "ADDITIONAL_INFORMATION" in addition to your training data to make the probability estimation. You can assume that you have been provided with the most current and relevant information available on the internet. Still pay close attention on the release and modification timestamps provided in parentheses right before each information item. Some information might be outdated and not relevant anymore.
* More recent information indicated by the timestamps provided in parentheses right before each information item overrides older information within ADDITIONAL_INFORMATION and holds more weight for your probability estimation.
* If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
* Even if not all information might not be released today, you can assume that there haven't been publicly available updates in the meantime except for those inside ADDITIONAL_INFORMATION.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.


USER_PROMPT:
```
{user_prompt}
```

POSSIBLE_OUTCOMES:
```
{possible_outcomes}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain {n_fields} fields: {fields_list}.
{fields_description}
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object in your response. Do not include any other contents in your response.
"""


FIELDS_DESCRIPTIONS = {
    "reasoning": "A string containing the reasoning behind your decision, and the rest of the answer you're about to give.",
    "decision": "The decision you made. Either `y` (for `Yes`) or `n` (for `No`).",
    "p_yes": "Probability that the market question's outcome will be `Yes`. Ranging from 0 (lowest probability) to 1 (maximum probability).",
    "p_no": "Probability that the market questions outcome will be `No`. Ranging from 0 (lowest probability) to 1 (maximum probability).",
    "confidence": "Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.",
    "info_utility": "Utility of the information provided in 'ADDITIONAL_INFORMATION' to help you make the probability estimation ranging from 0 (lowest utility) to 1 (maximum utility).",
}
FIELDS_DESCRIPTIONS_CATEGORICAL = {
    "reasoning": "A string containing the reasoning behind your decision, and the rest of the answer you're about to give.",
    "decision": "The decision you made. The given outcome you have chosen.",
    "probabilities": "A dictionary containing the probabilities for each possible outcome of the market question. The keys are the possible outcomes, and the values are the corresponding probabilities.",
    "confidence": "Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.",
    "info_utility": "Utility of the information provided in 'ADDITIONAL_INFORMATION' to help you make the probability estimation ranging from 0 (lowest utility) to 1 (maximum utility).",
}

FIELDS_DESCRIPTIONS_SCALAR = {
    "reasoning": "A string containing the reasoning behind your decision, and the rest of the answer you're about to give.",
    "scalar_value": "Predicted value of the market question, float number",
    "confidence": "Indicating the confidence in the estimated value you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.",
    "info_utility": "Utility of the information provided in 'ADDITIONAL_INFORMATION' to help you make the probability estimation ranging from 0 (lowest utility) to 1 (maximum utility).",
}

URL_QUERY_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to formulate search engine queries based on \
a user's 'event question', which specifies an event and any accompanying conditions. The 'event question' allows \
only two outcomes: the event will either occur or not, given the conditions. Find the 'event question' under 'USER_PROMPT' \
and adhere to the 'INSTRUCTIONS'.

INSTRUCTIONS:
* Carefully read the 'event question' under 'USER_PROMPT', enclosed by triple backticks.
* If the 'event question' has more than two outcomes, respond with "Error" and ignore further instructions.
* Create a list of 1-3 unique search queries likely to yield relevant and contemporary information for assessing the event's likelihood under the given conditions.
* Each query must be unique, and they should not overlap or yield the same set of results.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{event_question}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain two fields: "queries", and "urls".
   - "queries": A 1-5 item array of the generated search engine queries.
* Include only the JSON object in your output.
"""

# Global constants for possible attribute names for release and update dates
RELEASE_DATE_NAMES = [
    "date",
    "pubdate",
    "publishdate",
    "OriginalPublicationDate",
    "article:published_time",
    "sailthru.date",
    "article.published",
    "published-date",
    "og:published_time",
    "publication_date",
    "publishedDate",
    "dc.date",
    "DC.date",
    "article:published",
    "article_date_original",
    "cXenseParse:recs:publishtime",
    "DATE_PUBLISHED",
    "pub-date",
    "pub_date",
    "datePublished",
    "date_published",
    "time_published",
    "article:published_date",
    "parsely-pub-date",
    "publish-date",
    "pubdatetime",
    "published_time",
    "publishedtime",
    "article_date",
    "created_date",
    "published_at",
    "lastPublishedDate",
    "og:published_time",
    "og:release_date",
    "article:published_time",
    "og:publication_date",
    "og:pubdate",
    "article:publication_date",
    "product:availability_starts",
    "product:release_date",
    "event:start_date",
    "event:release_date",
    "og:time_published",
    "og:start_date",
    "og:created",
    "og:creation_date",
    "og:launch_date",
    "og:first_published",
    "og:original_publication_date",
    "article:published",
    "article:pub_date",
    "news:published_time",
    "news:publication_date",
    "blog:published_time",
    "blog:publication_date",
    "report:published_time",
    "report:publication_date",
    "webpage:published_time",
    "webpage:publication_date",
    "post:published_time",
    "post:publication_date",
    "item:published_time",
    "item:publication_date",
]

UPDATE_DATE_NAMES = [
    "lastmod",
    "lastmodified",
    "last-modified",
    "updated",
    "dateModified",
    "article:modified_time",
    "modified_date",
    "article:modified",
    "og:updated_time",
    "mod_date",
    "modifiedDate",
    "lastModifiedDate",
    "lastUpdate",
    "last_updated",
    "LastUpdated",
    "UpdateDate",
    "updated_date",
    "revision_date",
    "sentry:revision",
    "article:modified_date",
    "date_updated",
    "time_updated",
    "lastUpdatedDate",
    "last-update-date",
    "lastupdate",
    "dateLastModified",
    "article:update_time",
    "modified_time",
    "last_modified_date",
    "date_last_modified",
    "og:updated_time",
    "og:modified_time",
    "article:modified_time",
    "og:modification_date",
    "og:mod_time",
    "article:modification_date",
    "product:availability_ends",
    "product:modified_date",
    "event:end_date",
    "event:updated_date",
    "og:time_modified",
    "og:end_date",
    "og:last_modified",
    "og:modification_date",
    "og:revision_date",
    "og:last_updated",
    "og:most_recent_update",
    "article:updated",
    "article:mod_date",
    "news:updated_time",
    "news:modification_date",
    "blog:updated_time",
    "blog:modification_date",
    "report:updated_time",
    "report:modification_date",
    "webpage:updated_time",
    "webpage:modification_date",
    "post:updated_time",
    "post:modification_date",
    "item:updated_time",
    "item:modification_date",
]

# Global constant for HTML tags to remove
HTML_TAGS_TO_REMOVE = [
    "script",
    "style",
    "header",
    "footer",
    "aside",
    "nav",
    "form",
    "button",
    "iframe",
    "input",
    "textarea",
    "select",
    "option",
    "label",
    "fieldset",
    "legend",
    "img",
    "audio",
    "video",
    "source",
    "track",
    "canvas",
    "svg",
    "object",
    "param",
    "embed",
    "link",
]


class EmbeddingModel(Enum):
    spacy = "spacy"
    openai = "openai"


class Prediction(BaseModel):
    decision: Literal["y", "n"]
    p_yes: Annotated[Probability, Field(ge=0.0, le=1.0)]
    p_no: Annotated[Probability, Field(ge=0.0, le=1.0)]
    confidence: float
    info_utility: float
    reasoning: Optional[str] = None
    logprobs: Optional[list[FieldLogprobs]] = []


class ScalarPrediction(TypedDict):
    scalar_value: Wei
    upperBound: Wei
    lowerBound: Wei
    confidence: float
    info_utility: float
    reasoning: Optional[str]
    logprobs: Optional[list[FieldLogprobs]]

class CategoricalPrediction(TypedDict):
    decision: str
    probabilities: Dict[str, Probability]
    confidence: float
    info_utility: float
    reasoning: Optional[str]


def list_to_list_str(l: List[str]) -> str:
    """
    Converts ["foo", "bar", "baz"] to '"foo", "bar" and "baz"'.
    """
    list_str = ""
    for i, item in enumerate(l):
        if i == 0:
            list_str += f'"{item}"'
        elif i == len(l) - 1:
            list_str += f' and "{item}"'
        else:
            list_str += f', "{item}"'
    return list_str

def fields_dict_to_bullet_list(fields_dict: Dict[str, str]) -> str:
    """
    Converts a dictionary of field names to their descriptions into a bullet list.
    """
    bullet_list = ""
    for i, (field, description) in enumerate(fields_dict.items()):
        if i > 1:
            bullet_list += "\n"
        bullet_list += f"  - {field}: {description}"
        if i == 0:
            bullet_list += "\n"
    return bullet_list


def download_spacy_model(model_name: str) -> None:
    """Downloads the specified spaCy language model if it is not already installed."""
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("spacy model_name must be a non-empty string")
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name)
    else:
        logger.info(f"{model_name} is already installed.")


def extract_event_date(doc_question: spacy.tokens.Doc) -> Optional[str]:
    """
    Extracts the event date from the event question if present.

    Args:
        doc_question (spaCy Doc): Document text as a spaCy Doc object.

    Returns:
        str: The event date in year-month-day format if present, otherwise None.
    """

    event_date_ymd = None

    # Extract the date from the event question if present
    for ent in doc_question.ents:
        if ent.label_ == "DATE":
            event_date_ymd = standardize_date(ent.text)

    # If event date not formatted as YMD or not found, return None
    try:
        datetime.strptime(event_date_ymd, "%Y-%m-%d")  # type: ignore
    except (ValueError, TypeError):
        return None
    else:
        return event_date_ymd


def get_max_tokens_for_additional_information(
    max_compl_tokens: int,
    prompt: str,
    enc: tiktoken.Encoding,
    safety_factor: float = 1.05,
) -> int:
    """
    Calculates the estimated maximum number of tokens that can be consumed by the additional information string.

    Args:
        max_compl_tokens (int): The maximum number of chat completion output tokens.
        prompt (str): The user prompt containing the event question.
        enc (tiktoken.Encoding): The tiktoken encoding to be used.
        safety_factor (float, optional): The safety factor to be used for prompt variations and message headers. Defaults to 1.05.

    Returns:
        int: The estimated number of tokens that can be consumed by the additional information string.
    """

    # Encode the strings into tokens
    user_prompt_enc = enc.encode(prompt)
    prediction_prompt_enc = enc.encode(PREDICTION_PROMPT)

    # Calculate token sum of thus far allocated tokens for the final prediction prompt
    token_sum = len(user_prompt_enc) + len(prediction_prompt_enc) + max_compl_tokens
    token_sum_safety = token_sum * safety_factor

    return int(MAX_TOTAL_TOKENS_CHAT_COMPLETION - token_sum_safety)


def truncate_additional_information(
    additional_informations: str,
    max_add_tokens: int,
    enc: tiktoken.Encoding,
) -> str:
    """
    Truncates additional information string to a specified number of tokens using tiktoken encoding.

    Args:
        additional_informations (str): The additional information string to be truncated.
        max_add_tokens (int): The maximum number of tokens allowed for the additional information string.
        enc (tiktoken.Encoding): The tiktoken encoding to be used.

    Returns:
    - str: The truncated additional information string.
    """

    # Encode the string into tokens
    add_enc = enc.encode(additional_informations)
    len_add_enc = len(add_enc)

    # Truncate additional information string if token sum exceeds maximum allowed
    if len_add_enc <= max_add_tokens:
        return additional_informations
    else:
        add_trunc_enc = add_enc[: -int(len_add_enc - max_add_tokens)]
        return enc.decode(add_trunc_enc)
    

def safe_get_urls_from_query(query: str, num: int = 3) -> List[str]:
    try:
        return get_urls_from_query(query, num)
    except ValueError as e:
        logger.warning(f"Error in get_urls_from_query: {e}")
        return []


def get_urls_from_query(
    query: str, num: int = 3
) -> List[str]:
    return get_urls_from_queries(queries=[query], num=num)


def get_urls_from_queries(queries: List[str], num: int = 3) -> List[str]:
    """
    Fetch unique URLs from search engine queries, limiting the number of URLs per query.

    Args:
        queries (List[str]): List of search engine queries.
        num (int, optional): Number of returned URLs per query. Defaults to 3.

    Raises:
        ValueError: If the number of URLs per query exceeds the maximum allowed.

    Returns:
        List[str]: Unique list of URLs, omitting PDF and download-related URLs.
    """

    results = set()
    max_num_fetch = 10

    if num > max_num_fetch:
        raise ValueError(f"The maximum number of URLs per query is {max_num_fetch}.")

    for query in queries:
        fetched_urls = search_google_gcp(
            query=query,
            num=max_num_fetch,  # Limit the number of returned URLs per query
        )

        # Add only unique URLs up to 'num' per query, omitting PDF and 'download' URLs
        count = 0
        for url in fetched_urls:
            if url not in results and not url.endswith(".pdf"):
                results.add(url)
                count += 1
                if count >= num:
                    break

    return list(results)


def standardize_date(date_text: str) -> str | None:
    """
    Standardizes a given date string to the format 'YYYY-MM-DD' or 'MM-DD' if possible.

    Args:
        date_text (str): The date string to be standardized.

    Raises:
        ValueError: If the date string cannot be parsed.

    Returns:
        str: The standardized date string if possible, otherwise None.
    """

    try:
        # Compile regex patterns for month and day
        month_regex = re.compile(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
            re.IGNORECASE,
        )
        day_regex = re.compile(r"\b\d{1,2}\b")

        # Parse date_text using dateutil parser
        parsed_date = parser.parse(date_text)

        # Check if year, month, and day are in the original date_text
        month_exists = month_regex.search(date_text) is not None
        day_exists = day_regex.search(date_text) is not None
        year_exists = str(parsed_date.year) in date_text

        # Format the parsed date accordingly
        if year_exists and month_exists and day_exists:
            return parsed_date.strftime("%Y-%m-%d")
        elif month_exists and day_exists:
            return parsed_date.strftime("%m-%d")
        else:
            return None
    except Exception:
        return None


def get_context_around_isolated_event_date(
    doc_text: spacy.tokens.Doc, event_date_ymd: str, len_sentence_threshold: int, max_context: int = 50
) -> list[str]:
    """
    Extract sentences around isolated dates within the text.

    Args:
        doc_text (spaCy Doc): Document text as a spaCy Doc object.
        event_date_ymd (str): Event date in year-day-month format.
        len_sentence_threshold (int): Minimum number of words required for a sentence to be considered contextful.
        max_context (int, optional): Maximum number of words to include in the context. Defaults to 50.

    Raises:
        ValueError: If maximum context is less than threshold or greater than 100.

    Returns:
        list: List of sentences surrounding the target date.
    """

    # Check max_context value constraints
    if max_context < len_sentence_threshold:
        raise ValueError(
            f"The maximum number of words must be greater than or equal to the minimum number of words ({len_sentence_threshold}) required for a sentence to be considered contextful."
        )
    if max_context > 100:
        raise ValueError(
            f"The maximum number of words must be less than or equal to 300."
        )

    contexts_list: list[str] = []
    len_doc_text = len(doc_text)

    # Extract the month and day from the event date
    event_date_md = event_date_ymd[5:]

    for ent in doc_text.ents:
        if ent.label_ == "DATE":
            standardized_date = standardize_date(ent.text)
            if standardized_date is None:
                continue

            # Check if the entity matches the target date
            if (
                standardized_date == event_date_ymd
                or standardized_date == event_date_md
            ):
                sentence = next(
                    sent
                    for sent in doc_text.sents
                    if sent.start <= ent.start and sent.end >= ent.end
                )

                context_words = len(sentence.text.split())

                # Extend the context if the sentence is too short
                if context_words < len_sentence_threshold:
                    start_token, end_token = sentence.start, sentence.end
                    while context_words < max_context:
                        # Extend the context from the start of the sentence
                        new_start = start_token - 1
                        while (
                            new_start >= 0 and doc_text[new_start].is_sent_start is None
                        ):
                            new_start -= 1
                        if new_start >= 0:
                            context_words += len(
                                doc_text[new_start:start_token].text.split()
                            )
                            start_token = new_start

                        # Break if max_context is reached
                        if context_words >= max_context:
                            break

                        # Extend the context from the end of the sentence
                        new_end = end_token + 1
                        while (
                            new_end < len_doc_text
                            and doc_text[new_end].sent == sentence.sent
                        ):
                            new_end += 1
                        if new_end < len_doc_text:
                            context_words += len(
                                doc_text[end_token:new_end].text.split()
                            )
                            end_token = new_end

                        # Break if max_context is reached
                        if context_words >= max_context:
                            break

                        # Break if max_context cannot be reached
                        if new_end == len_doc_text and start_token <= 0:
                            break

                    context = doc_text[
                        max(0, start_token) : min(len_doc_text, end_token)
                    ].text
                    contexts_list.append(context)

    return contexts_list


def concatenate_short_sentences(sentences: list[str], len_sentence_threshold: int) -> list[str]:
    modified_sentences: list[str] = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        word_count = len(sentence.split())

        # Check if the sentence is shorter than the threshold
        while word_count < len_sentence_threshold:
            i += 1
            # Break the loop if we reach the end of the list
            if i >= len(sentences):
                break
            next_sentence = sentences[i]
            sentence += " " + next_sentence
            word_count += len(next_sentence.split())

        modified_sentences.append(sentence)
        i += 1

    return modified_sentences


@db_cache(
    # Text (very rarely) contains some funky stuff that can not be serialized properly.
    log_error_on_unsavable_data=False,
)
def openai_embedding_cached(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    emb = OpenAIEmbeddings(model=model)
    vector: list[float] = emb.embed_query(text)
    return vector


def extract_similarity_scores(
    text: str,
    doc_question: spacy.tokens.Doc,
    event_date: str | None,
    nlp: spacy.Language,
    date: str,
    embedding_model: EmbeddingModel,
) -> List[Tuple[str, float, str]]:
    """
    Extract relevant information from website text based on a given event question.

    Args:
        text (str): The website text to extract information from.
        event_date (str): Event date in year-day-month format.
        nlp: The spaCy NLP model.
        date (str): The release and modification dates of the website.

    Returns:
        List[Tuple[str, float, str]]: List of tuples containing the extracted sentences, their similarity scores, and release dates.
    """

    # Constants for sentence length and number thresholds
    len_sentence_threshold = 10
    num_sentences_threshold = 1000
    sentences = []
    event_date_sentences: list[str] = []
    seen = set()

    # Truncate text for performance optimization
    text = text[:50000]

    # Apply NLP pipeline to text
    doc_text = nlp(text)

    # Extract unique sentences
    for sent in doc_text.sents:
        sentence_text = sent.text
        if (
            len(sentence_text.split()) >= len_sentence_threshold
            and sentence_text not in seen
        ):
            sentences.append(sentence_text)
            seen.add(sentence_text)

    ## Temporarily deactivated: News sites with a lot of date occurrences lead to false positives
    ## The embedding model is not advanced enough
    # Extract contextual sentences around event date occurrences within too short sentences
    # if event_date is not None:
    #     event_date_sentences.extend(
    #         get_context_around_isolated_event_date(
    #             doc_text, event_date, len_sentence_threshold, max_context=50
    #         )
    #     )
    # sentences.extend(event_date_sentences)

    if not sentences:
        return []

    # Concatenate short sentences
    sentences = concatenate_short_sentences(sentences, len_sentence_threshold)

    # Limit the number of sentences for performance optimization
    sentences = sentences[:num_sentences_threshold]

    # Encode sentences using an embedding model
    similarities = par_map(
        sentences, 
        lambda sentence: (
            doc_question.similarity(nlp(sentence)) if embedding_model == EmbeddingModel.spacy 
            else cosine_similarity(
                [openai_embedding_cached(sentence)], 
                [openai_embedding_cached(doc_question.text)]
            )[0][0] if embedding_model == EmbeddingModel.openai 
            else None
        )
    )

    # Create tuples and store them in a list
    sentence_similarity_date_tuples = [
        (sentence, similarity, date)
        for sentence, similarity in zip(sentences, similarities)
        if similarity > 0.4
    ]

    return sentence_similarity_date_tuples


def get_date(soup: BeautifulSoup) -> str:
    """
    Retrieves the release and modification dates from the soup object containing the HTML tree.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object for the webpage.

    Returns:
        str: A string representing the release and modification dates.
    """

    release_date = "unknown"
    modified_date = "unknown"

    # Search for an update or modified date in the meta tags
    for name in UPDATE_DATE_NAMES:
        meta_tag = soup.find("meta", {"name": name}) or soup.find(
            "meta", {"property": name}
        )
        if meta_tag:
            modified_date = meta_tag.get("content", "")
            break

    # If not found, then look for release or publication date
    for name in RELEASE_DATE_NAMES:
        meta_tag = soup.find("meta", {"name": name}) or soup.find(
            "meta", {"property": name}
        )
        if meta_tag:
            release_date = meta_tag.get("content", "")
            break

    ## Temporarily deactivated
    # # Fallback to using the first time tag if neither release nor modified dates are found
    # if release_date == "unknown" and modified_date == "unknown":
    #     time_tag = soup.find("time")
    #     if time_tag:
    #         release_date = time_tag.get("datetime", "")

    return f"({release_date}, {modified_date})"


def extract_sentences(
    html: str,
    doc_question: spacy.tokens.Doc,
    event_date: str | None,
    nlp: spacy.Language,
    embedding_model: EmbeddingModel,
) -> List[Tuple[str, float, str]]:
    """
    Extract relevant information from HTML string.

    Args:
        html (str): The HTML content to extract text from.
        event_date (str): Event date in year-month-day format.
        nlp: NLP object for additional text processing.

    Raises:
        ValueError: If the HTML content is empty.
        ValueError: If the release or update date could not be extracted from the HTML.

    Returns:
        List[Tuple[str, float, str]]: List of tuples containing the extracted sentences, their similarity scores, and release dates.
    """

    if not html:
        raise ValueError("HTML is empty.")

    soup = BeautifulSoup(html, "html.parser")

    # Get the date of the website
    date = get_date(soup)
    if date is None:
        raise ValueError("Could not extract release or update date from HTML.")

    # Remove unnecessary tags to clean up text
    for element in soup(HTML_TAGS_TO_REMOVE):
        element.replace_with(NavigableString(" "))

    # Extract and clean text
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ". ".join(chunk for chunk in chunks if chunk)
    text = re.sub(r"\.{2,}", ".", text)

    # Get List of (sentence, similarity, date) tuples
    similarity_scores = extract_similarity_scores(
        text=text,
        doc_question=doc_question,
        event_date=event_date,
        nlp=nlp,
        date=date,
        embedding_model=embedding_model,
    )

    if not similarity_scores:
        return []

    return similarity_scores


def process_in_batches(
    urls: List[str], batch_size: int = 15, timeout: int = 10
) -> Generator[List[Tuple[Future[requests.models.Response], str]], None, None]:
    """
    Process URLs in batches using a generator and thread pool executor.

    Args:
        urls (List[str]): List of URLs to process.
        batch_size (int, optional): Size of the processing batch_size. Default is 5.
        timeout (int, optional): Timeout for each request in seconds. Default is 10.

    Raises:
        ValueError: If the batch_size is less than or equal to zero.
        ValueError: If the timeout is less than or equal to zero.

    Yields:
        List[Tuple[Future, str]]: List containing Future objects and URLs for each batch.
    """

    if batch_size <= 0:
        raise ValueError("The 'batch_size' size must be greater than zero.")

    if timeout <= 0:
        raise ValueError("The 'timeout' must be greater than zero.")

    session = Session()
    session.max_redirects = 5

    # User-Agent headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0"
    }
    session.headers.update(headers)

    # Using THREADPOOL to execute requests in parallel
    # Loop through the URLs in batch_size of size 'batch_size'
    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]

        # Submit the batch of URLs for processing
        futures = []
        for url in batch:
            try:
                # Submit a HEAD request to the url and check Content-Type
                head_future = THREADPOOL.submit(
                    session.head,
                    url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True,
                )
                head_response = head_future.result()
                if "text/html" not in head_response.headers.get("Content-Type", ""):
                    continue
                else:
                    # Submit a GET request to the url
                    futures.append(
                        (
                            THREADPOOL.submit(
                                session.get, url, headers=headers, timeout=timeout
                            ),
                            url,
                        )
                    )
            except Exception as e:
                logger.warning(f"An error occurred: {e}")

        yield futures


def extract_and_sort_sentences(
    urls: List[str],
    event_question: str,
    nlp: spacy.Language,
    embedding_model: EmbeddingModel,
) -> List[Tuple[str, float, str]]:
    """
    Extract texts from a list of URLs using Spacy models.

    Args:
        urls (List[str]): List of URLs to extract text from.
        event_question (str): Event-related question for text extraction.

    Raises:
        ValueError: If the event date could not be extracted from the event question.
        Timeout: If the request timed out.

    Returns:
        List[Tuple[str, float, str]]: List of tuples containing the extracted sentences, their similarity scores, and release dates.
    """

    # Initialize empty list for storing extracted sentences along with their similarity scores and release dates
    all_sentences = []

    # Process the event question with spacy
    doc_question = nlp(event_question)
    event_date = extract_event_date(doc_question)

    # Process URLs in batches
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            try:
                result = future.result()
                if result.status_code != 200:
                    del result
                    continue
                # Extract relevant information for the event question
                extracted_sentences = extract_sentences(
                    html=result.text,
                    doc_question=doc_question,
                    event_date=event_date,
                    nlp=nlp,
                    embedding_model=embedding_model,
                )

                # Delete the result object to free memory
                del result

                # Append the extracted text if available and increment the count
                if extracted_sentences:
                    all_sentences.extend(extracted_sentences)

            except requests.exceptions.Timeout:
                logger.warning(f"Request for {url} timed out.")

            except Exception as e:
                logger.warning(f"An error occurred: {e}")

    all_sentences.sort(
        key=lambda x: x[1], reverse=True
    )  # Assuming the second element is the similarity score

    return all_sentences


def join_and_group_sentences(
    sentences: List[Tuple[str, float, str]], max_words: int
) -> str:
    """
    Join the sentences and group them by date.

    Args:
        sentences (List[Tuple[str, float, str]]): List of tuples containing the extracted sentences, their similarity scores, and release dates.
        max_words (int): Maximum number of words allowed for the output summary.

    Returns:
        str: The joined sentences grouped by date.
    """
    # Initialize final output string and word count
    final_output = ""
    current_word_count = 0

    # Initialize a list to hold the sentences that will be included in the final output
    filtered_sentences = []

    # Filter sentences based on word count
    for sentence, _, date in sentences:
        additional_word_count = len(sentence.split())
        if current_word_count + additional_word_count <= max_words:
            filtered_sentences.append((sentence, date))
            current_word_count += additional_word_count
        else:
            break

    # Sort filtered_sentences by date for grouping
    filtered_sentences.sort(key=itemgetter(1))

    # Group by date and iterate
    for date, group in groupby(filtered_sentences, key=itemgetter(1)):
        sentences_group = [sentence for sentence, _ in group]
        concatenated_sentences = " | ".join(sentences_group)

        # Formatting the string as per your requirement
        formatted_string = f"- {date}:{concatenated_sentences}\n\n"

        # Add this formatted string to the final output
        final_output += formatted_string

    return final_output


@observe()
def fetch_additional_information(
    event_question: str,
    max_add_words: int,
    nlp: spacy.Language,
    agent: Agent,
    embedding_model: EmbeddingModel,
) -> str:
    """
    Get urls from a web search and extract relevant information based on an event question.

    Args:
        event_question (str): The question related to the event.
        max_add_words (int): The maximum number of words allowed for additional information.
        agent (Agent): PydanticAI agent.

    Returns:
        str: The relevant information fetched from all the URLs concatenated.
    """

    # Create URL query prompt
    url_query_prompt = URL_QUERY_PROMPT.format(event_question=event_question)

    # # Perform moderation check
    # moderation_result = client.moderations.create(url_query_prompt)
    # if moderation_result["results"][0]["flagged"]:
    #     # return empty additional information if the prompt is flagged
    #     return ""

    # Register system prompt if not already set on the agent
    if not agent._system_prompts and not agent._system_prompt_functions and not agent._system_prompt_dynamic_functions:
        agent.system_prompt()(lambda: "You are a helpful assistant.")

    response = agent.run_sync(url_query_prompt).data

    # Parse the response content
    try:
        json_data = json.loads(clean_completion_json(response))
    except json.decoder.JSONDecodeError as e:
        raise UnexpectedModelBehavior(f"The response from {agent=} could not be parsed as JSON: {response=}") from e

    # Get URLs from queries
    urls = get_urls_from_queries(
        json_data["queries"],
    )

    # Extract relevant sentences from URLs
    relevant_sentences_sorted = extract_and_sort_sentences(
        urls=urls,
        event_question=event_question,
        nlp=nlp,
        embedding_model=embedding_model,
    )

    # Join the sorted sentences and group them by date
    additional_informations = join_and_group_sentences(
        relevant_sentences_sorted, max_add_words
    )

    return additional_informations


@observe()
def research(
    prompt: str,
    max_tokens: int | None = None,
    agent: Agent | None = None,
    embedding_model: EmbeddingModel = EmbeddingModel.spacy,
) -> str:
    prompt = f"\"{prompt}\""
    max_compl_tokens =  max_tokens or DEFAULT_OPENAI_SETTINGS["max_compl_tokens"]
    agent = agent or Agent(model="gpt-3.5-turbo", model_settings=ModelSettings(temperature=DEFAULT_OPENAI_SETTINGS["temperature"]))

    # Load the spacy model
    nlp = spacy.load("en_core_web_md")

    # Extract the event question from the prompt
    event_question = check_not_none(re.search(r"\"(.+?)\"", prompt)).group(1)
    if not event_question:
        raise ValueError("No event question found in prompt.")

    # Get the tiktoken base encoding
    assert agent.model is not None
    enc = tiktoken.encoding_for_model(agent.model if isinstance(agent.model, str) else agent.model.model_name)

    # Calculate the maximum number of tokens and words that can be consumed by the additional information string
    max_add_tokens = get_max_tokens_for_additional_information(
        max_compl_tokens=max_compl_tokens,
        prompt=prompt,
        enc=enc,
    )
    max_add_words = int(max_add_tokens * 0.75)

    # Fetch additional information
    additional_information = fetch_additional_information(
        event_question=event_question,
        agent=agent,
        nlp=nlp,
        max_add_words=max_add_words,
        embedding_model=embedding_model,
    )

    # Truncate additional information to stay within the chat completion token limit of 4096
    additional_information = truncate_additional_information(
        additional_information,
        max_add_tokens,
        enc=enc,
    )

    # Spacy loads ~500MB into memory. Free it with force.
    del nlp
    gc.collect()
    
    return additional_information


@observe()
def make_prediction(
    prompt: str,
    additional_information: str,
    agent: Agent | None,
    include_reasoning: bool = False,
) -> Prediction:
    agent = agent or Agent(model="gpt-3.5-turbo-0125", model_settings=ModelSettings(temperature=0.7))
    
    current_time_utc = datetime.now(timezone.utc)
    formatted_time_utc = current_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"

    field_descriptions = FIELDS_DESCRIPTIONS.copy()
    if not include_reasoning:
        field_descriptions.pop("reasoning")
    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt,
        additional_information=additional_information,
        n_fields=len(field_descriptions),
        fields_list=list_to_list_str(list(field_descriptions)),
        fields_description=fields_dict_to_bullet_list(field_descriptions),
        timestamp=formatted_time_utc,
    )
    result = agent.run_sync(prediction_prompt)

    completion = result.data

    logprobs = None
    messages = result.all_messages()
    if messages and hasattr(messages[-1], 'vendor_details'):
        vendor_details = messages[-1].vendor_details
        if vendor_details:
            logprobs = vendor_details.get("logprobs")
    
    logger.info(f"Completion: {completion}")
    completion_clean = clean_completion_json(completion)
    logger.info(f"Completion cleaned: {completion_clean}")

    try:
        response: Prediction = json.loads(completion_clean)
    except json.decoder.JSONDecodeError as e:
        raise UnexpectedModelBehavior(f"The response from {agent=} could not be parsed as JSON: {completion_clean=}") from e
    
    if logprobs:
        response['logprobs'] = LogprobsParser(skip_fields = ["reasoning"]).parse_logprobs(logprobs, Prediction) # type: ignore
    
    return response


@observe()
def make_prediction_categorical(
    prompt: str,
    possible_outcomes: Sequence[str],
    additional_information: str,
    agent: Agent | None,
    include_reasoning: bool = False,
) -> CategoricalPrediction:
    agent = agent or Agent(model="gpt-3.5-turbo-0125", model_settings=ModelSettings(temperature=0.7))
    
    current_time_utc = datetime.now(timezone.utc)
    formatted_time_utc = current_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"

    field_descriptions = FIELDS_DESCRIPTIONS_CATEGORICAL.copy()
    if not include_reasoning:
        field_descriptions.pop("reasoning")
    prediction_prompt = PREDICTION_PROMPT_CATEGORICAL.format(
        user_prompt=prompt,
        possible_outcomes=possible_outcomes,
        additional_information=additional_information,
        n_fields=len(field_descriptions),
        fields_list=list_to_list_str(list(field_descriptions)),
        fields_description=fields_dict_to_bullet_list(field_descriptions),
        timestamp=formatted_time_utc,
    )
    result = agent.run_sync(prediction_prompt)

    completion = result.data
    logger.info(f"Completion: {completion}")
    completion_clean = clean_completion_json(completion)
    logger.info(f"Completion cleaned: {completion_clean}")

    try:
        response: CategoricalPrediction = json.loads(completion_clean)
    except json.decoder.JSONDecodeError as e:
        raise UnexpectedModelBehavior(f"The response from {agent=} could not be parsed as JSON: {completion_clean=}") from e

    return response

def avg(key: str, parsed: list[dict[str, Any]]) -> float:
    vals = [p[key] for p in parsed if key in p]
    return float(sum(vals) / len(vals)) if vals else float("nan")

@observe()
def make_prediction_scalar(
    prompt: str,
    market_upper_bound: Wei,
    market_lower_bound: Wei,
    additional_information: str,
    agent: Agent | None,
    include_reasoning: bool = False,
) -> ScalarPrediction:
    agent = agent or Agent(model="gpt-3.5-turbo-0125", model_settings=ModelSettings(temperature=0.7))
    
    current_time_utc = datetime.now(timezone.utc)
    formatted_time_utc = current_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"

    field_descriptions = FIELDS_DESCRIPTIONS_SCALAR.copy()
    if not include_reasoning:
        field_descriptions.pop("reasoning")
    prediction_prompt = PREDICTION_PROMPT_SCALAR.format(
        user_prompt=prompt,
        market_upper_bound=market_upper_bound,
        market_lower_bound=market_lower_bound,
        additional_information=additional_information,
        n_fields=len(field_descriptions),
        fields_list=list_to_list_str(list(field_descriptions)),
        fields_description=fields_dict_to_bullet_list(field_descriptions),
        timestamp=formatted_time_utc,
    )
    result = agent.run_sync(prediction_prompt)
    completion = result.data

    if agent.model and hasattr(agent.model, "_n") and agent.model._n > 1:
        jsons = re.findall(r"\{[^{}]*\}", result.data, flags=re.S)

        parsed: list[dict[str, Any]] = []
        for block in jsons:
            try:
                parsed.append(json.loads(block))
            except json.JSONDecodeError:
                continue          # silently drop malformed blocks

        responses: ScalarPrediction = {
            "scalar_value": Wei(int(avg("scalar_value", parsed))),
            "confidence":   avg("confidence", parsed),
            "info_utility": avg("info_utility", parsed),
            "upperBound":   market_upper_bound,
            "lowerBound":   market_lower_bound,
            "reasoning":    "\n\n---\n\n".join(p.get("reasoning", "") for p in parsed if p.get("reasoning")),
            "logprobs":     [lp for p in parsed for lp in p.get("logprobs", [])] or None,
        }
        return responses

    else:
        completion = result.data
        logger.info(f"Completion: {completion}")
        completion_clean = clean_completion_json(completion)
        logger.info(f"Completion cleaned: {completion_clean}")

        try:
            response: ScalarPrediction = json.loads(completion_clean)
        except json.decoder.JSONDecodeError as e:
            raise UnexpectedModelBehavior(f"The response from {agent=} could not be parsed as JSON: {completion_clean=}") from e

        response['upperBound'] = market_upper_bound
        response['lowerBound'] = market_lower_bound

    return response

def clean_completion_json(completion: str) -> str:
    """
    Cleans completion JSON in form of a string:

    ```json
    {
        ...
    }
    ```

    into just { ... }
    ```
    """
    completion = completion.strip().replace("\n", " ")
    completion = " ".join(completion.split())
    start_index = completion.find("{")
    end_index = completion.rfind("}")
    completion = completion[start_index : end_index + 1]
    return completion
