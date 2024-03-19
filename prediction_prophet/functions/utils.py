import tiktoken
import os
from datetime import datetime
from typing import NoReturn, Type, TypeVar, Optional
from googleapiclient.discovery import build
from prediction_prophet.functions.cache import persistent_inmemory_cache

T = TypeVar("T")


def check_not_none(
    value: Optional[T],
    msg: str = "Value shouldn't be None.",
    exp: Type[ValueError] = ValueError,
) -> T:
    """
    Utility to remove optionality from a variable.

    Useful for cases like this:

    ```
    keys = pma.utils.get_keys()
    pma.omen.omen_buy_outcome_tx(
        from_addres=check_not_none(keys.bet_from_address),  # <-- No more Optional[HexAddress], so type checker will be happy.
        ...,
    )
    ```
    """
    if value is None:
        should_not_happen(msg=msg, exp=exp)
    return value


def should_not_happen(
    msg: str = "Should not happen.", exp: Type[ValueError] = ValueError
) -> NoReturn:
    """
    Utility function to raise an exception with a message.

    Handy for cases like this:

    ```
    return (
        1 if variable == X
        else 2 if variable == Y
        else 3 if variable == Z
        else should_not_happen(f"Variable {variable} is uknown.")
    )
    ```

    To prevent silent bugs with useful error message.
    """
    raise exp(msg)

    
def trim_to_n_tokens(content: str, n: int, model: str) -> str:
    encoder = tiktoken.encoding_for_model(model)
    return encoder.decode(encoder.encode(content)[:n])


@persistent_inmemory_cache
def url_is_older_than(url: str, older_than: datetime) -> bool:
    service = build("customsearch", "v1", developerKey=os.environ["GOOGLE_SEARCH_API_KEY"])
    date_restrict = f"d{(datetime.now().date() - older_than.date()).days}"  # {d,w,m,y}N to restrict the search to the last N days, weeks, months or years.

    search = (
        service
        .cse()
        .list(
            # Possible options: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
            q=url,
            cx=os.environ["GOOGLE_SEARCH_ENGINE_ID"],
            num=1,
            dateRestrict=date_restrict,  
            # We can also restrict the search to a specific date range, but it seems we can not have restricted date range + relevance sorting, so that is not useful for us.
            # sort="date:r:20000101:20230101",  #  "YYYYMMDD:YYYYMMDD"
        )
        .execute()
    )
    return True if int(search["searchInformation"]["totalResults"]) == 0 or not any(url in item["link"] for item in search["items"]) else False


def time_restrict_urls(urls: list[str], time_restriction_up_to: datetime) -> list[str]:
    restricted_urls: list[str] = []
    for url in urls:
        if url_is_older_than(url, time_restriction_up_to):
            restricted_urls.append(url)
    return restricted_urls
