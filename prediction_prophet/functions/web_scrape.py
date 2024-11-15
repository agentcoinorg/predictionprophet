import logging
from markdownify import markdownify
import requests
from bs4 import BeautifulSoup
from requests import Response
import tenacity
from datetime import timedelta
from prediction_market_agent_tooling.tools.caches.db_cache import db_cache


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1), reraise=True)
def fetch_html(url: str, timeout: int) -> Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    return response

@db_cache(max_age=timedelta(days=1), ignore_args=["timeout"])
def web_scrape_strict(url: str, timeout: int = 10) -> str:
    response = fetch_html(url=url, timeout=timeout)

    if 'text/html' in response.headers.get('Content-Type', ''):
        soup = BeautifulSoup(response.content, "html.parser")
        
        [x.extract() for x in soup.findAll('script')]
        [x.extract() for x in soup.findAll('style')]
        [x.extract() for x in soup.findAll('noscript')]
        [x.extract() for x in soup.findAll('link')]
        [x.extract() for x in soup.findAll('head')]
        [x.extract() for x in soup.findAll('image')]
        [x.extract() for x in soup.findAll('img')]
        
        text: str = soup.get_text()
        text = markdownify(text)
        text = "  ".join([x.strip() for x in text.split("\n")])
        text = " ".join([x.strip() for x in text.split("  ")])
        
        return text
    else:
        print("Non-HTML content received")
        logging.warning("Non-HTML content received")
        return ""

def web_scrape(url: str, timeout: int = 10) -> str:
    """
    Do not throw if the HTTP request fails.
    """
    try:
        return web_scrape_strict(url=url, timeout=timeout)
    except requests.RequestException as e:
        print(f"HTTP request failed: {e}")
        logging.warning(f"HTTP request failed: {e}")
        return ""
