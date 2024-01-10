import logging
from markdownify import markdownify
import requests
from bs4 import BeautifulSoup
from requests import Response

def fetch_html(url: str, timeout: int) -> Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    return response

def web_scrape(url: str, timeout: int = 10000) -> tuple[str, str]:
    print(f"-- Scraping {url} --")
    try:
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
            
            text = soup.get_text()
            text = markdownify(text)
            text = "  ".join([x.strip() for x in text.split("\n")])
            text = " ".join([x.strip() for x in text.split("  ")])
            
            return (text, url)
        else:
            print("Non-HTML content received")
            logging.warning("Non-HTML content received")
            return ("", url)

    except requests.RequestException as e:
        print(f"HTTP request failed: {e}")
        logging.error(f"HTTP request failed: {e}")
        return ("", url)
