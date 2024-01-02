import logging
import os
from markdownify import markdownify
import requests
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient
import base64 

def web_scrape(url: str) -> tuple[str, str]:
    cached = read_from_cache(url)
    if cached is not None:
        print(f"-- Using cached {url} --")
        return (cached, url)
 
    print(f"-- Scraping {url} --")
    api_key = os.getenv("SCRAPINGBEE_API_KEY")
    client = ScrapingBeeClient(api_key=api_key)

    try:
        response = client.get(url=url)

        if 'text/html' in response.headers.get('Content-Type', ''):
            soup = BeautifulSoup(response.content, "html.parser")
            
            [x.extract() for x in soup.findAll('script')]
            [x.extract() for x in soup.findAll('style')]
            [x.extract() for x in soup.findAll('head')]
            
            text = soup.get_text()
            text = markdownify(text)
            text = "  ".join([x.strip() for x in text.split("\n")])
            
            write_to_cache(url, text)
            return (text, url)
        else:
            logging.warning("Non-HTML content received")
            return ("", url)

    except requests.RequestException as e:
        logging.error(f"HTTP request failed: {e}")
        return ("", url)
    
def read_from_cache(url: str) -> str | None:
    file_name = to_base64(url)

    if os.path.isfile("cache/" + file_name):
        with open("cache/" + file_name, "r") as f:
            return f.read()
    else:
        return None

def write_to_cache(url: str, content: str) -> None:
    file_name = to_base64(url)

    if not os.path.isdir("cache"):
        os.mkdir("cache")

    with open("cache/" + file_name, "w") as f:
        f.write(content)

def to_base64(url: str) -> str:
    url_bytes = url.encode("ascii") 
    base64_bytes = base64.b64encode(url_bytes) 
    return base64_bytes.decode("ascii")

def from_base64(base64_str: str) -> str:
    base64_bytes = base64_str.encode("ascii") 
    url_bytes = base64.b64decode(base64_bytes) 
    return url_bytes.decode("ascii")