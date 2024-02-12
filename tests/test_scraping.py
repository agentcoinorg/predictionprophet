from evo_researcher.functions.web_scrape import web_scrape


def test_scraping() -> None:
    result = web_scrape("https://www.statista.com/statistics/272120/revenue-of-tesla/")
    
    print(result)
    
    assert False