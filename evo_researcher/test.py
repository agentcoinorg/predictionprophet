import click

from dotenv import load_dotenv
from autogen import config_list_from_json

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from evo_researcher.autonolas.research import research as research_autonolas
from evo_researcher.agents.planner import create_planner
from evo_researcher.agents.researcher import create_researcher
from evo_researcher.functions.web_scrape import web_scrape
from evo_researcher.functions.web_research import web_search

load_dotenv()

def generate_queries(goal: str):
    planning_prompt_template = """
    You are a professional researcher. Your goal is to prepare a research plan for {goal}.
    
    The plan will consist of multiple web searches separated by commas.
    Return ONLY the web searches, separated by commas.
    
    Keep it to a max of 3 searches.
    """
    planning_prompt = ChatPromptTemplate.from_template(template=planning_prompt_template)
    
    plan_searches_chain = (
        planning_prompt |
        ChatOpenAI(model="gpt-4-1106-preview") |
        CommaSeparatedListOutputParser()
    )

    web_searches = plan_searches_chain.invoke({
        "goal": goal
    })

    return [goal] + [search.strip('\"') for search in web_searches]

def scrape_results(results: list[(str, str)]):
    scraped = []
    for (query, result) in results:
        scraped_content = web_scrape(result["url"], query, 5000)
        scraped.append({
            "query": query,
            "url": result["url"],
            "title": result["title"],
            "content": scraped_content
        })
    return scraped

def prepare_report(goal: str, scraped: list):
    evaluation_prompt_template = """
    You are a professional researcher. Your goal is to answer: '{goal}'.
    
    Here are the results of relevant web searches:
    
    {search_results}
    
    Prepare a comprehensive report that answers the question. If that is not possible,
    state why
    """
    evaluation_prompt = ChatPromptTemplate.from_template(template=evaluation_prompt_template)
    
    research_evaluation_chain = (
        evaluation_prompt |
        ChatOpenAI(model="gpt-4-1106-preview") |
        StrOutputParser()
    )
    
    response = research_evaluation_chain.invoke({
        "search_results": scraped,
        "goal": goal
    })
    
    return response

def search(queries: list[str], filter = lambda x: True):
    results = [web_search(query, max_results=3) for query in queries]

    results_with_queries = []

    # Each result will have a query associated with it
    # We only want to keep the results that are unique
    for i in range(len(results)):
        for result in results[i]:
            if result["url"] not in [existing_result["url"] for (_,existing_result) in results_with_queries]:
                if filter(result):
                  results_with_queries.append((queries[i], result))

    return results_with_queries

def research(goal: str):
    queries = generate_queries(goal)

    results = search(queries, lambda result: not result["url"].startswith("https://www.youtube"))

    scraped = scrape_results(results)
    report = prepare_report(goal, scraped) 

    print(report)

research("Will Twitter implement a new misinformation policy before the 2024 elections?")