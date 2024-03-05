import click
import time
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from evo_researcher.autonolas.research import make_prediction
from evo_researcher.functions.grade_info import grade_info
from evo_researcher.functions.research import research as evo_research

load_dotenv()

def create_output_file(info: str, path: str) -> str:
    with open(path, 'w') as file:
        file.write(info)

def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

@click.group()
def cli() -> None:
    pass
    
@cli.command()
@click.argument('prompt')
@click.argument('file')
def research(
    prompt: str,
    file: str | None = None
) -> None:
    start = time.time()
    
    with get_openai_callback() as cb:
      report = evo_research(goal=prompt, model="gpt-4-1106-preview", use_summaries=False)
    
    end = time.time()
    
    print(f"\n\nTime elapsed: {end - start}\n\n{cb}\n\n")
    
    if file:
        create_output_file(report)
        print(f"Output saved to '{file}'")
        return
    
    print(f"Research results:\n\n{report}")

    
@cli.command()
@click.argument('prompt')
@click.argument('path')
def evaluate(prompt: str, path: str) -> None:
    information = read_text_file(path)
    scores = grade_info(question=prompt, information=information)
    print(scores)
    

@cli.command()
@click.argument('prompt')
@click.argument('path')
def predict(prompt: str, path: str | None = None) -> None:
    information = read_text_file(path)
    
    start = time.time()
    with get_openai_callback() as cb:
        prediction = make_prediction(prompt=prompt, additional_information=information)
    end = time.time()
    
    print(f"\n\nTime elapsed: {end - start}\n\n{cb}\n\n")
    print(prediction)
    

if __name__ == '__main__':
    cli()