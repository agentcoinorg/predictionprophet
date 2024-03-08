import click
import time
from dotenv import load_dotenv
from evo_researcher.benchmark.agents import _make_prediction
from langchain_community.callbacks import get_openai_callback
from evo_researcher.functions.research import research as evo_research

load_dotenv()

def create_output_file(info: str, path: str) -> None:
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
@click.option('--file', '-f', default=None)
def research(
    prompt: str,
    file: str | None = None
) -> None:
    start = time.time()
    
    with get_openai_callback() as cb:
      report = evo_research(goal=prompt, use_summaries=False, model="gpt-4-1106-preview")
    
    end = time.time()
    
    if file:
        create_output_file(report, file)
        print(f"Output saved to '{file}'")
        print(f"\n\nTime elapsed: {end - start}\n\n{cb}\n\n")
        return
    
    print(f"Research results:\n\n{report}")
    print(f"\n\nTime elapsed: {end - start}\n\n{cb}\n\n")


@cli.command()
@click.argument('prompt')
@click.option('--path', '-p', default=None)
def predict(prompt: str, path: str | None = None) -> None:
    start = time.time()

    with get_openai_callback() as cb:
        if path:
            report = read_text_file(path)
        else:
            report = evo_research(goal=prompt, model="gpt-4-1106-preview", use_summaries=False)
        
        prediction = _make_prediction(market_question=prompt, additional_information=report, engine="gpt-4-1106-preview", temperature=0.0)

    end = time.time()
    
    print(prediction)
    print(f"\n\nTime elapsed: {end - start}\n\n{cb}\n\n")
    

if __name__ == '__main__':
    cli()