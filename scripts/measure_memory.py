import time
import typer
from prediction_prophet.benchmark.agents import AGENTS


def main(sleep: int = 120) -> None:
    """
    1. Decorate functions that you want to track with `@profile`
    2. (a) Run `mprof run --include-children scripts/measure_memory.py` and then plot it with `mprof plot`.
    2. (b) Or, run `python -m memory_profiler scripts/measure_memory.py` for line-by-line analysis.
    """
    questions = [
        "Will GNO hit $1000 by the end of 2024?",
        "Will GNO hit $10000 by the end of 2025?",
        "Will GNO hit $100000 by the end of 2026?",
        "Will GNO hit $1000000 by the end of 2027?",
        "Will GNO hit $10000000 by the end of 2028?",
        "Will GNO hit $100000000 by the end of 2029?",
        "Will GNO hit $1000000000 by the end of 2030?",
    ]

    print("Sleeping so we can see initial memory usage in the plot.")
    time.sleep(sleep)

    run(questions, sleep)


def run(questions: list[str], sleep: int) -> None:
    for AgentClass in AGENTS:
        agent = AgentClass(
            model="gpt-3.5-turbo-0125",  # Just some cheap model, no need for good results here.
        )

        for question in questions:
            agent.predict(question)

        print("Sleeping so we can differ it in the plot.")
        time.sleep(sleep)


if __name__ == "__main__":
    typer.run(main)
