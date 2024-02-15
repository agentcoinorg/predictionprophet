"""
PYTHONPATH=. streamlit run scripts/agent_app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, isntead of just this one.
"""
import inspect
import typing as t
import streamlit as st
from enum import Enum 
from prediction_market_agent_tooling.benchmark.utils import EvaluatedQuestion, get_markets, MarketSource
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from evo_researcher.benchmark.agents import AGENTS

SENTINTEL = object()

st.set_page_config(layout="wide")

st.title("Agent's decision-making process")

# Fetch markets from the selected market source.
market_source = MarketSource(st.selectbox("Select a market source", [market_source.value for market_source in MarketSource]))
markets = get_markets(42, market_source)

# Select an agent from the list of available agents.
agent_class_names = st.multiselect("Select agents", [agent_class.__name__ for agent_class in AGENTS]) 
if not agent_class_names:
    st.warning("Please select at least one agent.")
    st.stop()

# Duplicate the classes if we want to see the same agent, but with a different config.
with st.expander("Duplicate agents", expanded=False):
    st.write("Optionally, you can duplicate the number of times class is selected. This is useful if you want to compare the same agent with different configurations.")
    agent_name_to_n_times: dict[str, int] = {agent_name: int(st.number_input(f"Number of times to duplicate {agent_name}", value=1)) for agent_name in agent_class_names}

# Get the agent classes from the names.
agent_classes: list[t.Type[AbstractBenchmarkedAgent]] = []
for agent_class in AGENTS:
    if agent_class.__name__ in agent_class_names:
        agent_classes.extend([agent_class for _ in range(agent_name_to_n_times[agent_class.__name__])])

# Ask the user to provide a question.
custom_question_input = st.checkbox("Provide a custom question", value=False)
question = (st.text_input("Question") if custom_question_input else st.selectbox("Select a question", [m.question for m in markets]))
if not question:
    st.warning("Please enter a question.")
    st.stop()

do_question_evaluation = st.checkbox("Enable question evaluation step", value=False)

# Show the agent's titles.
for idx, (column, agent_class) in enumerate(zip(st.columns(len(agent_classes)), agent_classes)):
    column.write(f"## {agent_class.__name__} {idx}")

agents: list[AbstractBenchmarkedAgent] = []

with st.expander("Show agent's parameters", expanded=False):
    for idx, (column, agent_class) in enumerate(zip(st.columns(len(agent_classes)), agent_classes)):
        # Inspect the agent's __init__ method to get arguments it accepts, 
        # ask the user to provide values for these arguments, 
        # and fill in defaults where possible.
        inspect_class_init = inspect.getfullargspec(agent_class.__init__)
        class_arg_to_default_value = {
            arg_name: arg_default
            for arg_name, arg_default in zip(
                # Only last arguments can have a default value, so we need to slice the args list.
                inspect_class_init.args[-(len(inspect_class_init.defaults or [])):], 
                inspect_class_init.defaults or [],
            )
        }
        default_arguments = {
            "model": "gpt-3.5-turbo-0125",
            "temperature": 0.0,
        }
        class_inputs: dict[str, t.Any] = {}
        for arg_name in inspect_class_init.args:
            # Skip these, no need to ask the user for them.
            if arg_name in ("self", "agent_name", "max_workers"):
                continue
            # We need SENTINEL to differentiate between not having the default value and when the default value is None.
            default_value = class_arg_to_default_value.get(arg_name, default_arguments.get(arg_name, SENTINTEL))
            input_type = type(default_value)
            class_inputs[arg_name]  = (
                # Show checkbox for booleans.
                column.checkbox(arg_name, default_value if default_value is not SENTINTEL else False, key=f"{idx}-{arg_name}") 
                if input_type == bool 
                else
                # Number input for numbers.
                column.number_input(arg_name, default_value if default_value is not SENTINTEL else 0, key=f"{idx}-{arg_name}")
                if input_type in (int, float)
                else 
                # Convert strings to Enum, if the default value is an Enum.
                input_type(str(column.text_input(arg_name, default_value, key=f"{idx}-{arg_name}")).replace(f"{default_value.__class__.__name__}.", ""))
                if isinstance(default_value, Enum)
                else
                # Default to just a text input.
                column.text_input(arg_name, default_value if default_value is not SENTINTEL else "", key=f"{idx}-{arg_name}")
            )

        # Instantiate the agent with the provided arguments.
        agent: AbstractBenchmarkedAgent = agent_class(**class_inputs)
        agents.append(agent)

# Use checkboxes instead of expanders, because expanders don't work inside of columns.
show_evaluation = st.checkbox("Show evaluation", value=False)
show_research = st.checkbox("Show research", value=False)

for idx, (column, agent) in enumerate(zip(st.columns(len(agents)), agents)):
    # Optionally, do the evaluation.
    if do_question_evaluation:
        with st.spinner("Evaluating..."):
            evaluated = agent.evaluate(market_question=question) 
        if show_evaluation:
            column.markdown(f"""## Evaluation

Is predictable: `{evaluated.is_predictable}`
""")
        if not evaluated.is_predictable:
            column.error("The agent thinks this question is not predictable.")
            if not column.checkbox("Show research and prediction anyway"):
                st.stop()
    else:
        evaluated = EvaluatedQuestion(question=question, is_predictable=True)

    # Do the research and prediction.
    with st.spinner("Researching..."):
        researched = agent.research(market_question=question)
    if not researched:
        column.error("No research report was generated.")
        st.stop()
    if show_research:
        column.markdown(f"""## Research report
                    
```
{researched}     
```
""")

    with st.spinner("Predicting..."):
        prediction = agent.predict(market_question=question, researched=researched, evaluated=evaluated)
    with column.expander("Show agent's prediction", expanded=False):
        column.markdown(f"""## Prediction

```       
{prediction.outcome_prediction}
```
""")
    if not prediction:
        column.error("No prediction was generated.")
        st.stop()
