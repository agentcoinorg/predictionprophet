import inspect
import typing as t
import streamlit as st
from evo_researcher.benchmark.agents import AbstractBenchmarkedAgent, AGENTS

st.set_page_config(layout="wide")

st.title("Agent's decision-making process")

# Select an agent from the list of available agents.
agent_class_name = st.selectbox("Select an agent", [agent_class.__name__ for agent_class in AGENTS])
agent_class = next((agent_class for agent_class in AGENTS if agent_class.__name__ == agent_class_name), None)
assert agent_class is not None, f"Bug: Selected agent class {agent_class_name} not found."

# Inspect the agent's __init__ method to get arguments it accepts, 
# ask the user to provide values for these arguments, 
# and fill in defaults where possible.
inspect_class_init = inspect.getfullargspec(agent_class.__init__)
class_arg_to_default_value = {
    arg_name: arg_default
    for arg_name, arg_default in zip(
        # Only last arguments can have a default value, so we need to slice the args list.
        inspect_class_init.args[-len(inspect_class_init.defaults):], 
        inspect_class_init.defaults,
    )
}
default_arguments = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
}
with st.expander("Show agent's parameters", expanded=False):
    class_inputs = {
        arg_name: st.text_input(arg_name, class_arg_to_default_value.get(arg_name, default_arguments.get(arg_name, "")))
        for arg_name in inspect_class_init.args
        if arg_name not in ("self", "agent_name", "max_workers")
    }

# Instantiate the agent with the provided arguments.
agent: AbstractBenchmarkedAgent = agent_class(**class_inputs)

# Ask the user to provide a question.
question = st.text_input("Question")
if not question:
    st.warning("Please enter a question.")
    st.stop()

# Optionally, do the evaluation.
if st.checkbox("Enable question evaluation", value=False):
    with st.spinner("Evaluating..."):
        evaluated = agent.evaluate(market_question=question) 
    st.markdown(f"""## Evaluation

    Is predictable: `{evaluated.is_predictable}`

    Is predictable's completion: 
    ```
    {evaluated.is_predictable.completion}
    ```
    """)
    if not evaluated.is_predictable:
        st.error("The agent thinks this question is not predictable.")
        if not st.checkbox("Show research and prediction anyway"):
            st.stop()

# Do the research and prediction.
with st.spinner("Researching..."):
    researched = agent.research(market_question=question)
if not researched:
    st.error("No research report was generated.")
    st.stop()
st.markdown(f"""## Research report
            
```
{researched}     
```
""")

with st.spinner("Predicting..."):
    prediction = agent.predict(market_question=question, researched=researched)
st.markdown(f"""## Prediction
            
```
{prediction}
```
""")
if not prediction:
    st.error("No prediction was generated.")
    st.stop()
