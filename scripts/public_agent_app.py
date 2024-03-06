"""
PYTHONPATH=. streamlit run scripts/agent_app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, isntead of just this one.
"""
import streamlit as st
from evo_researcher.benchmark.agents import EvoAgent

SENTINTEL = object()

st.set_page_config(layout="wide")

st.title("Evo Predict")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input('Question', placeholder="Will Twitter implement a new misinformation policy before the end of 2024")
    submit_button = st.form_submit_button('Predict')

agent = EvoAgent(model="gpt-4-0125-preview")

if submit_button and question:
    with st.container():
        with st.spinner("Evaluating question..."):
            is_predictable = agent.is_predictable(market_question=question) 

        st.container().markdown(f"""## Evaluation\n\nIs predictable: `{is_predictable}`""")
        if not is_predictable:
            st.container().error("The agent thinks this question is not predictable.")
            st.stop()

        with st.spinner("Predicting..."):
            prediction = agent.predict(market_question=question)
        with st.container().expander("Show agent's prediction", expanded=False):
            st.container().markdown(f"""## Prediction

        ```       
        {prediction.outcome_prediction}
        ```
        """)
            if not prediction:
                st.container().error("No prediction was generated.")
                st.stop()
