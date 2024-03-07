"""
PYTHONPATH=. streamlit run scripts/agent_app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, isntead of just this one.
"""
from typing import cast
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction
)
from evo_researcher.benchmark.logger import BaseLogger
import streamlit as st
from evo_researcher.benchmark.agents import EvoAgent

class StreamlitLogger(BaseLogger):
    logs: list[str] = []
    
    def __init__(self) -> None:
        super().__init__()
    
    def debug(self, msg: str) -> None:
        st.write(msg)
    
    def info(self, msg: str) -> None:
        st.write(msg)
    
    def warning(self, msg: str) -> None:
        st.write(msg)
    
    def error(self, msg: str) -> None:
        st.write(msg)
    
    def critical(self, msg: str) -> None:
        st.write(msg)

st.set_page_config(layout="wide")

st.title("Evo Predict")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input('Question', placeholder="Will Twitter implement a new misinformation policy before the end of 2024")
    api_key = st.text_input('OpenAI API Key', placeholder="sk-...", type="password")
    submit_button = st.form_submit_button('Predict')

logger = StreamlitLogger()
agent = EvoAgent(model="gpt-4-0125-preview", logger=logger)

if submit_button and question and api_key:
    with st.container():
        with st.spinner("Evaluating question..."):
            is_predictable = agent.is_predictable(market_question=question) 

        st.container(border=True).markdown(f"""### Question evaluation\n\nQuestion: **{question}**\n\nIs predictable: `{is_predictable}`""")
        if not is_predictable:
            st.container().error("The agent thinks this question is not predictable.")
            st.stop()
            
        with st.spinner("Researching..."):
            with st.container(border=True):
                report = agent.research(goal=question, use_summaries=False, api_key=api_key)
        with st.container().expander("Show agent's research report", expanded=False):
            st.container().markdown(f"""{report}""")
            if not report:
                st.container().error("No research report was generated.")
                st.stop()
                
        with st.spinner("Predicting..."):
            with st.container(border=True):
                prediction = agent.predict_from_research(market_question=question, research_report=report)
        with st.container().expander("Show agent's prediction", expanded=False):
            if prediction.outcome_prediction == None:
                st.container().error("The agent failed to generate a prediction")
                st.stop()
                
            outcome_prediction = cast(OutcomePrediction, prediction.outcome_prediction)
            
            st.container().markdown(f"""
        ## Prediction
        
        ### Probability
        `{outcome_prediction.p_yes * 100}%`
        
        ### Confidence
        `{outcome_prediction.confidence * 100}%`
        """)
            if not prediction:
                st.container().error("No prediction was generated.")
                st.stop()
