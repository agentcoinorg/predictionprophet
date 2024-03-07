import os
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
    
    def log(self, msg: str) -> None:
        st.write(msg)
    
    debug = info = warning = error = critical = log

st.set_page_config(layout="wide")

st.title("Evo Predict")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input('Question', placeholder="Will Twitter implement a new misinformation policy before the end of 2024")
    openai_api_key = st.text_input('OpenAI API Key', placeholder="sk-...", type="password")
    submit_button = st.form_submit_button('Predict')

logger = StreamlitLogger()
agent = EvoAgent(model="gpt-4-0125-preview", logger=logger)
tavily_api_key = os.environ.get('TAVILY_API_KEY')

if tavily_api_key == None:
    try:
        tavily_api_key = st.secrets['TAVILY_API_KEY']
    except:
        st.container().error("No Tavily API Key provided")
        st.stop()

if submit_button and question and openai_api_key:
    with st.container():
        with st.spinner("Evaluating question..."):
            is_predictable = agent.is_predictable(market_question=question) 

        st.container(border=True).markdown(f"""### Question evaluation\n\nQuestion: **{question}**\n\nIs predictable: `{is_predictable}`""")
        if not is_predictable:
            st.container().error("The agent thinks this question is not predictable.")
            st.stop()
            
        with st.spinner("Researching..."):
            with st.container(border=True):
                report = agent.research(goal=question, use_summaries=False, openai_api_key=openai_api_key, tavily_api_key=tavily_api_key)
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
