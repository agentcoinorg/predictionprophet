import datetime
from crewai import Task
from crewai import Agent
from crewai import Crew, Process
from langchain_openai import ChatOpenAI


PREDICTION_PROMPT = """
Determine the probability of a prediction market question being answered 'Yes' or 'No'.
Use the question provided in 'USER_PROMPT' and follow these guidelines:
* Focus on the question inside double quotes in 'USER_PROMPT'.
* The question must have only 'Yes' or 'No' outcomes. If not, respond with "Error".
* Use 'ADDITIONAL_INFORMATION' from a recent Google search for your estimation.
* Consider the market's closing date for your prediction. If the event hasn't happened by this date, the outcome is 'No'; otherwise, it's 'Yes'.
* Your estimation must be as accurate as possible to avoid financial losses.
* Evaluate recent information more heavily than older information.
* The closer the current time ({timestamp}) is to the closing date without clear evidence of the event happening, the more likely the outcome is 'No'.
* Your response should include:
    - "decision": 'y' for 'Yes' or 'n' for 'No'.
    - "p_yes": Probability of 'Yes', from 0 to 1.
    - "p_no": Probability of 'No', from 0 to 1.
    - "confidence": Your confidence in these estimates, from 0 to 1.
    
    Ensure p_yes + p_no equals 1.
USER_PROMPT: {user_prompt}
ADDITIONAL_INFORMATION:
```
{additional_information}
```
Let's think through this step by step
"""


DESIRED_OUTPUT = """
JSON object with:
    - "reasoning": step by step rationale for the values.
    - "decision": 'y' for 'Yes' or 'n' for 'No'.
    - "p_yes": Probability of 'Yes', from 0 to 1.
    - "p_no": Probability of 'No', from 0 to 1.
    - "confidence": Your confidence in these estimates, from 0 to 1.
    Ensure p_yes + p_no equals 1.
"""

def debate_prediction(prompt: str, additional_information: str) -> str:
    formatted_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds') + "Z"
    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt,
        additional_information=additional_information,
        timestamp=formatted_time_utc
    )

    predictor_a = Agent(
        role='Predictor',
        goal=f'Predict the outcome of {prompt} ocurring. Debating with other agents on their predictions until consensus is found',
        verbose=True,
        memory=True,
        backstory=(
            "Predictor agent."
        ),
        tools=[],
        allow_delegation=True
    )

    predictor_b = Agent(
        role='Predictor',
        goal=f'Predict the outcome of {prompt} ocurring. Debating with other agents on their predictions until consensus is found',
        verbose=True,
        memory=True,
        backstory=(
            "Predictor agent."
        ),
        tools=[],
        allow_delegation=True
    )

    initial_prediction_task_a = Task(
        description=prediction_prompt,
        expected_output=DESIRED_OUTPUT,
        tools=[],
        agent=predictor_a
    )
    
    initial_prediction_task_b = Task(
        description=prediction_prompt,
        expected_output=DESIRED_OUTPUT,
        tools=[],
        agent=predictor_b
    )

    debate_task = Task(
        description=(
            """
            Use the prediction rationales and results of other agents as additional advice, give an updated response. Debate others until consensus is found.
            Think through this step by step
            """
        ),
        expected_output=DESIRED_OUTPUT,
        tools=[],
        agent=predictor_a
    )
    
    crew = Crew(
        agents=[predictor_a, predictor_b],
        tasks=[initial_prediction_task_a, initial_prediction_task_b, debate_task],
        manager_llm=ChatOpenAI(temperature=0, model="gpt-4"),
        process=Process.hierarchical
    )
    
    result = crew.kickoff()
    print(result)