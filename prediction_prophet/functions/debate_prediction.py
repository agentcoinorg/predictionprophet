import datetime
import json
from autogen import ConversableAgent
from prediction_market_agent_tooling.benchmark.utils import (
    Prediction,
)
from prediction_prophet.benchmark.agents import completion_prediction_json_to_pydantic_model
from pydantic import SecretStr
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from prediction_market_agent_tooling.tools.utils import secret_str_from_env


PREDICTION_PROMPT = """
Your task is to determine the probability of a prediction market question being answered 'Yes' or 'No'.
Use the question provided in 'USER_PROMPT' and follow these guidelines:
* Focus on the question inside double quotes in 'USER_PROMPT'.
* The question must have only 'Yes' or 'No' outcomes. If not, respond with "Error".
* Use 'ADDITIONAL_INFORMATION' from a recent Google search for your estimation.
* Consider the market's closing date for your prediction. If the event hasn't happened by this date, the outcome is 'No'; otherwise, it's 'Yes'.
* Your estimation must be as accurate as possible to avoid financial losses.
* Evaluate recent information more heavily than older information.
* The closer the current time ({timestamp}) is to the closing date without clear evidence of the event happening, the more likely the outcome is 'No'.
* Your response should include:
    - "decision": The decision you made. Either `y` (for `Yes`) or `n` (for `No`).
    - "p_yes": Probability that the market question's outcome will be `Yes`. Ranging from 0 (lowest probability) to 1 (maximum probability).
    - "p_no": Probability that the market questions outcome will be `No`. Ranging from 0 (lowest probability) to 1 (maximum probability).
    - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.
    - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the probability estimation ranging from 0 (lowest utility) to 1 (maximum utility).
    
    Ensure p_yes + p_no equals 1.
USER_PROMPT: {user_prompt}
ADDITIONAL_INFORMATION:
```
{additional_information}
```
Let's think through this step by step.
"""

DEBATE_PREDICTION = """
For the following question: {user_prompt}; and considering the current time: {timestamp}

Given the following information:

{additional_information}

I made the following prediction:

```
{prediction_0}
```

And you made the following prediction:

```
{prediction_1}
```

Debate my prediction, considering your own prediction. Be brief, strong and critical to defend your position.
Ultimately, our objective is to reach consensus.
"""


EXTRACTION_PROMPT = """
You will be given information. From it, extract the JSON with the following content:
   - "decision": The decision you made. Either `y` (for `Yes`) or `n` (for `No`).
   - "p_yes": Probability that the market question's outcome will be `Yes`. Ranging from 0 (lowest probability) to 1 (maximum probability).
   - "p_no": Probability that the market questions outcome will be `No`. Ranging from 0 (lowest probability) to 1 (maximum probability).
   - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the probability estimation ranging from 0 (lowest utility) to 1 (maximum utility).
    
Return only the JSON and include nothing more in your response.

Information: {prediction_summary}
"""

PREDICTOR_SYSTEM_PROMPT = """
You are a critical and strong debater, information analyzer and future events predictor.

You will debate other agents's predictions. You can update your prediction if other agents
give you convincing arguments. Nonetheless, be strong in your position and argument back to defend your prediction.
"""
    
def make_debated_prediction(prompt: str, additional_information: str, api_key: SecretStr | None = None) -> Prediction:
    if api_key == None:
        api_key = secret_str_from_env("OPENAI_API_KEY")
        
    formatted_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds') + "Z"
    
    prediction_prompt = ChatPromptTemplate.from_template(template=PREDICTION_PROMPT)

    prediction_chain = (
        prediction_prompt |
        ChatOpenAI(model="gpt-4-0125-preview", api_key=api_key.get_secret_value() if api_key else None) |
        StrOutputParser()
    )

    predictions = prediction_chain.batch([{
        "user_prompt": prompt,
        "additional_information": additional_information,
        "timestamp": formatted_time_utc,
    } for _ in range(2)])
    
    agents = [
        ConversableAgent(
            name=f"Predictor_Agent_{i}",
            system_message=PREDICTION_PROMPT,
            llm_config={"config_list": [{"model": "gpt-4-0125-preview", "api_key": api_key.get_secret_value()}]},
            human_input_mode="NEVER")
        for i in range(2) ]
    
    chat_result = agents[0].initiate_chat(
        agents[1],
        message=DEBATE_PREDICTION.format(
            user_prompt=prompt,
            additional_information=additional_information,
            timestamp=formatted_time_utc,
            prediction_0=predictions[0],
            prediction_1=predictions[1],
        ),
        summary_method="reflection_with_llm",
        max_turns=3,
    )
            
    extraction_prompt = ChatPromptTemplate.from_template(template=EXTRACTION_PROMPT)

    extraction_chain = (
        extraction_prompt |
        ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key.get_secret_value() if api_key else None) |
        StrOutputParser()
    )

    result = extraction_chain.invoke({
        "prediction_summary": chat_result.summary
    })
    
    return completion_prediction_json_to_pydantic_model(json.loads(result))