import datetime
import os
from autogen import ConversableAgent
from autogen import GroupChatManager
from pydantic import SecretStr
from autogen import GroupChat
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from prediction_market_agent_tooling.tools.utils import secret_str_from_env
from prediction_market_agent_tooling.gtypes import secretstr_to_v1_secretstr

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
Let's think through this step by step. Give arguments for your prediction.
"""


EXTRACTION_PROMPT = """
You will be given information. From it, extract the JSON with the following content:

    - "decision": 'y' for 'Yes' or 'n' for 'No'.
    - "p_yes": Probability of 'Yes', from 0 to 1.
    - "p_no": Probability of 'No', from 0 to 1.
    - "confidence": Your confidence in these estimates, from 0 to 1.
    
Return only the JSON and include nothing more in your response.

Information: {prediction_summary}
"""
    
def make_debated_prediction(prompt: str, additional_information: str, max_debate_rounds: int = 1, api_key: SecretStr | None = None):
    if api_key == None:
        api_key = secret_str_from_env("OPENAI_API_KEY")
        
    formatted_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds') + "Z"
    
    agent_a = ConversableAgent(
        name=f"Predictor_Agent_A",
        system_message=PREDICTION_PROMPT.format(
            user_prompt=prompt,
            additional_information=additional_information,
            timestamp=formatted_time_utc
        ),
        llm_config={"config_list": [{"model": "gpt-4-0125-preview", "api_key": api_key.get_secret_value()}]},
        human_input_mode="NEVER"
    )
    
    agent_b = ConversableAgent(
        name=f"Predictor_Agent_B",
        system_message=PREDICTION_PROMPT.format(
            user_prompt=prompt,
            additional_information=additional_information,
            timestamp=formatted_time_utc
        ),
        llm_config={"config_list": [{"model": "gpt-4-0125-preview", "api_key": api_key.get_secret_value()}]},
        human_input_mode="NEVER"
    )
    
    
    group_chat = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=8,
        speaker_selection_method="round_robin"
    )
    
    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    )
    
    chat_result = agent_a.initiate_chat(
        group_chat_manager,
        message="Make each agent make a prediction, and make each agent defend its prediction in a debate with each other",
        summary_method="reflection_with_llm",
    )
    
    print(chat_result)
            
    subquery_generation_prompt = ChatPromptTemplate.from_template(template=EXTRACTION_PROMPT)

    subquery_generation_chain = (
        subquery_generation_prompt |
        ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=secretstr_to_v1_secretstr(api_key)) |
        StrOutputParser()
    )

    result = subquery_generation_chain.invoke({
        "prediction_summary": chat_result.summary
    })
    
    print(result)
    
    return result