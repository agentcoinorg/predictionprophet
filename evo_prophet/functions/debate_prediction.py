import datetime
import openai


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
Let's think through this step by step. Give arguments for your prediction
"""

CONSENSUS_PROMPT = """
These are the predictions to '{user_prompt}' from other agents: {past_results}
Using the opinion of other agents as additional advice, give an updated response. Think through this step by step
"""

FINALIZATION_PROMPT = PREDICTION_PROMPT + """
PAST AGENTS RESPONSES:
```
{past_results}
```
Let's think through this step by step.
"""


def debate_prediction(prompt: str, additional_information: str, agents: int = 2, debate_rounds: int = 2, model: str = "gpt-4-1106-preview"):
    formatted_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds') + "Z"
    client = openai.OpenAI()

    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt,
        additional_information=additional_information,
        timestamp=formatted_time_utc
    )

    def create_prediction(messages):
        return client.chat.completions.create(
            messages=messages,
            model=model,
        )

    # Initial responses from two agents
    agent_responses = [
        create_prediction([{"role": "system", "content": prediction_prompt}]) for _ in range(agents)
    ]

    print([a.choices[0].message.content for a in agent_responses])
    print("\n\n-------------------\n\n")

    # Consensus responses
    consensus_responses = []
    for i in range(debate_rounds):
        past_response = agent_responses[1 - i].choices[0].message.content
        consensus_prompt = CONSENSUS_PROMPT.format(past_results=past_response, user_prompt=prompt)
        consensus_responses.append(
            create_prediction([
                {"role": "system", "content": prediction_prompt},
                agent_responses[i].choices[0].message,
                {"role": "user", "content": consensus_prompt}
            ])
        )

    print([a.choices[0].message.content for a in consensus_responses])
    print("\n\n-------------------\n\n")

    # Finalization
    finalization_prompt = FINALIZATION_PROMPT.format(
        user_prompt=prompt,
        additional_information=additional_information,
        timestamp=formatted_time_utc,
        past_results="\n\n------------\n\n".join([
            response.choices[0].message.content for response in consensus_responses
        ])
    )
    final_response = create_prediction([{"role": "system", "content": finalization_prompt}])

    print(final_response.choices[0].message.content)