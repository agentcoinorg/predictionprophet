import datetime
import openai

def calculate_cumulative_similarity(numbers):
    def similarity(a, b):
        return abs(a - b)
    
    cumulative_similarity = {}
    
    for i in range(len(numbers)):
        total_similarity = 0
        for j in range(len(numbers)):
            if i != j:
                total_similarity += similarity(numbers[i], numbers[j])
        cumulative_similarity[i] = total_similarity
    
    return cumulative_similarity


PREDICTION_PROMPT = """
INTRODUCTION:
Your primary task is to accurately estimate the probabilities for the outcome of a 'market question', \
found in 'USER_PROMPT'. The market question is part of a prediction market, where users can place bets on the outcomes of market questions and earn rewards if the selected outcome occurrs. The 'market question' \
in this scenario has only two possible outcomes: `Yes` or `No`. Each market has a closing date at which the outcome is evaluated. This date is typically stated within the market question.  \
The closing date is considered to be 23:59:59 of the date provided in the market question. If the event specified in the market question has not occurred before the closing date, the market question's outcome is `No`. \
If the event has happened before the closing date, the market question's outcome is `Yes`. You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your probability estimation. You must adhere to the following 'INSTRUCTIONS'.  


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'market question'.
* If the 'market question' implies more than two outcomes, output the response "Error" and halt further processing.
* When the current time {timestamp} has passed the closing date of the market and the event specified in the market question has not happened, the market question's outcome is `No` and the user who placed a bet on `No` will receive a reward.
* When the current time {timestamp} has passed the closing date of the market and the event has happened before, the market question's final outcome is `Yes` and the user who placed a bet on `yes` will receive a reward.
* Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
* The probability estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user. 
* If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
* Even if not all information might not be released today, you can assume that there haven't been publicly available updates in the meantime except for those inside ADDITIONAL_INFORMATION.
* The closer the current time `{timestamp}` is to the closing time the higher the likelyhood that the outcome of the market question will be `No`, if recent information do not clearly indicate that the event will occur before the closing date.
* If there exist recent information indicating that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.


USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

Your output response must include:

- "decision": The decision you made. Either `y` (for `Yes`) or `n` (for `No`).
- "p_yes": Probability that the market question's outcome will be `Yes`. Ranging from 0 (lowest probability) to 1 (maximum probability).
- "p_no": Probability that the market questions outcome will be `No`. Ranging from 0 (lowest probability) to 1 (maximum probability).
- "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.
* The sum of "p_yes" and "p_no" must equal 1.

Let's think through this step by step
"""

CONSENSUS_PROMPT = """
These are the solutions to the problem from other agents: {past_results}
Using the opinion of other agents as additional advice, give an updated response. Think through this step by step
"""

FINALIZATION_PROMPT = """
INTRODUCTION:
Your primary task is to accurately estimate the probabilities for the outcome of a 'market question', \
found in 'USER_PROMPT'. The market question is part of a prediction market, where users can place bets on the outcomes of market questions and earn rewards if the selected outcome occurrs. The 'market question' \
in this scenario has only two possible outcomes: `Yes` or `No`. Each market has a closing date at which the outcome is evaluated. This date is typically stated within the market question.  \
The closing date is considered to be 23:59:59 of the date provided in the market question. If the event specified in the market question has not occurred before the closing date, the market question's outcome is `No`. \
If the event has happened before the closing date, the market question's outcome is `Yes`. You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your probability estimation. You must adhere to the following 'INSTRUCTIONS'.  


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'market question'.
* If the 'market question' implies more than two outcomes, output the response "Error" and halt further processing.
* When the current time {timestamp} has passed the closing date of the market and the event specified in the market question has not happened, the market question's outcome is `No` and the user who placed a bet on `No` will receive a reward.
* When the current time {timestamp} has passed the closing date of the market and the event has happened before, the market question's final outcome is `Yes` and the user who placed a bet on `yes` will receive a reward.
* Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
* The probability estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user. 
* If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
* Even if not all information might not be released today, you can assume that there haven't been publicly available updates in the meantime except for those inside ADDITIONAL_INFORMATION.
* The closer the current time `{timestamp}` is to the closing time the higher the likelyhood that the outcome of the market question will be `No`, if recent information do not clearly indicate that the event will occur before the closing date.
* If there exist recent information indicating that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.


USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

PAST AGENTS RESPONSES:
{past_results}

Your output response must include:

- "decision": The decision you made. Either `y` (for `Yes`) or `n` (for `No`).
- "p_yes": Probability that the market question's outcome will be `Yes`. Ranging from 0 (lowest probability) to 1 (maximum probability).
- "p_no": Probability that the market questions outcome will be `No`. Ranging from 0 (lowest probability) to 1 (maximum probability).
- "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.
* The sum of "p_yes" and "p_no" must equal 1.

Let's think through this step by step
"""


def make_prediction(prompt: str, additional_information: str, model: str = "gpt-4-1106-preview"):
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
        create_prediction([{"role": "system", "content": prediction_prompt}]) for _ in range(2)
    ]
    
    print([a.choices[0].message.content for a in agent_responses])
    print("\n\n-------------------\n\n")
    
    # Consensus responses
    consensus_responses = []
    for i in range(2):
        past_response = agent_responses[1 - i].choices[0].message.content
        consensus_prompt = CONSENSUS_PROMPT.format(past_results=past_response)
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