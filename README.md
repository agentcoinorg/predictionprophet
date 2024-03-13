# evo.prophet
![](./content/banner.png)

## Overview

evo.prophet is an agent that specializes in making informed predictions, based on web research.

It contains three primary functions: [Research](#research-function), [Prediction](#prediction-function), [Grading](#grading-function)

Below, there's a high level explanation of their implementations, respectively.

### Research Function

The research function takes a question, like `"Will Twitter implement a new misinformation policy before the 2024 elections?"`, and will then:

1. Generate `n` web search queries
2. Re-rank the queries using an LLM call, and then select the most relevant ones
3. Search the web for each query, using [Tavily](https://tavily.com/)
4. Scrape and sanitize the content of each result's website
5. Use Langchain's `RecursiveCharacterTextSplitter` to split the content of all pages into chunks.
6. Create embeddings of all chunks, and store the source of each as metadata.
7. Iterate over the queries selected on step `2` and vector search for the most relevant chunks created in step `5`.
8. Aggregate all relevant chunks and prepare a report.

### Prediction Function

Ported implementation from [Valory mech's prediction_sentence_embeddings implementation](https://github.com/valory-xyz/mech/blob/main/packages/valory/customs/prediction_request_embedding/prediction_sentence_embedding.py).


### Grading Function

The grading function grades information using criteria from https://guides.lib.unc.edu/evaluating-info/evaluate, ignoring `usability` and `intended audience`.

Upon receiving a question like `"Will Twitter implement a new misinformation policy before the 2024 elections?"` along with complimentary information, it will:

1. Create an information evaluation plan
2. Evaluate the information according to the plan
3. Extract the scores from the evaluation

## Installation

```bash
poetry install
poetry shell
```

## Run

### Research


```bash
poetry run research "Will Twitter implement a new misinformation policy before the 2024 elections?"
```

### Predict

```bash
poetry run predict "Will Twitter implement a new misinformation policy before the 2024 elections?"
```

## Run front-end app

```bash
poetry run streamlit run ./scripts/public_agent_app.py
```

## Need Help?

Join our [Discord community](https://discord.gg/k7UCsH3ps9) for support and discussions.

[![Join us on Discord](https://invidget.switchblade.xyz/k7UCsH3ps9)](https://discord.com/invite/k7UCsH3ps9)

If you have questions or encounter issues, please don't hesitate to [create a new issue](https://github.com/polywrap/evo.prophet/issues/new) to get support.


## Ideas for future improvement

### For the researcher:

- Using LLM re-ranking, like [Cursor](https://twitter.com/amanrsanger/status/1732145826963828997?s=03) to optimize context-space and reduce noise
- Use [self-consistency](https://www.promptingguide.ai/techniques/consistency) and generate several reports and compare them to choose the best, or even merge information
- Plan research using more complex techniques like [tree of thoughts](https://arxiv.org/abs/2305.10601)
- Implement a research loop, where research is performed and then evaluated. If the evaluation scores are under certain threshold, re-iterate to gather missing information or different sources, etc.
- Perform web searches under different topic or category focuses like [Tavily](https://app.tavily.com/home) does. For example, some questions benefit more from a "social media focused" research: gathering information from twitter threads, blog articles. Others benefit more from prioritizing scientific papers, institutional statements, and so on.
- Identify strong claims and perform sub-searches to verify them. This is the basis of AI powered fact-checkers like: https://fullfact.org/
- Evaluate sources credibility
- Further iterate over chunking and vector-search strategies
- Use [HyDE](https://medium.com/@juanc.olamendy/revolutionizing-retrieval-the-mastering-hypothetical-document-embeddings-hyde-b1fc06b9a6cc)

### For the information evaluator/grader

- Use [self-consistency](https://www.promptingguide.ai/techniques/consistency) to generate several scores and choose the most repeated ones.
- Enhance the evaluation and reduce its biases through the implementation of more advanced techniques, like the ones described here https://arxiv.org/pdf/2307.03025.pdf and here https://arxiv.org/pdf/2305.17926.pdf
- Further evaluate biases towards writing-style, length, among others described here: https://arxiv.org/pdf/2308.02575.pdf and mitigate them
- Evaluate using different evaluation criteria
