from prettyformatter import pprint

from langchain.chat_models import init_chat_model

# llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
llm = init_chat_model(
    model="google/gemma-3n-e4b",
    # because LM Studio mimics OpenAI's API
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    # LM Studio accepts any string for api key
    api_key="not-needed",
)


# 👀 use a simple prompt to test; you have to wait for a long time if the answer has too many words
res = llm.invoke("give a brief intro to reactjs in less than 80 words.")
pprint(res, json=True)
