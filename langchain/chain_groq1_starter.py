import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain.chat_models import init_chat_model


def main():
    # 📌 print("Hello, init chat model")
    # model = init_chat_model("qwen/qwen3-32b", model_provider="groq")
    model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

    system_template = "Translate the following from English into {language}"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"language": "Chinese", "text": "today is Friday"})

    # messages = [
    #     SystemMessage(content="Translate the following from English into Chinese"),
    #     HumanMessage(content="what day is it today?"),
    # ]
    # res = model.invoke(messages)

    res = model.invoke(prompt)
    print(res)
    print(f"Type of res: {type(res)}")
    print(f"Is instance of str: {isinstance(res, str)}")
    # print(json.dumps(res, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
