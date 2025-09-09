import requests
import json
from pydantic import BaseModel, Field
from typing import Optional


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


# Test basic chat with direct requests
url = "http://localhost:1234/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": "Bearer lm-studio"}

# Test basic chat
basic_data = {
    "model": "google/gemma-3n-e4b",
    "messages": [{"role": "user", "content": "Tell me a joke about cats"}],
}

try:
    response = requests.post(url, headers=headers, json=basic_data)
    if response.status_code == 200:
        result = response.json()
        print("Basic chat result:", result["choices"][0]["message"]["content"])
    else:
        print(f"Basic chat failed with status {response.status_code}: {response.text}")
except Exception as e:
    print("Basic chat failed:", e)

# Test structured output with function calling
structured_data = {
    "model": "google/gemma-3n-e4b",
    "messages": [{"role": "user", "content": "Tell me a joke about cats"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "create_joke",
                "description": "Create a joke with setup and punchline",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "setup": {
                            "type": "string",
                            "description": "The setup of the joke",
                        },
                        "punchline": {
                            "type": "string",
                            "description": "The punchline to the joke",
                        },
                        "rating": {
                            "type": "integer",
                            "description": "How funny the joke is, from 1 to 10",
                        },
                    },
                    "required": ["setup", "punchline"],
                },
            },
        }
    ],
    "tool_choice": "required",
}

try:
    response = requests.post(url, headers=headers, json=structured_data)
    if response.status_code == 200:
        result = response.json()
        tool_call = result["choices"][0]["message"]["tool_calls"][0]
        joke_args = json.loads(tool_call["function"]["arguments"])
        joke = Joke(**joke_args)
        print("Structured output result:", joke)
    else:
        print(
            f"Structured output failed with status {response.status_code}: {response.text}"
        )
except Exception as e:
    print("Structured output failed:", e)
