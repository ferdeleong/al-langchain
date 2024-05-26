
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from dotenv import load_dotenv

import os

load_dotenv()

os.getenv("OPENAI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)
# add to messages
messages.append(prompt)

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=api_key)

response = llm(messages)