from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.schema import (
  HumanMessage,
  SystemMessage
)


from dotenv import load_dotenv

import os

load_dotenv()

os.getenv("OPENAI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key)

prompt = PromptTemplate(
  input_variables=["product"],
  template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

print( chain.run("wireless headphones") )

chat = ChatOpenAI(model_name="gpt-4", temperature=0)

messages = [
	SystemMessage(content="You are a helpful assistant that translates English to French."),
	HumanMessage(content="Translate the following sentence: I love programming.")
]

batch_messages = [
  [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following sentence: I love programming.")
  ],
  [
    SystemMessage(content="You are a helpful assistant that translates French to English."),
    HumanMessage(content="Translate the following sentence: J'aime la programmation.")
  ],
]
print( chat.generate(batch_messages) )