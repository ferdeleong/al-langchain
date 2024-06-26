from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI


from dotenv import load_dotenv

import os

load_dotenv()

os.getenv("OPENAI_API_KEY")
os.getenv("OPENAI_PROJECT_ID")


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = "You are an assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

response = chat(chat_prompt.format_prompt(movie_title="Inception").to_messages())

print(response.content)

# Initialize language model
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
# document_loader = PyPDFLoader(file_path="path/to/your/pdf/file.pdf")
# document = document_loader.load()

# Summarize the document
# summary = summarize_chain(document)
# print(summary['output_text'])

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])

llm = OpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

chain.run("what is the meaning of life?")
