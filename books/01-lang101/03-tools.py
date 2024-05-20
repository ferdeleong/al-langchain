from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

from dotenv import load_dotenv

import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["GOOGLE_CSE_ID"] = google_cse_id

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# instantiate a LLMChain specifically for text summarization.

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

# Google Search via API.
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]

# agent that leverages two tools.

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

response = agent("What's the latest news about the Mars rover? Then please summarize the results.")
print(response['output'])