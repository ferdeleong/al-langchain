from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv

import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_project_id = os.getenv("OPENAI_PROJECT_ID")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_PROJECT_ID"] = openai_project_id

llm = OpenAI(model="gpt-3.5-turbo-instruct", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)