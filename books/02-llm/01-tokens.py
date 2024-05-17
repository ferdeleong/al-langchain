from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

from dotenv import load_dotenv

import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_project_id = os.getenv("OPENAI_PROJECT_ID")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_PROJECT_ID"] = openai_project_id

llm = OpenAI(model="gpt-3.5-turbo-instruct", n=2, best_of=2)

# track token usage
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)

# create our examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# load the model
chat = ChatOpenAI(model="gpt-4", temperature=0.0)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
print(chain.run("What's the meaning of life?"))
