from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.example_selector import LengthBasedExampleSelecto, SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings


from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run("I love programming.")

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

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
chain.run("What's the secret to happiness?")

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

example_template = """
Word: {word}
Antonym: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template
)

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n\n",
)

print(dynamic_prompt.format(input="big"))

# Create a PromptTemplate
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Define some examples
examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]

# create Deep Lake dataset
# TODO: use your organization id here.  (by default, org id is your username)
my_activeloop_org_id = os.getenv("ACTIVELOOP_TOKEN")
my_activeloop_dataset_name = "langchain_course_fewshot_selector"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path)

# Embedding function
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Instantiate SemanticSimilarityExampleSelector using the examples
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embeddings, db, k=1
)

# Create a FewShotPromptTemplate using the example_selector
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Convert the temperature from Celsius to Fahrenheit",
    suffix="Input: {temperature}\nOutput:", 
    input_variables=["temperature"],
)

print(similar_prompt.format(temperature="10°C")) 
print(similar_prompt.format(temperature="30°C"))

# Add a new example to the SemanticSimilarityExampleSelector
similar_prompt.example_selector.add_example({"input": "50°C", "output": "122°F"})
print(similar_prompt.format(temperature="40°C")) 