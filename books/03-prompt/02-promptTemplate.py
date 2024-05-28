from langchain.chains import LLMChain
from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.example_selector import LengthBasedExampleSelector

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided, answer
with "I don't know".
Context: Quantum computing is an emerging field that leverages quantum mechanics to solve complex problems faster than classical computers.
...
Question: {query}
Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

# Create the LLMChain for the prompt
chain = LLMChain(llm=llm, prompt=prompt_template)

# Set the query you want to ask
input_data = {"query": "What is the main advantage of quantum computing over classical computing?"}

# Run the LLMChain to get the AI-generated answer
response = chain.run(input_data)

print("Question:", input_data["query"])
print("Answer:", response)

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

examples = [
    {"animal": "lion", "habitat": "savanna"},
    {"animal": "polar bear", "habitat": "Arctic ice"},
    {"animal": "elephant", "habitat": "African grasslands"}
]

example_template = """
Animal: {animal}
Habitat: {habitat}
"""

example_prompt = PromptTemplate(
    input_variables=["animal", "habitat"],
    template=example_template
)

dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Identify the habitat of the given animal",
    suffix="Animal: {input}\nHabitat:",
    input_variables=["input"],
    example_separator="\n\n",
)

# Create the LLMChain for the dynamic_prompt
chain = LLMChain(llm=llm, prompt=dynamic_prompt)

# Run the LLMChain with input_data
input_data = {"input": "tiger"}
response = chain.run(input_data)

print(response)

# Save prompt templagte
prompt_template.save("awesome_prompt.json")

# retrieve prompt template
from langchain.prompts import load_prompt
loaded_prompt = load_prompt("awesome_prompt.json")

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

examples = [
    {
        "query": "How do I become a better programmer?",
        "answer": "Try talking to a rubber duck; it works wonders."
    }, {
        "query": "Why is the sky blue?",
        "answer": "It's nature's way of preventing eye strain."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative and funny responses to users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Create the LLMChain for the few_shot_prompt_template
chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

# Run the LLMChain with input_data
input_data = {"query": "How can I learn quantum computing?"}
response = chain.run(input_data)

print(response)

examples = [
    {
        "query": "How do you feel today?",
        "answer": "As an AI, I don't have feelings, but I've got jokes!"
    }, {
        "query": "What is the speed of light?",
        "answer": "Fast enough to make a round trip around Earth 7.5 times in one second!"
    }, {
        "query": "What is a quantum computer?",
        "answer": "A magical box that harnesses the power of subatomic particles to solve complex problems."
    }, {
        "query": "Who invented the telephone?",
        "answer": "Alexander Graham Bell, the original 'ringmaster'."
    }, {
        "query": "What programming language is best for AI development?",
        "answer": "Python, because it's the only snake that won't bite."
    }, {
        "query": "What is the capital of France?",
        "answer": "Paris, the city of love and baguettes."
    }, {
        "query": "What is photosynthesis?",
        "answer": "A plant's way of saying 'I'll turn this sunlight into food. You're welcome, Earth.'"
    }, {
        "query": "What is the tallest mountain on Earth?",
        "answer": "Mount Everest, Earth's most impressive bump."
    }, {
        "query": "What is the most abundant element in the universe?",
        "answer": "Hydrogen, the basic building block of cosmic smoothies."
    }, {
        "query": "What is the largest mammal on Earth?",
        "answer": "The blue whale, the original heavyweight champion of the world."
    }, {
        "query": "What is the fastest land animal?",
        "answer": "The cheetah, the ultimate sprinter of the animal kingdom."
    }, {
        "query": "What is the square root of 144?",
        "answer": "12, the number of eggs you need for a really big omelette."
    }, {
        "query": "What is the average temperature on Mars?",
        "answer": "Cold enough to make a Martian wish for a sweater and a hot cocoa."
    }
]

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100  
)

dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Existing example and prompt definitions, and dynamic_prompt_template initialization

# Create the LLMChain for the dynamic_prompt_template
chain = LLMChain(llm=llm, prompt=dynamic_prompt_template)

# Run the LLMChain with input_data
input_data = {"query": "Who invented the telephone?"}
response = chain.run(input_data)

print(response)