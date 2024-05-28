from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model_name="text-davinci-003", temperature=0, api_key=api_key)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""
prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

input_data = {"theme": "interstellar travel", "year": "3030"}

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""
example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="purple")

# Create the LLMChain for the prompt
chain = LLMChain(llm=llm, prompt=PromptTemplate(template=formatted_prompt, input_variables=[]))

# Run the LLMChain to get the AI-generated emotion associated with the input color
response = chain.run({})

print("Color: purple")
print("Emotion:", response)

template = "Tell me something about {topic}."
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)
prompt.format(topic="dogs")

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Prompt 1
template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer: """
prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

# Create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.run({})

# Extract the scientist's name from the response
scientist = response_question.strip()

# Create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# Input data for the second prompt
input_data = {"scientist": scientist}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)

# Bad chain prompt example
# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Prompt 1
template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Tell me something interesting about {scientist}.
Answer: """
prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

# Create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.run({})

# Extract the scientist's name from the response
scientist = response_question.strip()

# Create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# Input data for the second prompt
input_data = {"scientist": scientist}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Prompt 1
template_question = """What are some musical genres?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Tell me something about {genre1}, {genre2}, and {genre3} without giving any specific details.
Answer: """
prompt_fact = PromptTemplate(input_variables=["genre1", "genre2", "genre3"], template=template_fact)

# Create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.run({})

# Assign three hardcoded genres
genre1, genre2, genre3 = "jazz", "pop", "rock"

# Create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# Input data for the second prompt
input_data = {"genre1": genre1, "genre2": genre2, "genre3": genre3}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Genres:", genre1, genre2, genre3)
print("Fact:", response_fact)

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

examples = [
    {
        "query": "What's the secret to happiness?",
        "answer": "Finding balance in life and learning to enjoy the small moments."
    }, {
        "query": "How can I become more productive?",
        "answer": "Try prioritizing tasks, setting goals, and maintaining a healthy work-life balance."
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
life coach. The assistant provides insightful and practical advice to the users' questions. Here are some
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

# Create the LLMChain for the few-shot prompt template
chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

# Define the user query
user_query = "What are some tips for improving communication skills?"

# Run the LLMChain for the user query
response = chain.run({"query": user_query})

print("User Query:", user_query)
print("AI Response:", response)