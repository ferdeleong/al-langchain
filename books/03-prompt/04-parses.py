import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser, RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Ensure the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "OPENAI_API_KEY environment variable is not set."

# Define the Pydantic model for the output
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

    # Validators
    @validator('words', allow_reuse=True)
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field

    @validator('reasons', allow_reuse=True)
    def end_with_dot(cls, field):
        for idx, item in enumerate(field):
            if item[-1] != ".":
                field[idx] += "."
        return field

# Set up the Pydantic output parser
parser = PydanticOutputParser(pydantic_object=Suggestions)

# Define the prompt template
template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the model input
model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Initialize the OpenAI model
model = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.0)

# Generate the output from the model
output = model(model_input.to_string())

# Parse the output
parsed_output = parser.parse(output)
print(parsed_output)

# Example of handling misformatted output
missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

# Using OutputFixingParser to correct the output
outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
fixed_output = outputfixing_parser.parse(missformatted_output)
print(fixed_output)

# Example of handling a more complex misformatted output
missformatted_output = '{"words": ["conduct", "manner"]}'

# Using RetryWithErrorOutputParser to correct the output
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)
fixed_output_retry = retry_parser.parse_with_prompt(missformatted_output, model_input)
print(fixed_output_retry)
