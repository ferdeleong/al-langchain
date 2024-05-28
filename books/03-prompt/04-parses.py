from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from pydantic import BaseModel, Field, validator
from typing import List

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")

    # Throw error in case of receiving a numbered-list from API
    @validator('words')
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field

parser = PydanticOutputParser(pydantic_object=Suggestions)

template = """
Offer a list of suggestions to substitue the specified target_word based the presented context.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
			target_word="behaviour",
			context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

model = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.0)

output = model(model_input.to_string())

parser.parse(output)

template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")
    
    @validator('words')
    def not_start_with_number(cls, field):
      for item in field:
        if item[0].isnumeric():
          raise ValueError("The word can not start with numbers!")
      return field
    
    # ensure every reasoning ends with a dot
    @validator('reasons')
    def end_with_dot(cls, field):
      for idx, item in enumerate( field ):
        if item[-1] != ".":
          field[idx] += "."
      return field