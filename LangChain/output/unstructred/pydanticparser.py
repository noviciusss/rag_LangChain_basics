from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name :str = Field(description="The name of the person")
    age :int = Field(description="The age of the person in years", gt=18 , lt= 99)
    city :str = Field(description="The city where the person lives")
    

parser = PydanticOutputParser(pydantic_object=Person)
template  = PromptTemplate(
    template = 'Generate a name age and city of a fictional {place} person. \n {format_instructions}',
    input_variables = ['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
    
)

prompt= template.invoke({'place':'China'})
print(prompt)
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
