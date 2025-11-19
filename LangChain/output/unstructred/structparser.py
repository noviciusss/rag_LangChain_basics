from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser ,ResponseSchema

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1',description='first fact about the topic'),
    ResponseSchema(name='fact_2',description='second fact about the topic'),
    ResponseSchema(name='fact_3',description='third fact about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)
template  = PromptTemplate(
    template='Give me 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],      
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
chain = template | model | parser
prompt = template.invoke()

final_result = parser.parse({'topic':'the moon'})
print(final_result)