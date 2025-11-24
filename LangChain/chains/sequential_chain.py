from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
prompt1 = PromptTemplate(
    template = "Genrate a detailed report about {topic}",
    input_variables = ["topic"],
)

prompt2 = PromptTemplate(
    template = "Generate a 5 pointer summary on the following text . /n {text}",
    input_variables = ["text"], 
)

parser = StrOutputParser()

chain = prompt1 | model | prompt2 | model | parser

result = chain.invoke({"topic":'Ai and job automation'})
print(result)

chain.get_graph().print_ascii()