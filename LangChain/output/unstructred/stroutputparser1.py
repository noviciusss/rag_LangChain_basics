from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

template1 = PromptTemplate(
    template = "Write a  detailed report on {topic}",
    input_variables = ["topic"],
    
)

template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text . /n {text}",
    input_variables = ["text"],
    
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "semiconductor industry"})
print(result)