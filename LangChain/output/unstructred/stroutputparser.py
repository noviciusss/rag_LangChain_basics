from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_tokens=300)


template1 = PromptTemplate(
    template = "Write a  detailed report on {topic}",
    input_variables = ["topic"],
    
)

template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text . /n {text}",
    input_variables = ["text"],
    
)


prompt1 = template1.invoke({"topic":'black hole'})

result1 = model.invoke(prompt1)
print(result1.content)

prompt2 = template2.invoke({"text":result1.content})
result2 = model.invoke(prompt2)  
print(result2.content)