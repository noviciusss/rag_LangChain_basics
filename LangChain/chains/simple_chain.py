from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = PromptTemplate(
    template="Tell me a joke about {subject}.",
    input_variables=["subject"],    )

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature=0.7)
parser = StrOutputParser()

chain = prompt | model | parser ##  | is pipe operator 
result = chain.invoke({"subject": "chickens"})
print(result)


chain.get_graph().print_ascii()