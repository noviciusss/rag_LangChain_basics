from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
dotenv.load_dotenv()

system_message = SystemMessage(content="You are a helpful research assistant specialized in summarizing academic papers.")
human_message = HumanMessage(content="Please summarize the research paper titled 'Attention Is All You Need' focusing on its key contributions and methodologies.") 

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
response = model.invoke([system_message, human_message]) 
print(response)