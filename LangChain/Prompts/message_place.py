from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv

dotenv.load_dotenv()
chat_template = ChatPromptTemplate([
    SystemMessage(content="You are a helpful research assistant."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{input}")
])
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
with open("chat_history.txt", "r") as f:
    chat_history = f.readlines()