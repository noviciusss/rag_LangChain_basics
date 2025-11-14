from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import dotenv
dotenv.load_dotenv()


model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash" ,temperature=0.7)

chat_history = [
    SystemMessage(content="You are a helpful research assistant."),
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

print(chat_history)