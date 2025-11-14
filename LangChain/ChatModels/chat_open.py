from langchain_openai import ChatOpenAI
import dotenv
dotenv.load_dotenv()

model = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.7)

result = model.invoke("Explain the concept of LangChain in simple terms.")
print(result)
print(result.content )