from langchain_openai import OpenAI

import dotenv
dotenv.load_dotenv()

llm = OpenAI(model = "gpt-4.1-2025-04-14", temperature=0.7)

result = llm.invoke("What is langchain?")
print(result)