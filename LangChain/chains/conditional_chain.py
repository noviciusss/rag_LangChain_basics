from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description="Give the sentiment of feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template = "Classify the  sentinment of the following review as Positive or Negative: /n {review} /n {format_instruction}",
    input_variables = ["review"],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template='Write a appropriate response to the positive review: /n {review}',
    input_variables=['review'],
)
prompt3 = PromptTemplate(
    template='Write a appropriate response to the negative review: /n {review}',
    input_variables=['review'],
)

classification_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive' ,prompt2 | model| parser),
    (lambda x: x.sentiment == 'Negative' ,prompt3 | model| parser),
    RunnableLambda(lambda x : "Could not classify the sentiment properly.")
)

chain = classification_chain | branch_chain
result = chain.invoke({"review":'The product quality is excellent and exceeded my expectations!'})
print(result)
chain.get_graph().print_ascii()