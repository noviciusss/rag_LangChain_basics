from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

prompt1 = PromptTemplate(
    template = "Generate a detailed report about {topic}",
    input_variables = ["topic"],
)
prompt2 = PromptTemplate(
    template = "Generate a 5 pointer summary on the following text . /n {text}",
    input_variables = ["text"], 
)
prompt3 = PromptTemplate(
    template = "List 5 potential questions from the following summary . /n {summary}",
    input_variables = ["summary"],
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes ': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
    
})

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain

result = chain.invoke({"topic":'Laptop computers and their impact on productivity'})
print(result)
chain.get_graph().print_ascii()