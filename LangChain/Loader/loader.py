from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader,WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
# from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

##Too load folder direcly , glob to define which type of files to load
# loader = DirectoryLoader("./documents", 
#                         glob="**/*.pdf",
#                         loader_cls=PyPDFLoader)

##Lazy loading  - 

##Web base loader 

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Narendra_Modi")
documents = loader.load()
print(f"Number of documents loaded: {len(documents)}")
# print(documents[0].page_content)

# for doc in documents:
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
prompt = PromptTemplate(
    template="Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"]
)

parser = StrOutputParser()
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = Faiss.from_documents(documents, embedding=embeddings)

chain = prompt |model | parser 

response = chain.invoke({'context': documents[0].page_content, 'question': "Who is Narendra Modi?"})
print(response)



####Semantic chunking example

###vectorstore example  chroma 
from langchain.vectorstores import Chroma
vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory="./chroma_db",
    collection_name="documents_collection"
)