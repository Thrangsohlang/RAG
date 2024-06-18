# importing libraries
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from warnings import filterwarnings

# remove the warnings
filterwarnings('ignore')

# configure the environment
os.environ["GOOGLE_API_KEY"] = "AIzaSyBucpKjCJCH4zX4_R5qS__yAdEnUygmxzw"

# configure the model
llm_g = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)


def load_pdf(file):
    """This function load the pdf"""
    # Load the document and split it into chunks
    loader = PyPDFLoader(file)
    documents = loader.load()
    return documents

def retriever(documents):
    """This function returns the retriever"""
    # Split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load it into Chroma
    db = Chroma.from_documents(docs, embedding_function)

    # Query it
    retriever = db.as_retriever(search_type = 'similarity',search_kwargs={'k': 20})
    
    return retriever

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate.from_template(template)

def rag_chain(retriever):
    """This function returns the rag_chain"""
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_g
        | StrOutputParser()
    )
    return rag_chain
