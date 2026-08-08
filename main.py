import os
import uvicorn
import faiss

from fastapi import FastAPI
from langserve import add_routes

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain.agents import create_agent

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PDF RAG Agent API",
    version="1.0",
    description="RAG Agent using PDF Parsing and Google Gemini"
)


# ============================================================
# GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is missing.")


# ============================================================
# PDF PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(
    BASE_DIR,
    "your_document.pdf"
)

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"PDF file not found: {PDF_PATH}"
    )

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"PDF file not found: {PDF_PATH}"
    )


# ============================================================
# LOAD PDF
# ============================================================

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"PDF loaded successfully.")
print(f"Number of pages: {len(documents)}")


# ============================================================
# SPLIT PDF INTO CHUNKS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = len(
    embeddings.embed_query("test query")
)

index = faiss.IndexFlatL2(dimension)


vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)


# ============================================================
# ADD PDF CHUNKS TO VECTOR STORE
# ============================================================

vector_store.add_documents(chunks)

print("PDF embeddings stored in FAISS.")


# ============================================================
# RAG RETRIEVAL TOOL
# ============================================================

@tool
def retrieve_pdf_context(query: str) -> str:
    """
    Retrieve relevant information from the uploaded PDF.
    Use this tool whenever answering questions about the PDF.
    """

    docs = vector_store.similarity_search(
        query,
        k=4
    )

    if not docs:
        return "No relevant information was found in the PDF."

    results = []

    for doc in docs:

        page_number = doc.metadata.get(
            "page",
            "unknown"
        )

        results.append(
            f"Page {page_number + 1 if isinstance(page_number, int) else page_number}:\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(results)


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# RAG AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[retrieve_pdf_context],

    system_prompt="""
You are a PDF-based RAG Agent.

Your job is to answer questions using ONLY information
retrieved from the provided PDF.

Rules:

1. Always use the retrieve_pdf_context tool when the
   user asks a question about the PDF.

2. Do not use outside knowledge.

3. Do not invent information.

4. If the requested information is not present in
   the PDF, respond exactly:

   "I don't know based on the provided PDF."

5. Give clear and concise answers.

6. When possible, mention the page number where
   the information was found.
"""
)


# ============================================================
# INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Ask a question about the PDF"
    )


# ============================================================
# FORMAT INPUT FOR AGENT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# EXTRACT AGENT RESPONSE
# ============================================================

def extract_text_response(agent_output):

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict):

                if "messages" in value:

                    messages = value["messages"]
                    break

    if messages:

        last = messages[-1]

        content = getattr(
            last,
            "content",
            str(last)
        )

        # Gemini messages can sometimes return
        # structured content
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    if "text" in item:
                        text_parts.append(
                            item["text"]
                        )

                else:
                    text_parts.append(
                        str(item)
                    )

            return "\n".join(text_parts)

        return str(content)

    return str(agent_output)


# ============================================================
# CREATE RAG CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# ROOT ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "PDF RAG Agent is running",
        "endpoint": "/agent/invoke"
    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
