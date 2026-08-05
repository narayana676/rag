import os
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.tools import tool
from langchain.agents import create_agent

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="Internet RAG API",
    version="1.0"
)

# ==========================================================
# API Key
# ==========================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# ==========================================================
# Knowledge Base
# ==========================================================

big_paragraph = """
The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices.

The origins of the Internet date back to the development of packet switching and research commissioned by the United States Department of Defense in the 1960s.

The primary precursor network was the ARPANET.

The commercialization of the Internet in the mid-1990s marked a turning point in its expansion.

Today the Internet supports cloud computing, online gaming, social media, video conferencing, e-commerce, education and healthcare.
"""

documents = [Document(page_content=big_paragraph)]

# ==========================================================
# Split Documents
# ==========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

# ==========================================================
# Embeddings
# ==========================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

dimension = len(embeddings.embed_query("hello"))

index = faiss.IndexFlatL2(dimension)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(chunks)

# ==========================================================
# LLM
# ==========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
)

# ==========================================================
# Tool
# ==========================================================

@tool(response_format="content_and_artifact")
def retrieve_internet_context(query: str):
    """Retrieve internet information."""

    docs = vector_store.similarity_search(query, k=2)

    text = "\n\n".join(
        f"{doc.page_content}"
        for doc in docs
    )

    return text, docs

# ==========================================================
# Agent
# ==========================================================

agent = create_agent(
    model=llm,
    tools=[retrieve_internet_context],
    system_prompt=(
        "You answer only using the retrieved context. "
        "If the answer is not available, say 'I don't know.'"
    ),
)

# ==========================================================
# Request Model
# ==========================================================

class Query(BaseModel):
    question: str

# ==========================================================
# API Endpoint
# ==========================================================

@app.get("/")
def home():
    return {"message": "Internet RAG API Running"}

@app.post("/chat")
def chat(query: Query):

    try:

        result = ""

        for event in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query.question
                    }
                ]
            },
            stream_mode="values",
        ):

            msg = event["messages"][-1]

            if isinstance(msg.content, list):

                for block in msg.content:

                    if block.get("type") == "text":
                        result += block.get("text", "")

            else:
                result += str(msg.content)

        return {
            "response": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
