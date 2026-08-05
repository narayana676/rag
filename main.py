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


# ---------------------------------------------------
# FastAPI
# ---------------------------------------------------

app = FastAPI(
    title="Internet RAG API",
    version="1.0",
    description="LangServe Internet Knowledge Agent"
)

# ---------------------------------------------------
# API KEY
# ---------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is missing.")

# ---------------------------------------------------
# Knowledge Base
# ---------------------------------------------------

big_paragraph = """
The Internet is a global system of interconnected computer networks that uses TCP/IP.

The origins of the Internet date back to packet switching research in the 1960s.

ARPANET was the primary precursor network.

The commercialization of the Internet happened during the 1990s.

Today the Internet powers cloud computing, education,
e-commerce, healthcare, communication and social media.
"""

documents = [Document(page_content=big_paragraph)]

# ---------------------------------------------------
# Split Documents
# ---------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# ---------------------------------------------------
# Embeddings
# ---------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

dimension = len(embeddings.embed_query("hello"))

index = faiss.IndexFlatL2(dimension)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(chunks)

# ---------------------------------------------------
# Tool
# ---------------------------------------------------

@tool
def retrieve_internet_context(query: str) -> str:
    """Retrieve Internet information from the knowledge base."""

    docs = vector_store.similarity_search(query, k=2)

    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------------------------------
# Agent
# ---------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[retrieve_internet_context],
    system_prompt=(
        "You answer only using the retrieved context. "
        "If the answer is unavailable, say "
        "'I don't know based on the provided knowledge base.'"
    )
)

# ---------------------------------------------------
# LangServe
# ---------------------------------------------------

class AgentInput(BaseModel):
    input: str = Field(description="Ask a question")

def format_for_agent(x):
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {
        "messages": [
            ("user", user_input)
        ]
    }

def extract_text_response(agent_output):

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict) and "messages" in value:

                messages = value["messages"]
                break

    if messages:

        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)

# ---------------------------------------------------
# Route
# ---------------------------------------------------

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)

# ---------------------------------------------------
# Run
# ---------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
