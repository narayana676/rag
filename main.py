import os
import uvicorn
import faiss

from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel, Field

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
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="KT RAG Agent API",
    version="1.0",
    description="RAG Agent for Knowledge Transfer using PDF Parsing",
)


# ============================================================
# GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing.")


# ============================================================
# KT PDF PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(
    BASE_DIR,
    "KT_Document.pdf"
)

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"KT_Document.pdf not found at: {PDF_PATH}"
    )


# ============================================================
# LOAD KT PDF
# ============================================================

print("========================================")
print("Loading Knowledge Transfer PDF...")
print("========================================")

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print("KT PDF loaded successfully.")
print(f"Number of pages: {len(documents)}")


# ============================================================
# SPLIT PDF INTO CHUNKS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# ============================================================
# GOOGLE GEMINI EMBEDDINGS
# ============================================================

print("Creating Gemini embeddings...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# CREATE FAISS VECTOR INDEX
# ============================================================

print("Creating FAISS vector database...")

dimension = len(
    embeddings.embed_query(
        "Knowledge Transfer test query"
    )
)

index = faiss.IndexFlatL2(dimension)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


# ============================================================
# STORE KT DOCUMENT CHUNKS
# ============================================================

vector_store.add_documents(chunks)

print("KT document embeddings stored successfully.")


# ============================================================
# KT RETRIEVAL TOOL
# ============================================================

@tool
def retrieve_kt_context(query: str) -> str:
    """
    Retrieve relevant information from the Knowledge
    Transfer PDF document.
    """

    docs = vector_store.similarity_search(
        query,
        k=4,
    )

    if not docs:
        return (
            "No relevant information was found "
            "in the provided KT document."
        )

    results = []

    for doc in docs:

        page_number = doc.metadata.get(
            "page",
            "unknown"
        )

        if isinstance(page_number, int):
            page_number = page_number + 1

        results.append(
            f"Page {page_number}:\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(results)


# ============================================================
# GOOGLE GEMINI LLM
# ============================================================

print("Initializing Gemini LLM...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# KNOWLEDGE TRANSFER RAG AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[retrieve_kt_context],

    system_prompt="""
You are an AI Knowledge Transfer (KT) Assistant.

Your purpose is to help users understand the project
information contained in the provided KT PDF.

IMPORTANT RULES:

1. Always use the retrieve_kt_context tool when the
   user asks a question related to the KT document.

2. Use ONLY information retrieved from the KT document.

3. Do NOT use outside knowledge.

4. Do NOT invent, guess, or assume information.

5. If the answer is not available in the KT document,
   respond exactly:

"I don't know based on the provided KT document."

6. Give clear and concise answers.

7. Explain technical information in simple language
   when possible.

8. Mention the source PDF page number whenever possible.

9. The document may contain information about:
   - Project overview
   - Project objectives
   - Technology stack
   - System architecture
   - PDF processing
   - Text chunking
   - Embeddings
   - FAISS vector database
   - RAG workflow
   - API endpoints
   - Setup and installation
   - Configuration
   - Deployment
   - Troubleshooting
   - Security
   - Testing
   - FAQs

10. Always retrieve relevant KT information before
    generating the final answer.
""",
)


# ============================================================
# INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Ask a question about the KT document"
    )


# ============================================================
# FORMAT USER INPUT
# ============================================================

def format_for_agent(data):

    if isinstance(data, dict):
        user_question = data["input"]
    else:
        user_question = data.input

    return {
        "messages": [
            {
                "role": "user",
                "content": user_question,
            }
        ]
    }


# ============================================================
# EXTRACT FINAL AGENT RESPONSE
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

    if not messages:
        return str(agent_output)

    # Find final AI response
    for message in reversed(messages):

        message_type = getattr(
            message,
            "type",
            ""
        )

        # Skip tool output
        if message_type == "tool":
            continue

        content = getattr(
            message,
            "content",
            None
        )

        if content is None:
            continue

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

            if text_parts:
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
    output_type=str,
)


# ============================================================
# LANGSERVE API ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
)


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "project": "RAG Agent for KT using PDF Parsing",
        "status": "running",
        "message": "Knowledge Transfer RAG Agent is running successfully.",
        "endpoint": "/agent/invoke",
    }


# ============================================================
# RUN APPLICATION
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
        port=port,
    )
