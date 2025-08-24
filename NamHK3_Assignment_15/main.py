# Step 0: Imports & Environment Setup
import os
from typing import TypedDict, Optional
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Step 1: Example Amazon Documents
os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"] = "your-embedding-api-key"
os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"] = "https://aiportalapi.stu-platform.live/jpe"
os.environ["AZURE_OPENAI_EMBED_MODEL"] = "text-embedding-3-small"
os.environ["AZURE_OPENAI_LLM_API_KEY"] = "your-llm-api-key"
os.environ["AZURE_OPENAI_LLM_ENDPOINT"] = "https://aiportalapi.stu-platform.live/jpe"
os.environ["AZURE_OPENAI_LLM_MODEL"] = "GPT-4o-mini"

# --- Step 2: Sample Documents ---
docs = [
    Document(
        page_content="Walmart customers may return electronics within 30 days with a receipt and original packaging."
    ),
    Document(
        page_content="Grocery items at Walmart can be returned within 90 days with proof of purchase, except perishable products."
    ),
    Document(
        page_content="Walmart offers a 1-year warranty on most electronics and appliances. See product details for exceptions."
    ),
    Document(
        page_content="Walmart Plus members get free shipping with no minimum order amount."
    ),
    Document(
        page_content="Prescription medications purchased at Walmart are not eligible for return or exchange."
    ),
    Document(
        page_content="Open-box items are eligible for return at Walmart within the standard return period, but must include all original accessories."
    ),
    Document(
        page_content="If a Walmart customer does not have a receipt, most returns are eligible for store credit with valid photo identification."
    ),
    Document(
        page_content="Walmart allows price matching for identical items found on Walmart.com and local competitor ads."
    ),
    Document(
        page_content="Walmart Vision Center purchases may be returned or exchanged within 60 days with a receipt."
    ),
    Document(
        page_content="Returns on cell phones at Walmart require the device to be unlocked and all personal data erased."
    ),
    Document(
        page_content="Walmart gift cards cannot be redeemed for cash except where required by law."
    ),
    Document(
        page_content="Seasonal merchandise at Walmart (e.g., holiday decorations) may have modified return windows, see in-store signage."
    ),
    Document(
        page_content="Bicycles purchased at Walmart can be returned within 90 days if not used outdoors and with all accessories present."
    ),
    Document(
        page_content="For online Walmart orders, customers can return items in store or by mail using the prepaid label."
    ),
    Document(
        page_content="Walmart reserves the right to deny returns suspected of fraud or abuse."
    ),
]
# --- Step 3: Typed State for LangGraph ---


class RAGState(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]


# --- Step 4: Embedding & Vector Store ---
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
    model=os.environ["AZURE_OPENAI_EMBED_MODEL"],
    api_version="2024-07-01-preview",
)
vectorstore = FAISS.from_documents(
    docs,
    embeddings,
    docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)}),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# --- Step 5: Azure Chat Model ---
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_LLM_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_LLM_API_KEY"],
    deployment_name=os.environ["AZURE_OPENAI_LLM_MODEL"],
    api_version="2024-07-01-preview",
    temperature=0,
)
# --- Step 6: Prompt Template ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful Amazon support assistant. Use the provided information to answer product and policy questions. Always cite the retrieved info in your answer.",
        ),
        ("human", "{context}\n\nUser question: {question}"),
    ]
)
# --- Step 7: Define Nodes ---


def retrieve_node(state: RAGState) -> RAGState:
    docs = retriever.get_relevant_documents(state["question"])
    context = "\n".join([doc.page_content for doc in docs])
    return {**state, "context": context}


def generate_node(state: RAGState) -> RAGState:
    formatted_prompt = prompt.format(
        context=state["context"], question=state["question"]
    )
    answer = llm.invoke(formatted_prompt)
    return {**state, "answer": answer.content}


# --- Step 8: Build LangGraph ---
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.set_finish_point("generate")
rag_graph = builder.compile()

if __name__ == "__main__":
    user_question = "Can I return an opened Amazon Kindle?"
    result = rag_graph.invoke({"question": user_question})
    print(" Retrieved Context:\n", result["context"])
    print("\n Generated Answer:\n", result["answer"])
