# !pip install langchain-openai langchain-community langchain_openai langchain-tavily langgraph faiss-cpu
from IPython.display import Image, display

import os
from langchain_core.tools import tool
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage

mock_chunks = [
    Document(
        page_content="Patients with a sore throat should drink warm fluids and avoid cold beverages."
    ),
    Document(
        page_content="Mild fevers under 38.5°C can often be managed with rest and hydration."
    ),
    Document(
        page_content="If a patient reports dizziness, advise checking their blood pressure and hydration level."
    ),
    Document(
        page_content="Persistent coughs lasting more than 2 weeks should be evaluated for infections or allergies."
    ),
    Document(
        page_content="Patients experiencing fatigue should consider iron deficiency or poor sleep as potential causes."
    ),
]

os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"] = "your-embedding-api-key"
os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"] = "https://aiportalapi.stu-platform.live/jpe"
os.environ["AZURE_OPENAI_EMBED_MODEL"] = "text-embedding-3-small"
os.environ["AZURE_OPENAI_LLM_API_KEY"] = "your-llm-api-key"
os.environ["AZURE_OPENAI_LLM_ENDPOINT"] = "https://aiportalapi.stu-platform.live/jpe"
os.environ["AZURE_OPENAI_LLM_MODEL"] = "GPT-4o-mini"
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# --- Setup LLM ---
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_LLM_ENDPOINT"],
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
    api_version="2024-07-01-preview",
)

# --- Setup FAISS Retriever from Mock Chunks ---
embedding_model = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBED_MODEL"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
)
db = FAISS.from_documents(mock_chunks, embedding_model)
retriever = db.as_retriever()

# Retrieve advicde tool.
@tool
def retrieve_advice(user_input: str) -> str:
    """Searches internal documents for relevant patient advice."""
    docs = retriever.get_relevant_documents(user_input)
    return "\n".join(doc.page_content for doc in docs)


# Tavily tools.
tavily_tool = TavilySearchResults()
# --- LLM SETUP ---
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_LLM_ENDPOINT"],
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
    api_version="2024-02-15-preview",
)

# Bind tool to model.
llm_with_tools = llm.bind_tools([retrieve_advice, tavily_tool])

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


tool_node = ToolNode([retrieve_advice, tavily_tool])
# Build the graph.
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges("call_model", should_continue, ["tools",
                                                                    END])
graph_builder.add_edge("tools", "call_model")
graph = graph_builder.compile()

if __name__ == "__main__":
    png_bytes = graph.get_graph().draw_mermaid_png()
    display(Image(png_bytes))

    result = graph.invoke(
        {
            "messages": [
                SystemMessage(
                    content="You are a helpful medical assistant. Use tools if needed."
                ),
                HumanMessage(
                    content="I feel tired and have a sore throat. What should I do?"
                ),
            ]
        }
    )
    print("Final Response:")
    print(result["messages"][-1].content)
