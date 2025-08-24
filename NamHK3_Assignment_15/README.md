# Building a RAG Chatbot for Product and Policy Support in Retail


This project demonstrates how to build a **retrieval-augmented conversational assistant** using:

* **LangGraph** for managing structured conversation flows
* **Azure OpenAI** for chat-based reasoning and embeddings
* **FAISS** for document similarity search
* **InMemoryDocstore** for lightweight storage of example documents

Although the example code currently uses **Amazon policy documents**, the architecture is directly extendable to **healthcare use cases** (e.g., collecting patient symptoms, retrieving advice, and providing preliminary guidance).

---

## 📦 Installation

Install the dependencies:

```bash
pip install langchain-openai langchain-community langgraph faiss-cpu
```

---

## 🔑 Environment Setup

Before running, set your Azure OpenAI API keys and endpoints:

```bash
export AZURE_OPENAI_EMBEDDING_API_KEY="your-embedding-api-key"
export AZURE_OPENAI_EMBEDDING_ENDPOINT="your-embedding-endpoint"
export AZURE_OPENAI_EMBED_MODEL="text-embedding-3-small"

export AZURE_OPENAI_LLM_API_KEY="your-llm-api-key"
export AZURE_OPENAI_LLM_ENDPOINT="your-llm-endpoint"
export AZURE_OPENAI_LLM_MODEL="GPT-4o-mini"
```

⚠️ For demo purposes, keys are hardcoded in the script — in production, always use environment variables or a secret manager.

---

## 📂 Code Workflow

1. **Documents Setup**
   Example documents are embedded (Amazon return policies in this sample). In a healthcare chatbot, these could be **medical guidelines or patient care notes**.

2. **FAISS Retriever**
   A vector store is built with `AzureOpenAIEmbeddings` and queried for relevant context.

3. **LangGraph Workflow**

   * `retrieve_node`: pulls context relevant to the user’s question.
   * `generate_node`: constructs a prompt and generates a response via AzureChatOpenAI.
   * Graph edges ensure the pipeline flows `Retrieve → Generate → End`.

4. **Prompting**
   A `ChatPromptTemplate` guides the model to always use retrieved information and cite sources in answers.

---

## ▶️ Running the Script

Run the assistant with:

```bash
python main.py
```

Example question:

```text
Can I return an opened Amazon Kindle?
```

### ✅ Example Output

```
Retrieved Context:
 Open-box items are eligible for return at Walmart within the standard return period, but must include all original accessories.
Walmart customers may return electronics within 30 days with a receipt and original packaging.

 Generated Answer:
 Yes, you can return an opened Amazon Kindle, but it must be within the standard return period, which is typically 30 days from the date of delivery. Make sure to include all original accessories and packaging when you return the item. For more specific details, you can check Amazon's return policy on their website.
```

---