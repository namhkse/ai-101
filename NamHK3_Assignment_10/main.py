# pip install pinecone openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI
import os

# Step 1: Initialize Pinecone client and embedding model
load_dotenv()

client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# Step 2: Create or connect to an index
index_name = "product-similarity-index"
if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Step 3: Upsert sample product vectors into the index
products = [
 {"id": "prod1", "title": "Red T-Shirt", "description": "Comfortable cotton t-shirt in bright red"},
 {"id": "prod2", "title": "Blue Jeans", "description": "Stylish denim jeans with relaxed fit"},
 {"id": "prod3", "title": "Black Leather Jacket", "description": "Genuine leather jacket with classic style"},
 {"id": "prod4", "title": "White Sneakers", "description": "Comfortable sneakers perfect for daily wear"},
 {"id": "prod5", "title": "Green Hoodie", "description": "Warm hoodie made of organic cotton"},
]


def get_embedding(text):
    response = client.embeddings.create(
        input=text, model=os.getenv("AZURE_DEPLOYMENT_NAME")
    )
    return response.data[0].embedding

vectors = []

for p in products:
    embedding = get_embedding(p["description"])
    vectors.append({
        "id": p["id"],
        "values": embedding,
        "metadata": {
            "title": p["title"],
            "summary": p["description"]
        }
    })

index.upsert(vectors) # TODO: uncomment

# Step 4: Prepare input query embedding
query = "clothing item for summer"
query_embedding = get_embedding(query)

# Step 5: Query Pinecone index for top 3 most similar vectors
top_k = 3
results = index.query(vector=query_embedding,
                      top_k=top_k, include_metadata=False)
print("All result", results)

# Step 6: Display results
print(f"Top {top_k} similar products for the query: '{query}'\n")
for match in results.matches:
    product_id = match.id
    score = match.score
    # Find product details
    product = next(p for p in products if p["id"] == product_id)
    print(f"- {product['title']} (Similarity score: {score:.4f})")
