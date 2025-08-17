# Product Similarity Search with Azure OpenAI + Pinecone

This example demonstrates how to build a simple semantic product search engine using Azure OpenAI embeddings and Pinecone vector database.

## 🔹 What the code does

1. Initialize clients
- Connects to Azure OpenAI for embeddings.
- Connects to Pinecone for vector storage.

2. Create a vector index
- Creates (or reuses) a Pinecone index called product-similarity-index with 1536 dimensions.

3. Embed and upsert product data
- Generates vector embeddings for sample product descriptions.
- Stores them in Pinecone with product metadata (title + description).

4. Run similarity search
- Embeds a query like "clothing item for summer".
- Retrieves the top 3 most similar products from the index.

5. Display results
- Prints product titles and similarity scores.

## 🔹 Example Output
```sh
All result {'matches': [{'id': 'prod1', 'score': 0.384327471, 'values': []},
             {'id': 'prod4', 'score': 0.337542564, 'values': []},
             {'id': 'prod5', 'score': 0.332922548, 'values': []}],
 'namespace': '',
 'usage': {'read_units': 1}}
Top 3 similar products for the query: 'clothing item for summer'

- Red T-Shirt (Similarity score: 0.3843)
- White Sneakers (Similarity score: 0.3375)
- Green Hoodie (Similarity score: 0.3329)
```

⚡ In short: this script shows how to use Azure OpenAI embeddings + Pinecone to build a semantic search for products.