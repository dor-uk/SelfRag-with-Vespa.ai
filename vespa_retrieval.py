import requests
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from vespa_indexing import indexing_main



def get_embedding(text, model, tokenizer):
    """Generate embedding for a given text using dbmdz/bert-base-turkish-cased."""
    
    
    # Use the same parameters as during chunking
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=False, 
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding.tolist()  

def retrieve_documents(query, top_k=3, chance = 1):
    """Retrieve the most relevant document chunks from Vespa."""
    # Vespa endpoint
    VESPA_ENDPOINT = "http://localhost:8080/search/"

    # Load Turkish BERT model for embeddings
    MODEL_NAME = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    embedding_vector = get_embedding(query, model, tokenizer)  # Convert query to embedding

    print("\n Query Embedding (first 10 values):", embedding_vector[:10])  

    
    vespa_embedding = {"values": embedding_vector}


    query_payload = {
        "yql": """
            select filename, topic, chunk_id, content from mydoc 
            where userQuery() or ([{"targetHits": 10}]nearestNeighbor(embedding, q_embedding))
        """,
        "input": {
            "query(q_embedding)": vespa_embedding
        },
        "userQuery": query,
        "ranking.profile": "hybrid",
        "hits": top_k
    }


    print("\n Sending Query to Vespa:", query_payload)  
    try:
        response = requests.post(VESPA_ENDPOINT, json=query_payload, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f" Error: Could not connect to Vespa ({e})")
        return []

    if response.status_code != 200:
        print(f" Error: Vespa request failed with status code {response.status_code}")
        print(" Response:", response.text)
        return []

    results = response.json().get("root", {}).get("children", [])

    #print("\n Raw Response from Vespa:", results)  # Debug print

    # Extract relevant documents
    relevant_chunks = []
    for result in results:
        relevance = result.get("relevance", 0)
        doc = result.get("fields", {})

        relevant_chunks.append({
            "filename": doc.get("filename", ""),
            "topic": doc.get("topic", ""),
            "chunk_id": doc.get("chunk_id", -1),
            "content": doc.get("content", ""),
            "relevance": relevance
        })

        

    
    if len(relevant_chunks) == 0 and chance == 1:
        print(" Not enough relevant chunks found. Retrying with adjusted parameters...")
        chance = 0
        return retrieve_documents(query, top_k=3, chance = 0)  

    return relevant_chunks



if __name__ == "__main__":
    #indexing_main()
    query_text = "7354 sayili kanun"
    retrieved_chunks = retrieve_documents(query_text,)

    print("\n Retrieved Chunks:")
    for chunk in retrieved_chunks:
        #print(f"[{chunk['chunk_id']}] {chunk['topic']} → {chunk['content'][:200]}...\n")
        print(chunk)
