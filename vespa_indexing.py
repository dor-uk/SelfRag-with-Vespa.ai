import json
import torch
from transformers import AutoTokenizer, AutoModel
from vespa.application import Vespa
import numpy as np




def get_embedding(text, model, tokenizer):
    """Generate embedding for a given text using dbmdz/bert-base-turkish-cased."""
    
    
    # Use the same parameters as during chunking
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=False,  # No need for padding with single texts
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding.tolist()




def indexing_main():
    # Step 1: Connect to Vespa
    app = Vespa(url="http://localhost", port=8080)

        # Step 2: Load BERTurk model for embeddings
    MODEL_NAME = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Step 3: Load and Process Corpus
    with open("processed_corpus.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Step 4: Index Documents in Vespa
    for doc in documents:
        embedding = get_embedding(doc["content"], model, tokenizer)  # Convert text to vector
        
        # Print the first 10 values of embedding for debugging
        #print(f"Embedding for {doc['filename']} - Chunk {doc['chunk_id']}:\n{embedding[:10]}...")

        


        response = app.feed_data_point(
            schema="mydoc",
            data_id=f"{doc['filename']}_{doc['chunk_id']}",
            fields={
                "filename": doc["filename"],
                "topic": doc["topic"],
                "chunk_id": doc["chunk_id"],
                "content": doc["content"],
                #"embedding": {
                #    "cells": [{"address": {"d0": i}, "value": val} for i, val in enumerate(embedding)]
                #}
                "embedding": {"values": embedding}  
                
            }
        )

        if response.status_code != 200:
            print(f" Failed to index: {doc['filename']} - Chunk {doc['chunk_id']}")
            print(response.json())  # Show detailed error message
        else:
            print(f" Successfully indexed: {doc['filename']} - Chunk {doc['chunk_id']}")

    print(" All documents indexed successfully!")

indexing_main()
