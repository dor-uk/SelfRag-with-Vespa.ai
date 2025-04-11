import torch
from vespa.application import Vespa
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import requests
import json
import numpy as np
import re

# Import functions from other scripts
from vespa_indexing import indexing_main
from vespa_retrieval import retrieve_documents
from generation import generation_pipeline
from answer_check import answer_check





def self_rag_pipeline(query):
    print(f"\n Retrieving relevant documents for: {query}")
    retrieved_chunks = retrieve_documents(query)

    if len(retrieved_chunks) == 0:
        print(" No relevant documents found.")
        return None

    print(" Generating answer...")
    answer = generation_pipeline(query, retrieved_chunks)

    print(" Evaluating answer usefulness...")
    score = answer_check(query, answer)
    # score = int(score)

    # # If answer quality is too low (≤2), retry retrieval & generation once
    # if score <= 2:
    #     print(" Low-quality answer detected. Retrying retrieval & generation...")
    #     retrieved_chunks = retrieve_documents(query)
    #     answer = generation_pipeline(query, retrieved_chunks)
    #     score = answer_check(query, answer)  # Grade again

    print(f"\n Final Answer (Score: {score}): {answer}")
    return {"answer": answer, "score": score}


#  Run Test Query
if __name__ == "__main__":
    indexing_main()
    test_query = "Sözleşmeli öğretmenliğe başvuru şartları nelerdir?"
    result = self_rag_pipeline(test_query)
    print("\nFinal Output:\n", result)
