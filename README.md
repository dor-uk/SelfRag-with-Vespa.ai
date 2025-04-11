# SelfRag-with-Vespa.ai

This repository implements a **self-checking Retrieval-Augmented Generation (RAG)** pipeline. It combines document retrieval, local LLM generation, and self-evaluation to ensure high-quality, hallucination-free answers.

---

## Overview

- Uses **Vespa** as a semantic vector search engine
- Processes and chunks `.txt` files with semantic & token-based splitting
- Answers generated using **Phi-2** model locally via HuggingFace Transformers
- Each answer is validated by the same LLM for factual consistency
- Interactive user interface via **Streamlit**

---

## Stack

| Layer         | Technology Used                          |
|---------------|-------------------------------------------|
| Retrieval     | Vespa + Turkish BERT (`dbmdz/bert-base-turkish-cased`) |
| Chunking      | LangChain Text Splitters + NLTK          |
| Embedding     | Transformers + Manual pooling             |
| Generation    | `microsoft/phi-2`                         |
| Verification  | Answer-check using LLM itself             |
| UI            | Streamlit                                |

---

##  Self-RAG Verification

Each answer is verified by the same model (`phi-2`) using a second LLM prompt that checks consistency with the retrieved text. This reduces hallucinations and boosts trustworthiness. However phi-2 doesn"t work well with Turkishso another LLM will improve the results.

---
