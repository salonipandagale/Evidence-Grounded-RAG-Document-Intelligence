# Evidence-Grounded RAG Document Intelligence

An end-to-end **Retrieval-Augmented Generation (RAG)** system that answers questions from user-provided documents using **semantic retrieval, FAISS vector search, and grounded text generation**.

The system is designed to reduce unsupported or hallucinated responses by requiring the language model to generate answers only from the retrieved document context and by displaying the evidence used for each response.

## Live Demo

**Live Application:** https://evidence-grounded-rag-document.onrender.com/

**GitHub Repository:**  
https://github.com/salonipandagale/Evidence-Grounded-RAG-Document-Intelligence

---

## Overview

Large Language Models can generate fluent answers but may produce information that is not supported by the source documents.

This project implements an **evidence-grounded RAG pipeline** that connects document retrieval with language generation:

```text
User Documents
      ↓
Document Parsing
      ↓
Sentence-Aware Chunking
      ↓
Semantic Embeddings
      ↓
FAISS Vector Index
      ↓
Similarity Search
      ↓
Score Threshold Filtering
      ↓
Retrieved Evidence
      ↓
Grounded Prompt
      ↓
FLAN-T5 Generator
      ↓
Answer + Evidence
