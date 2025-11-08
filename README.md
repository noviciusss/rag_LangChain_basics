# RAG Learning Playground

This project walks through a minimal Retrieval-Augmented Generation (RAG) workflow using LangChain, Groq LLMs, and Hugging Face embeddings.

## What You Can Do Here
- Load structured data (CSV) or unstructured documents (PDF) into LangChain `Document` objects.
- Split content into manageable chunks, embed them with Hugging Face models, and store vectors in a FAISS index.
- Run a retrieval QA chain that sends top matches to a Groq-hosted LLM for grounded answers.

## Getting Started
1. Install dependencies listed in the notebooks (e.g., `langchain`, `langchain-community`, `huggingface-hub`, `faiss-cpu`).
2. Create a `.env` file containing `GROQ_API_KEY` and `HUGGINGFACEHUB_API_TOKEN`, then load it like the notebooks show.
3. Open `rag_first/rag_first.ipynb` for the CSV-based demo, or `rag_second/rag_second.ipynb` for the PDF version, and run the cells top-to-bottom.

Feel free to duplicate the notebooks and experiment with different loaders, embedding models, or prompt templates as you continue exploring RAG systems.
