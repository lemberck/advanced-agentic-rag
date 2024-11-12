"""
Retrieve Node for Advanced RAG System

This script defines the retrieve node, which is responsible for fetching relevant
documents based on the user's question from the vector database.

Key Components:
1. GraphState: Represents the shared state of the RAG graph.
2. retriever: Imported retriever component for document fetching.
3. retrieve function: Core logic for the retrieve node.
This node initiates the process of gathering context for answering the user's question.
"""

from typing import Any, Dict

from graph.state_dev import GraphState
from ingestion_dev import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from the vector database based on the user's question.
    1. It extracts the question from the current state in GraphState.
    2. It uses the retriever to fetch relevant documents.
    3. It returns an updated state with the retrieved documents.

    This function is typically called at the beginning of the RAG pipeline,
    setting the stage for document grading and response generation, if the user's question is
    related to the topics in the vector database.
    """
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
