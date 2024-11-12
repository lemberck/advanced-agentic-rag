"""
Generate Node for Advanced RAG System

This script defines a node that produces responses to user questions using
retrieved context documents.

Key Components:
1. GraphState: Imported type representing the shared state of the RAG graph.
2. generation_chain: Imported chain for generating responses.
3. generate function: Core logic for the generate node. 
Generates a response based on the current state (question and documents).
"""

from typing import Any, Dict
from graph.chains.generation_dev import generation_chain
from graph.state_dev import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response based on the current graph state.
    1. It extracts the question and retrieved documents from the current state in GraphState.
    2. It uses the generation_chain to produce a response.
    3. It returns an updated state with the generated response.
    """

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
