"""
Graph State for Advanced RAG System

This script defines the shared state structure used across all nodes of the RAG system:
1. Core-RAG: Basic document retrieval and response generation
2. Corrective-RAG (C-RAG): Document relevance assessment
3. Self-RAG: Response quality evaluation (Hallucination check and Generated response relevance assessment)
4. Adaptive-RAG: Dynamic source selection and web search

Key Components:
1. GraphState (TypedDict): Shared memory structure containing:
   - User inputs (question)
   - Generated outputs (generation)
   - Document management (documents)
   - Web search control (web_search, web_search_attempts)

The state is accessed and modified by:
- Nodes (retrieve, grade_documents, generate, web_search)
- Conditional edges (decide_to_generate, grade_generation)
- Entry points (route_question)

>> Each component's requirements are reflected in the GraphState attributes.
"""

from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of the graph.
    
    This class uses TypedDict to create a dictionary with a predefined structure,
    providing type hints.
    
    Attributes:
        question (str): The user's input question.
        generation (str): The LLM-generated response.
        web_search (bool): Flag indicating whether to include web search results.
        documents (List[str]): List of retrieved documents or context from web search.
        web_search_attempts (int): Counter for number of web search attempts made.
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    web_search_attempts: int
