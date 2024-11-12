"""
Constants for Advanced RAG System

This script defines node name constants used in the graph workflow,
representing each processing step in the RAG pipeline.

Key Constants:
1. RETRIEVE: Vector DB document retrieval
2. GRADE_DOCUMENTS: Document relevance assessment (C-RAG)
3. GENERATE: Response generation 
4. WEBSEARCH: Web data retrieval (Adaptive-RAG)

These constants are used in graph.add_node() and conditional edges to maintain
consistent node identification across the workflow definition.
"""

# Constant for the retrieve node
RETRIEVE = "retrieve"

# Constant for the grade_documents node
GRADE_DOCUMENTS = "grade_documents"

# Constant for the generate node
GENERATE = "generate"

# Constant for the web_search node
WEBSEARCH = "websearch"
