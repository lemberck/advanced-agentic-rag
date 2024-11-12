"""
Web Search Node for Advanced RAG System

This script defines the web_search node, which is responsible for fetching
information from the web in two scenarios:

1. Corrective-RAG Trigger: When vector DB documents are insufficient (<3 relevant docs)
2. Self-RAG Trigger: When generated responses contain hallucinations or are incomplete.

>> This is part of the Adaptive-RAG component of the system.

Key Components:
1. GraphState: Represents the shared state of the RAG graph.
2. TavilySearchResults: A tool for performing web searches (limited to 3 results).
3. web_search function: Core logic for augmenting document context.

This node enhances the RAG system by:
- Enhancing vector DB results with web data
- Handling out-of-vocabulary (OOV) topics not in vector DB
- Limiting web searches to 2 attempts per query
"""

from typing import Any, Dict
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from graph.state_dev import GraphState
from graph.utils.output_formatter import print_rag_documents

# init the web search tool with a limit of 3 results
web_search_tool = TavilySearchResults(k=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Augment document context with web search results.

    >> This is part of the Adaptive-RAG component's web search functionality.

    The function:
    1. Performs web search using Tavily API (limited to 3 results)
    2. Converts results to Document objects with metadata (URL, title)
    3. Combines new web documents with existing vector DB documents
    4. Tracks search attempts (max 2 per query)

    Args:
        state (GraphState): Current state with:
            - question: User query
            - documents: Optional existing docs
            - web_search_attempts: Search counter

    Returns:
        Dict[str, Any]: Updated state with:
            - documents: Combined docs (existing + web)
            - web_search_attempts: Incremented counter
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    web_search_attempts = state.get("web_search_attempts", 0) + 1
    
    docs = web_search_tool.invoke({"query": question})
    web_docs = [
        Document(
            page_content=doc["content"],
            metadata={
                "source": doc.get("url", "Web Search Result"),
                "title": doc.get("title", "Unknown Title")
            }
        ) 
        for doc in docs
    ]
    
    # Combine existing and new documents
    all_documents = documents + web_docs
    print_rag_documents(web_docs, "Web Search Results")
    
    return {
        "documents": all_documents,
        "web_search_attempts": web_search_attempts
    }
