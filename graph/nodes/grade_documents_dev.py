"""
Document Grading Node for Advanced RAG System

This script defines the grade_documents node, which is responsible for assessing
the relevance of retrieved documents to the user's question. 

>> This is part of the Corrective-RAG component of the system.

Key Components:
1. GraphState: Represents the shared state of the RAG graph.
2. retrieval_grader: Imported chain for grading document relevance.
3. grade_documents function: Core logic for the document grading node, grading 
each retrieved document based on relevance to the question (yes/no).

This node filters out irrelevant documents and determines if web search is necessary
based on having less than 3 relevant documents, with a maximum of 2 web search attempts.
"""

from typing import Any, Dict
from graph.chains.retrieval_grader_dev import retrieval_grader
from graph.state_dev import GraphState
from graph.utils.output_formatter import print_rag_documents


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grade the relevance of retrieved documents to the user's question.
    1. It extracts the question and documents from the current state.
    2. It uses the retrieval_grader to assess each document's relevance.
    3. It counts the number of relevant documents.
    4. It sets a flag for web search if fewer than 3 relevant documents are found, for information enrichment.
    5. It tracks and limits web search attempts to a maximum of 2.
    6. It returns an updated state with filtered documents and web search metadata.

    This function is typically called after document retrieval and before generation,
    ensuring that only relevant documents are used for answer generation while
    maintaining control over web search iterations.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    print_rag_documents(documents)
    filtered_docs = []
    relevant_count = 0
    
    # init web search attempts if not present
    web_search_attempts = state.get("web_search_attempts", 0)
    
    # Grade each document based on the page content and the question
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            relevant_count += 1
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    # Determine if web search is needed based on relevance count and attempts
    web_search = relevant_count < 3 and web_search_attempts < 2 # True/False
    
    if web_search:
        web_search_attempts += 1
        
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
        "web_search_attempts": web_search_attempts
    }

