"""
Chains Unit Tests for AWS Documentation RAG System

This script contains unit tests for the RAG system components that handle AWS documentation,
specifically focused on SageMaker documentation retrieval and question answering.

Key Test Areas:
1. Document Retrieval: Tests accurate retrieval of AWS documentation from the vector store
2. Answer Generation: Verifies generation of accurate responses about AWS services
3. Hallucination Prevention: Ensures responses are grounded in official AWS documentation
4. Query Routing: Tests correct handling of AWS-related vs unrelated queries

The test suite validates:
- Accurate retrieval and answering of SageMaker-related questions
- Proper handling of out-of-scope queries (routing to web search)
- Factual consistency with AWS documentation
- Source attribution to official AWS documentation

Note: These tests use Claude 3.5 Sonnet for evaluation, ensuring consistent 
grading of response quality and factual accuracy against AWS documentation.
"""

# pytest graph/chains/tests/test_chains_dev.py -v -s

from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

from graph.chains.generation_dev import generation_chain
from graph.chains.hallucination_grader_dev import GradeHallucinations, hallucination_grader
from graph.chains.retrieval_grader_dev import GradeDocuments, retrieval_grader
from graph.chains.router_dev import RouteQuery, question_router
from ingestion_dev import retriever

def test_generation_chain() -> None:
    """
    Test the generation chain's ability to produce responses.
    
    This test retrieves documents for a given question, passes them to the
    generation chain, and prints the result.
    """
    question = "sagemaker"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)

def test_retrival_grader_answer_yes() -> None:
    """
    Test the retrieval grader with a relevant document.
    
    This test checks if the grader correctly identifies a relevant document
    by returning a 'yes' score.
    """
    question = "sagemaker"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"

def test_retrival_grader_answer_no() -> None:
    """
    Test the retrieval grader with an irrelevant document.
    
    This test checks if the grader correctly identifies an irrelevant document
    by returning a 'no' score.
    """
    question = "sagemaker"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"

def test_hallucination_grader_answer_yes() -> None:
    """
    Test the hallucination grader with a grounded response.
    
    This test checks if the grader correctly identifies a response that is
    grounded in the provided documents.
    """
    question = "sagemaker"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    """
    Test the hallucination grader with a hallucinated response.
    
    This test checks if the grader correctly identifies a response that is not
    grounded in the provided documents (i.e., a hallucination).
    """
    question = "sagemaker"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    """
    Test the query router for vectorstore routing.
    
    This test checks if the router correctly routes a query related to the vectorstore content to it.
    """
    question = "sagemaker"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    """
    Test the query router for web search routing.
    
    This test checks if the router correctly routes a query not related to the vectorstore to web search.
    """
    question = "how to make pizza in 3 steps"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
