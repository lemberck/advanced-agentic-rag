"""
Retrieval Grader for Advanced RAG System

This script defines a grading chain that evaluates whether retrieved documents
are relevant to the user's question.

>> This is part of the C-RAG (Corrective-RAG) component of the Advanced RAG System (first layer of C-RAG).

Key Components:
1. LLM: ChatAnthropic model with temperature 0 for consistent outputs
2. Structured Output: Pydantic model for binary grading results
3. Prompt: Custom prompt for document relevance assessment
4. Grading Chain: Pipeline that evaluates document relevance

The chain uses LCEL (LangChain Expression Language) to create a pipeline:
prompt -> LLM -> binary score
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from os import getenv
from dotenv import load_dotenv
load_dotenv()

# init ChatAnthropic model with temperature 0 for more consistent outputs
llm = ChatAnthropic(temperature=0, model=getenv("ANTHROPIC_MODEL"))

# Define a Pydantic model for structured output from the LLM
class GradeDocuments(BaseModel):
    """
    Pydantic model for the structured output of the retrieval grader.
    Returns a binary score (yes/no) indicating whether the retrieved document
    is relevant to the user's question.
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Create a structured output wrapper for the LLM based on the GradeDocuments model
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Define the custom system prompt for the retrieval grader
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# Create a ChatPromptTemplate for the retrieval grading task
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Combine the prompt and structured LLM grader into a single chain using LCEL
retrieval_grader = grade_prompt | structured_llm_grader
