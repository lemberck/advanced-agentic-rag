"""
Answer Grader for Advanced RAG System

This script defines a grading chain to assess whether a generated answer
adequately addresses the user's question.

>> This is part of the Self-RAG component of the Advanced RAG System.

Key Components:
1. GradeRelevance: Pydantic model for structured output from the LLM.
2. ChatAnthropic: LLM model with temperature set to 0 for consistent outputs.
3. ChatPromptTemplate: Custom prompt template for answer assessment.
4. RunnableSequence: Combines the prompt and LLM for answer grading.

This grader ensures that generated responses are relevant to the original question (second layer of self-RAG).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from os import getenv

load_dotenv()

class GradeRelevance(BaseModel):
    """
    Pydantic model for the structured output of the answer grader.
    Returns a binary score (yes/no) indicating whether the answer is relevant
    and fully addresses the question asked.
    """

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# init llm with temperature 0 for consistent outputs
llm = ChatAnthropic(temperature=0,model=getenv("ANTHROPIC_MODEL"))

# define a structured output wrapper for the LLM based on the created GradeRelevance model
structured_llm_grader = llm.with_structured_output(GradeRelevance)

# create the system message for the grader
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes means that the answer resolves the question."""

# Create a prompt template for the answer grader
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# Combine the prompt and the structured LLM into a runnable sequence 'relevance_grader'
relevance_grader: RunnableSequence = answer_prompt | structured_llm_grader

"""
This grader is used in the graph to assess the relevance of generated answers.
By providing a binary score, it helps determine whether the RAG system should
continue with the current answer or attempt to improve it through further steps.
"""
