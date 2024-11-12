"""
Hallucination Grader for Advanced RAG System

This script defines a grading chain that checks if an LLM's response
is factually supported by the provided context documents (grounded to data).

>> This is part of the Self-RAG component of the Advanced RAG System (first layer of self-RAG).

Key Components:
1. LLM: ChatAnthropic model with temperature 0 for consistent outputs
2. Structured Output: Pydantic model for binary grading results
3. Prompt: Custom prompt for fact-checking assessment
4. Grading Chain: Pipeline that evaluates response accuracy

The chain uses LCEL (LangChain Expression Language) to create a pipeline:
prompt -> LLM -> binary score
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_anthropic import ChatAnthropic
from os import getenv
from dotenv import load_dotenv

load_dotenv()

# init ChatAnthropic model with temperature 0 for more consistent outputs
llm = ChatAnthropic(temperature=0, model=getenv("ANTHROPIC_MODEL"))

# Define a Pydantic model for binary output from the LLM
class GradeHallucinations(BaseModel):
    """
    Pydantic model for structured output from the LLM.
    Returns a binary score (yes/no) indicating whether the generated response
    is factually supported by the provided context documents.
    """

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Create a structured output wrapper for the LLM based on the created GradeHallucinations model
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# define the system prompt for the hallucination grader
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# create a ChatPromptTemplate for the hallucination grading task
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# Combine the prompt and structured LLM grader into a single chain using LCEL
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

