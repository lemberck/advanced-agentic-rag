"""
Query Router for Advanced RAG System

This script defines a routing chain to determine the most appropriate datasource
for a given user query.

>> This is part of the Adaptive-RAG component of the Advanced RAG System.

Key Components:
1. LLM: ChatAnthropic model with temperature set to 0 for consistent outputs.
2. Structured Output: Uses Pydantic for type validation and structured LLM output.
3. Prompt: A custom prompt template for query routing.
4. Routing Chain: Combines the prompt and LLM for query routing.

The chain utilizes LangChain Expression Language (LCEL) for composition.
"""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from os import getenv
from dotenv import load_dotenv
load_dotenv()


class RouteQuery(BaseModel):
    """
    Pydantic model for the structured output of the query router.
    Returns a routing decision (vectorstore/websearch) indicating which
    datasource is most appropriate for the user's question or websearch to enhance the context.
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
    
# init ChatAnthropic model with temperature 0 for more consistent outputs
llm = ChatAnthropic(temperature=0, model=getenv("ANTHROPIC_MODEL"))

# Create a structured output wrapper for the LLM based on the RouteQuery model
structured_llm_router = llm.with_structured_output(RouteQuery)

# Define the custom system prompt for the query router
####### The vecstore topics are hardcoded to this prompt!! #######
# TODO: Make system prompt dynamic by adding a node to extract topics from ingested vectorstore content
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related AWS documentation, AWS Sagemaker, and AWS resources.
Use the vectorstore for questions on these topics. For all else, use web-search."""

# Create a ChatPromptTemplate for the query routing task
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Combine the prompt and structured LLM router into a single chain using LCEL
question_router = route_prompt | structured_llm_router
