"""
Generation Chain for Advanced RAG System

This script defines a generation chain that produces responses to user questions
based on provided context documents.

Key Components:
1. LLM: ChatAnthropic model with temperature 0 for consistent outputs
2. Prompt: Pre-defined RAG prompt from Langchain hub
3. Output Parser: Converts LLM output to string format

The chain uses LCEL (LangChain Expression Language) to create a pipeline:
prompt -> LLM -> string output
"""

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from os import getenv
from dotenv import load_dotenv
load_dotenv()

# init ChatAnthropic model with temperature 0 for consistent outputs
llm = ChatAnthropic(temperature=0, model=getenv("ANTHROPIC_MODEL"))

# Pull a pre-defined RAG prompt from langchain's prompt hub
# The "rlm/rag-prompt" is a specific prompt designed for RAG tasks
prompt = hub.pull("rlm/rag-prompt")

#print(prompt.messages[0].prompt.template) # To check the prompt message


# create the chain pipeline: prompt -> LLM -> StrOutputParser
generation_chain = prompt | llm | StrOutputParser()
