"""
RAG System Main Execution Script
================================

This script serves as the main entry point for the Advanced Retrieval-Augmented Generation (RAG) system.
It demonstrates how to use the RAG graph defined in graph.py to process questions and generate answers.

Key components:
1. Environment setup: Loads environment variables
2. Graph import: Imports the compiled RAG graph application
3. Query execution: Runs a sample question through the RAG system
4. Result streaming: Processes and displays the output from each node in the graph with streaming process.

The script showcases the real-time operation of the RAG system, allowing for
step-by-step monitoring of the question processing and answer generation pipeline.
"""

from dotenv import load_dotenv
load_dotenv()

from graph.graph_dev import app
from graph.utils.output_formatter import print_rag_query, print_rag_final_result
from pathlib import Path

# Define a sample user question
question1 = "Who won the last superbowl?" #info NOT in vecstore OK
question2 = "How to check if an endpoint is KMS encrypted?" #info in vecstore OK
question3 = "What is Amazon SageMaker?" #info in vecstore OK
question4 = "What is SageMaker?" #info in vecstore OK

###
question5 = "What are all AWS regions where SageMaker is available?" #info in vecstore?
###

question6 = "What are SageMaker Geospatial capabilities?" #info in vecstore OK

question7 = "What is amazon eventbridge and how it relates to sagemaker?" #info in vecstore OK

inputs = {"question": question6}

# Print initial query table
print_rag_query(inputs["question"])

for output in app.stream(inputs, config={"configurable": {"thread_id": "2"}}):
    if 'generate' in output:
        answer = output['generate']['generation']
        # Handle sources differently based on whether they're local files or URLs
        sources = [
            Path(doc.metadata.get('source')).name 
            if doc.metadata.get('source').startswith('/') 
            else doc.metadata.get('source')
            for doc in output['generate'].get('documents', [])
        ]
        print_rag_final_result(inputs["question"], answer, sources)