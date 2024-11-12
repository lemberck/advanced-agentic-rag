"""
Output Formatting Utilities for Advanced RAG System

This script provides console output formatting for the RAG pipeline's
key events and decisions:
- Pipeline phase transitions and progress
- Decision points with explanations
- Quality evaluation results
- Retrieved document displays
- Query and answer formatting

Key Functions:
1. print_rag_phase: Display current pipeline phase
2. print_rag_decision: Show decision points and their rationale
3. print_rag_evaluation: Present quality check results
4. print_rag_documents: Format retrieved documents with metadata
5. print_rag_query: Display user's initial query
6. print_rag_final_result: Show final answer with sources

Uses Rich library for consistent, colored console output across the RAG workflow.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from typing import List, Tuple

console = Console()

def print_rag_phase(title: str, message: str, style: str = "blue"):
    """Print a RAG phase header with descriptive message."""
    console.print(Panel(
        Text.assemble(
            ("🔄 RAG PHASE: ", "bold white"),
            (f"{title}\n", f"bold {style}"),
            message
        ),
        border_style=style,
        title=f"Step: {title}"
    ))

def print_rag_decision(decision: str, details: str = "", style: str = "yellow"):
    """Print a RAG decision point with explanation."""
    titles = {
        "VECTOR STORE": "Source Selection",
        "WEB SEARCH": "Source Selection",
        "RETRIEVING DOCUMENTS": "Retrieval Process",
        "PROCEED TO GENERATION": "Document Assessment",
        "WEB SEARCH NEEDED": "Document Assessment"
    }
    title = titles.get(decision, "Decision Point")
    
    console.print(Panel(
        Text.assemble(
            ("🤔 DECISION POINT: ", "bold white"),
            (f"{decision}\n", f"bold {style}"),
            details
        ),
        border_style=style,
        title=title
    ))

def print_rag_evaluation(title: str, checks: List[Tuple[str, bool]], style: str = "green", explanation: str = None, next_step: str = None):
    """Print evaluation results with optional explanation and next steps."""
    console = Console()
    table = Table(title=f"📊 QUALITY CHECK: {title}", style=style)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="white")
    
    for check, passed in checks:
        status = "✅" if passed else "❌"
        table.add_row(check, status)
    
    if explanation:
        table.add_row("Explanation", explanation)
    if next_step:
        table.add_row("Next Step", next_step)
        
    console.print(Panel(table, border_style=style, title=f"Evaluation: {title}"))
    print("\n")

def print_rag_documents(documents: list, title: str = "Retrieved Documents"):
    """Print retrieved documents with their sources."""
    console.print(Panel(
        Text.assemble(
            ("📚 SOURCES:\n", "bold white"),
            *[
                Text.assemble(
                    (f"\n[{i+1}] ", "bold yellow"),
                    (f"Source: {doc.metadata.get('source', 'Unknown source')}\n", "bold blue"),
                    (f"Title: {doc.metadata.get('title', 'No title')}\n", "dim blue"),
                    (f"{doc.page_content[:200]}...\n", "white")
                )
                for i, doc in enumerate(documents)
            ]
        ),
        border_style="blue",
        title=title,
        expand=True
    ))

def print_rag_query(question: str):
    """Print the initial user query in a formatted table."""
    console = Console()
    table = Table(title="🤔 User Query", style="blue")
    table.add_column("Question", style="cyan")
    table.add_row(question)
    console.print(table)
    print("\n")

def print_rag_final_result(question: str, answer: str, sources: list = None):
    """Print the final question, answer and sources in a formatted table."""
    console = Console()
    table = Table(title="🎯 Final Result", style="green")
    table.add_column("Component", style="cyan")
    table.add_column("Content", style="white")
    table.add_row("Question", question)
    table.add_row("Answer", answer)
    if sources:
        sources_text = "\n".join([f"- {src}" for src in sources])
        table.add_row("Sources", sources_text)
    console.print(table)
    print("\n")