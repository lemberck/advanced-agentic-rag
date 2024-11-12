"""
Rich Console Formatting Utilities for AWS Documentation Ingestion

This script provides console output formatting for the ingestion pipeline's
key events and metrics:
- Document processing progress tracking
- Status updates with color coding
- Summary statistics and metrics

Key Functions:
1. print_ingestion_start: Display initial ingestion configuration
2. print_document_status: Show per-document processing states
3. print_ingestion_summary: Present final ingestion metrics

Uses Rich library for consistent, colored console output across the ingestion workflow.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from datetime import datetime

console = Console()

def print_ingestion_start(docs_dir: str, total_files: int):
    """Print initial ingestion status."""
    table = Table(title="📚 AWS Documentation Ingestion", show_header=True)
    table.add_column("Status", style="cyan")
    table.add_column("Details", style="white")
    
    table.add_row("Directory", docs_dir)
    table.add_row("Files Found", str(total_files))
    table.add_row("Started At", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print(Panel(table, expand=False))

def print_document_status(status: str, filepath: str, details: str = None):
    """Print document processing status with rich formatting."""
    status_colors = {
        "NEW": "green",
        "MODIFIED": "yellow",
        "DELETED": "red",
        "SKIPPED": "blue"
    }
    
    text = Text()
    text.append("• ", style="white")
    text.append(f"[{status}] ", style=status_colors.get(status, "white"))
    text.append(filepath, style="white")
    if details:
        text.append(f"\n  └─ {details}", style="dim")
    
    console.print(text)

def print_ingestion_summary(total_files: int, new_docs: int, modified_docs: int, deleted_chunks: int, total_chunks: int):
    """Print final ingestion summary."""
    table = Table(title="📊 Ingestion Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="white", justify="right")
    
    table.add_row("Total Files Scanned", str(total_files))
    table.add_row("New Documents", str(new_docs))
    table.add_row("Modified Documents", str(modified_docs))
    table.add_row("Chunks Deleted", str(deleted_chunks))
    table.add_row("Total Chunks in VectorDB", str(total_chunks))
    
    console.print(Panel(table, expand=False))
