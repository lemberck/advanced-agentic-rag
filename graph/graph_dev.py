import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

"""
Advanced RAG System Graph Definition

This script implements a directed graph workflow combining four RAG components:

1. Core-RAG: Basic retrieval and generation
   - RETRIEVE node: Vector DB document fetching
   - GENERATE node: Response creation from context

2. Corrective-RAG (C-RAG): Document quality control
   - GRADE_DOCUMENTS node: Relevance assessment
   - decide_to_generate(): Conditional edge for web search decisions
   - Enforces minimum 3 relevant retrieved documents

3. Self-RAG: Response quality verification
   - grade_generation_grounded_in_documents_and_question(): 
     - Checks for hallucinations
     - Verifies generated answer relevance
   - Triggers web search for improvement if needed

4. Adaptive-RAG: Dynamic source selection
   - route_question(): Entry point for source routing
   - web_search node: External data retrieval
   - Limited to 2 web search attempts

Graph Structure:
- Nodes: RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
- Conditional Edges: For C-RAG and Self-RAG decisions
- Entry Points: Adaptive routing between sources
- State Management: Via GraphState shared memory

Memory Management:
- SQLite Checkpointing: Persistent state storage
- In-Memory Option: For faster development/testing
- Configurable via SqliteSaver
- Tracks graph execution state and decisions

Dependencies:
- LangGraph: For graph construction and execution
- Custom Chains: For routing, grading, and generation
- Rich: For console output formatting
"""

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader_dev import relevance_grader
from graph.chains.hallucination_grader_dev import hallucination_grader
from graph.chains.router_dev import question_router, RouteQuery
from graph.consts_dev import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes.generate_dev import generate
from graph.nodes.grade_documents_dev import grade_documents
from graph.nodes.retrieve_dev import retrieve
from graph.nodes.web_search_dev import web_search
from graph.state_dev import GraphState
from graph.utils.output_formatter import print_rag_phase, print_rag_decision, print_rag_evaluation

# Load env vars and set up memory management
load_dotenv()

# MemorySaver for development/testing (faster, no persistence needed for the PoC)
memory = MemorySaver()

### C-RAG
def decide_to_generate(state):
    """
    >> This will be the function of the C-RAG component conditional edge.
    Decides whether to generate a response or perform a web search based on document relevance.
    
    1. It checks if web search is needed based on having less than 3 relevant documents.
    2. It enforces a maximum of 2 web search attempts.
    3. It logs the decision and attempt count for debugging purposes.
    4. It returns the next node to be executed (WEBSEARCH or GENERATE).
    
    Args:
        state (GraphState): The current state of the RAG graph containing:
            - web_search: bool indicating if web search is needed
            - web_search_attempts: int tracking number of web searches performed
    
    Returns:
        str: The next node to be executed (WEBSEARCH or GENERATE).
            WEBSEARCH if less than 3 relevant docs and attempts < 2
            GENERATE otherwise
    """
    print_rag_phase("DOCUMENT ASSESSMENT", "Evaluating retrieved documents for relevance")

    if state["web_search"] and state["web_search_attempts"] < 2:
        print_rag_decision("WEB SEARCH NEEDED", 
                         f"Found less than 3 relevant documents. Attempt {state['web_search_attempts']}/2")
        return WEBSEARCH
    else:
        if state["web_search_attempts"] >= 2:
            print_rag_decision("FORCED GENERATION", 
                             "Maximum web search attempts reached - proceeding with available documents")
        else:
            print_rag_decision("PROCEED TO GENERATION", 
                             "Found sufficient relevant documents")
        return GENERATE

### Self-RAG
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """Grade if the generation is grounded in documents and answers the question."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    web_search_attempts = state.get("web_search_attempts", 0)
    
    # First check: Hallucination Check
    print_rag_phase("HALLUCINATION CHECK", 
                    "Analyzing response for factual accuracy against source documents")
    
    hallucination_score = hallucination_grader.invoke({
        "documents": documents,
        "generation": generation
    })
    
    if not hallucination_score.binary_score and web_search_attempts < 2:
        print_rag_evaluation(
            "Factual Accuracy Issues", 
            [("Response not grounded in documents", True)],
            style="red",
            explanation="Generated response contains information not found in source documents",
            next_step="Will attempt web search for additional information"
        )
        return "not_useful_continue"
        
    print_rag_evaluation(
        "Factual Accuracy Verification", 
        [("Response grounded in documents", True)]
    )
    
    # Second check: Answer Relevance
    print_rag_phase("ANSWER RELEVANCE", 
                    "Evaluating if response directly addresses user question")
    
    answer_score = relevance_grader.invoke({
        "question": question,
        "generation": generation
    })
    
    if answer_score.binary_score or web_search_attempts >= 2:
        print_rag_evaluation(
            "Answer Relevance Verification", 
            [("Response directly addresses question", True)],
            style="green"
        )
        return "useful"
    else:
        print_rag_evaluation(
            "Answer Relevance Issues", 
            [("Response does not fully address question", True)],
            style="yellow",
            explanation="Current answer needs improvement",
            next_step="Will attempt web search for additional information"
        )
        return "not_useful_continue"

### Adaptive RAG
def route_question(state: GraphState) -> str:
    """
    Routes the initial question to either web search or RAG based on the question type.
    
    >> This will be the function of the Adaptive RAG component conditional entry point.
    
    This function acts as the entry point for the RAG system:
    1. It uses the question_router to determine the appropriate data source.
    2. It logs the routing decision for debugging purposes.
    3. It returns the initial node to be executed (WEBSEARCH or RETRIEVE).
    
    Args:
        state (GraphState): The current state of the RAG graph.
    
    Returns:
        str: The initial node to be executed (WEBSEARCH or RETRIEVE).
    """
    print_rag_phase("ROUTING", "Analyzing question to determine optimal data source")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == "websearch":
        print_rag_decision("WEB SEARCH", "Question requires external or real-time information")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print_rag_decision("VECTOR STORE", "Question matches internal knowledge base topics")
        return RETRIEVE

def decide_next_after_generation(state: GraphState) -> str:
    """
    Decides the next step after evaluating a generated response.
    
    This function determines whether to:
    1. Perform another web search (if attempts < 2)
    2. End the process (if attempts >= 2 or response is useful)
    
    Args:
        state (GraphState): The current state containing web_search_attempts counter
    
    Returns:
        str: One of the following values:
            - "not_useful_continue": for web search with attempts < 2
            - "not_useful_end": when max attempts reached
    """
    if state.get("web_search_attempts", 0) < 2:
        return "not_useful_continue"
    return "not_useful_end"

# Initialize the StateGraph with the GraphState structure
workflow = StateGraph(GraphState)

### Add nodes to the graph
# workflow.add_node(node_name, py_function_with_node_logic)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

### Set the CONDITIONAL entry point for the graph, using the routing of Adaptive RAG.
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH, # If the route_question function returns WEBSEARCH, go to WEBSEARCH.
        RETRIEVE: RETRIEVE, # If the route_question function returns RETRIEVE, go to RETRIEVE.
    },
)

# Define simple edges between nodes
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

### Set the CONDITIONAL edge for the Adaptive RAG component.
workflow.add_conditional_edges(
    GRADE_DOCUMENTS, # The node with the return value to be evaluated.
    decide_to_generate, # The conditional function to be executed.
    {
        WEBSEARCH: WEBSEARCH, # If the decide_to_generate function returns WEBSEARCH, go to WEBSEARCH.
        GENERATE: GENERATE, # If the decide_to_generate function returns GENERATE, go to GENERATE.
    },
)

# Define simple edges between nodes
workflow.add_edge(WEBSEARCH, GENERATE)

### Set the CONDITIONAL edge for the Self-RAG component.
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,  # good answer, end here
        "not_useful_continue": WEBSEARCH,  # try web search
    }
)

# Compile the workflow into an executable application
app = workflow.compile(checkpointer=memory)
