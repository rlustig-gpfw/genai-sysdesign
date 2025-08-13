"""Part 1 - Web Search implementation using LangGraph.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
"""

from typing import Dict, List, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from perplexia_ai.core.chat_interface import ChatInterface


class WebSearchState(TypedDict):
    query: str
    search_results: List[Dict[str, str]]
    response: str


class WebSearchChat(ChatInterface):
    """Week 2 Part 1 implementation for web search using LangGraph."""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None

        self._web_search_summarizer_chain = None
    
    def initialize(self) -> None:
        """Initialize components for web search.
        
        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph for web search workflow
        """       
        # Initialize LLM
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        
        # Initialize search tool
        self.search_tool = TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            search_depth="advanced",
        )

        self._create_web_search_summarizer_chain()        

        # Create the graph
        graph_builder = StateGraph(WebSearchState)

        # Add nodes to the graph
        graph_builder.add_node("search", self._create_search_node)
        graph_builder.add_node("process_results", self._create_web_search_summarizer_node)

        # Add edges to the graph
        graph_builder.add_edge(START, "search")
        graph_builder.add_edge("search", "process_results")
        graph_builder.add_edge("process_results", END)

        # Compile the graph
        self.graph = graph_builder.compile()
    
    def _create_web_search_summarizer_chain(self):
        """Create a chain that processes search results."""
        web_search_summarizer_prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that summarizes search results based on the user's query.
            You will need to provide a citation for each result.
            
            Here is the user's query:
            {query}

            Here is the list of search results:
            {search_results}

            Provide the response in the following example format below.
            If the answer is factual, the format below can be shortened to a single sentence with sources.
            
            1. Breakthrough in Error Correction
            Researchers at MIT announced a new quantum error correction method that improves qubit stability by 45%.

            2. Commercial Quantum Computing Milestones
            IBM's latest quantum processor reached 433 qubits, bringing practical quantum advantage closer.

            3. Quantum Machine Learning Applications
            New algorithms demonstrate 10x speedup for specific machine learning tasks.

            Sources:
            - MIT Technology Review (April 2023) (url)
            - IBM Research Blog (March 2023) (url)
            - Nature Quantum Information (May 2023) (url)

            The url should be a valid hyperlink from the search results.
            """
        )
        self._web_search_summarizer_chain = web_search_summarizer_prompt | self.llm | StrOutputParser()

    def _create_search_node(self, state: WebSearchState):
        """Create a node that performs web search."""
        results = self.search_tool.invoke({"query": state["query"]})
        return {"search_results": results}
    
    def _create_web_search_summarizer_node(self, state: WebSearchState):
        """Create a node that processes and formats search results."""
        response = self._web_search_summarizer_chain.invoke(
            {
                "query": state["query"],
                "search_results": state["search_results"]
            }
        )
        return {"response": response}
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using web search.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response with search results
        """
        result = self.graph.invoke({"query": message})   
        return result["response"]