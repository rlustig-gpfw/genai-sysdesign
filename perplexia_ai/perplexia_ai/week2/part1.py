"""Part 1 - Web Search implementation using LangGraph.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
"""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface


# TODO: Define state for the application.

# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.
class WebSearchChat(ChatInterface):
    """Week 2 Part 1 implementation for web search using LangGraph."""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for web search.
        
        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph for web search workflow
        """
        # Set up API key for Tavily
        # os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"
        
        # TODO: Initialize LLM
        
        # TODO: Initialize search tool
        
        # TODO: Create the graph
        # Define nodes:
        
        # Define the edges and the graph structure
        
        # Compile the graph
        pass
    
    def _create_search_node(self):
        """Create a node that performs web search."""
        # TODO: Implement search node
        pass
    
    def _create_process_results_node(self):
        """Create a node that processes and formats search results."""
        # TODO: Implement process results node
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using web search.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response with search results
        """
        # TODO: Implement web search processing
        # 1. Format the input message
        # 2. Run the graph
        # 3. Extract the response
        
        # This is just a placeholder
        return f"Web search result for: {message}" 