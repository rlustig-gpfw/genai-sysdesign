"""Part 3 - Corrective RAG-lite implementation using LangGraph.

This implementation focuses on:
- Intelligent routing between document knowledge and web search
- Relevance assessment of document chunks
- Combining multiple knowledge sources
- Handling information conflicts
"""

from typing import Dict, List, Optional, Any

from perplexia_ai.core.chat_interface import ChatInterface


# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.
class CorrectiveRAGChat(ChatInterface):
    """Week 2 Part 3 implementation for Corrective RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.search_tool = None
        self.document_paths = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for Corrective RAG.
        
        Students should:
        - Initialize the LLM
        - Set up document loading and processing
        - Create vector embeddings
        - Set up Tavily search tool
        - Build a Corrective RAG workflow using LangGraph
        """
        # TODO: Initialize LLM
        
        # TODO: Initialize embeddings
        
        # TODO: Set up Tavily search tool
        
        # TODO: Set paths to OPM documents
        # data_dir = Path("path/to/opm/documents")
        # self.document_paths = list(data_dir.glob("*.pdf"))
        
        # TODO: Process documents and create vector store
        # docs = self._load_and_process_documents()
        # self.vector_store = InMemoryVectorStore.from_documents(docs, self.embeddings)
        
        # TODO: Create the graph
        # Define nodes:
        # 1. Document retrieval node: Finds relevant document sections
        # 2. Relevance assessment node: Determines if retrieved documents are relevant
        # 3. Web search node: Performs web search if needed
        
        # Define the graph structure with conditional edges
        
        # Compile the graph
        pass
    
    def _load_and_process_documents(self) -> list[str]:
        """Load and process OPM documents."""
        # TODO: Implement document loading and processing
        # 1. Load the documents
        # 2. Split into chunks
        # 3. Return processed documents
        return []
    
    def _create_relevance_assessment_node(self):
        """Create a node that assesses document relevance."""
        # TODO: Implement relevance assessment node
        pass
    
    def _create_document_retrieval_node(self):
        """Create a node that retrieves relevant document sections."""
        # TODO: Implement document retrieval node
        pass
    
    def _create_web_search_node(self):
        """Create a node that performs web search when needed."""
        # TODO: Implement web search node
        pass
    
    def _should_use_web_search(self, state: Dict[str, Any]) -> bool:
        """Determine if web search should be used based on document relevance."""
        # TODO: Implement logic to decide when to use web search
        return False
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using Corrective RAG.
        
        Intelligently combines document knowledge with web search:
        - Uses documents when they contain relevant information
        - Falls back to web search when documents are insufficient
        - Combines information from both sources when appropriate
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response combining document and web knowledge
        """
        # TODO: Implement Corrective RAG processing
        # 1. Format the input message
        # 2. Run the graph
        # 3. Extract the response
        
        # This is just a placeholder
        return f"Corrective RAG result for: {message}" 
