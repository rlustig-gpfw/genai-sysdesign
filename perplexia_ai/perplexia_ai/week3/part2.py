"""Part 2 - Agentic RAG implementation.

This implementation focuses on:
- Building an Agentic RAG system with dynamic search strategy
- Using LangGraph for controlling the RAG workflow
- Evaluating retrieved information quality
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph

# For document retrieval
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.vectorstores import VectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores.base import VectorStoreRetriever

from perplexia_ai.core.chat_interface import ChatInterface


# NOTE: The TODOs here are ONLY a guideline, feel free to change the structure as you see fit.
class AgenticRAGChat(ChatInterface):
    """Week 3 Part 2 implementation focusing on Agentic RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.retriever_tool = None
        self.tavily_search_tool = None
        self.agent = None
        self.document_evaluator = None
        self.synthesizer = None
        self.workflow = None
    
    def initialize(self) -> None:
        """Initialize components for the Agentic RAG system.
        
        Students should:
        - Initialize models and embeddings
        - Load and index documents from Week 2
        - Create tools for the agent
        - Set up the agentic RAG workflow
        """
        # TODO: Initialize models (LLM, embeddings)
        
        # TODO: Load documents (reuse from Week 2)
        
        # TODO: Create vector store and retriever
        
        # TODO: Create tools (retriever tool, tavily search tool)
        
        # TODO: Create document evaluator
        
        # TODO: Create synthesizer
        
        # TODO: Create the agent and workflow
    
    def _load_documents(self) -> List[Document]:
        """Load the OPM documents from Week 2.
        
        Returns:
            List[Document]: List of loaded documents
        """
        # TODO: Implement document loading (reuse code from Week 2)
    
    def _setup_vector_store(self) -> Tuple[VectorStore, VectorStoreRetriever]:
        """Set up the vector store and retriever.
        
        Returns:
            Tuple[VectorStore, VectorStoreRetriever]: The vector store and retriever
        """
        # TODO: Create vector store with OPM documents
        # TODO: Create retriever with appropriate parameters
    
    def _create_tools(self) -> List[Any]:
        """Create and return the tools for the agent.
        
        Returns:
            List[Any]: List of tool objects
        """
        # TODO: Create retriever tool
        # TODO: Create Tavily search tool
    
    def _create_document_evaluator(self) -> Any:
        """Create a document evaluator that assesses retrieved document quality.
        
        Returns:
            Any: The document evaluator runnable
        """
        # TODO: Create an evaluator prompt
        # TODO: Create the evaluator chain
    
    def _create_synthesizer(self) -> Any:
        """Create a synthesizer that combines retrieved information.
        
        Returns:
            Any: The synthesizer runnable
        """
        # TODO: Create a synthesizer prompt
        # TODO: Create the synthesizer chain
    
    def _create_workflow(self) -> Any:
        """Create the agentic RAG workflow using LangGraph.
        
        Returns:
            Any: The compiled workflow
        """
        # TODO: Create the retrieval agent
        # TODO: Define workflow graph with nodes for agent, evaluator, synthesizer
        # TODO: Add conditional edges based on evaluation results
        # TODO: Set entry point and compile graph
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the Agentic RAG system.
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages
            
        Returns:
            str: The assistant's response
        """
        # TODO: Prepare input state with query and tracking variables
        # TODO: Run the workflow and return the result 
        return "Hello world from part 2"
