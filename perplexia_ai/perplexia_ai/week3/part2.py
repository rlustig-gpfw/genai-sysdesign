"""Part 2 - Agentic RAG implementation.

This implementation focuses on:
- Building an Agentic RAG system with dynamic search strategy
- Using LangGraph for controlling the RAG workflow
- Evaluating retrieved information quality
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph

# For document retrieval
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.vectorstores import VectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores.base import VectorStoreRetriever

from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


DOC_EVALUATION_PROMPT = PromptTemplate(
    """
    You are an expert at evaluating the relevance of a retrieved document to a user query.
    Here is the user query:
    {query}
    Here is the retrieved document:
    {relevant_docs}

    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    Return a binary score 'yes' or 'no' to indicate if the document is relevant to the user query.
    """
)


class DocumentEvaluationStructure(BaseModel):
    """Structure for the document evaluation."""
    is_relevant: str = Field(description="Whether the document is relevant to the user query")


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
        # Initialize LLM
        self.llm = init_chat_model(
            model="gpt-4o",
            temperature=0,
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()

        # Set paths to OPM documents
        data_dir = Path(os.getcwd()) / "perplexia_ai" / "docs"
        self.document_paths = list(data_dir.glob("*.pdf"))

        # Process documents and create vector store
        docs = self._load_and_process_documents()
        self.vector_store, self.retriever = self._setup_vector_store(docs)
        
        self.tools = self._create_tools()
        
        # TODO: Create document evaluator
        
        # TODO: Create synthesizer
        
        # TODO: Create the agent and workflow
    
    def _load_and_process_documents(self) -> list[Document]:
        """Load and process OPM documents."""
        # 1. Load the documents
        # 2. Split into chunks
        # 3. Return processed documents
        all_pages = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for path in self.document_paths:
            all_pages.extend(PyPDFLoader(path).load())
        return text_splitter.split_documents(all_pages)
    
    def _setup_vector_store(self, docs: List[Document]) -> Tuple[VectorStore, VectorStoreRetriever]:
        """Set up the vector store and retriever.
        
        Returns:
            Tuple[VectorStore, VectorStoreRetriever]: The vector store and retriever
        """
        vector_store = InMemoryVectorStore.from_documents(documents=docs, embedding=self.embeddings)
        retriever = vector_store.as_retriever()
        return vector_store, retriever
    
    def _create_tools(self) -> List[Any]:
        """Create and return the tools for the agent.
        
        Returns:
            List[Any]: List of tool objects
        """
        retriever_tool = create_retriever_tool(
            self.retriever,
             "OPM_Retriever",
             "Search and retrieve information from OPM documents",
        )

        search_tool = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            search_depth="advanced",
        )
        return [retriever_tool, search_tool]
    
    def _create_document_evaluator(self) -> Any:
        """Create a document evaluator that assesses retrieved document quality.
        
        Returns:
            Any: The document evaluator runnable
        """
        chain = DOC_EVALUATION_PROMPT | self.llm.with_structured_output(DocumentEvaluationStructure)
        # TODO: determine if this evaluates one document or multiple documents at a time
        return chain
    
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
