"""Part 2 - Document RAG implementation using LangGraph.

This implementation focuses on:
- Setting up document loading and processing
- Creating vector embeddings and storage
- Implementing retrieval-augmented generation
- Formatting responses with citations from OPM documents
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from perplexia_ai.core.chat_interface import ChatInterface


class DocumentRAGState(TypedDict):
    query: str
    documents: List[Document]
    response: str


class ResponseGenerationStructure(BaseModel):
    answer: str = Field(description="The answer to the user's query")
    sources: List[str] = Field(description="Names of the source documents that were used to answer the user's query")


# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.
class DocumentRAGChat(ChatInterface):
    """Week 2 Part 2 implementation for document RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings_model = None
        self.vector_store = None
        self.document_paths = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for document RAG.
        
        Students should:
        - Initialize the LLM
        - Set up document loading and processing
        - Create vector embeddings
        - Build retrieval system
        - Create LangGraph for RAG workflow
        """
        # Initialize LLM
        self.llm = init_chat_model(
            model="gpt-4o-mini",
            temperature=0,
        )
        
        # Initialize embeddings
        self.embeddings_model = OpenAIEmbeddings()

        # Set paths to OPM documents
        data_dir = Path(os.getcwd()) / "perplexia_ai" / "docs"
        self.document_paths = list(data_dir.glob("*.pdf"))

        # Process documents and create vector store
        docs = self._load_and_process_documents()
        self.vector_store = InMemoryVectorStore.from_documents(docs, self.embeddings_model)
                
        # Define the edges and the graph structure
        graph_builder = StateGraph(DocumentRAGState)
        graph_builder.add_node("retrieval", self._create_retrieval_node)
        graph_builder.add_node("generation", self._create_generation_node)

        graph_builder.add_edge(START, "retrieval")
        graph_builder.add_edge("retrieval", "generation")
        graph_builder.add_edge("generation", END)
        self.graph = graph_builder.compile()
    
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
    
    def _create_retrieval_node(self, state: DocumentRAGState):
        """Create a node that retrieves relevant document sections."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        relevant_docs = retriever.invoke(state["query"])
        return {"documents": relevant_docs}
    
    def _create_generation_node(self, state: DocumentRAGState):
        """Create a node that generates responses using retrieved context."""
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that generates responses based on the user's query and the list of retrieved document sections.
            Here is the user's query:
            {query}
            Here is the list of retrieved document sections:
            {documents}
            
            If the query is not answerable from the retrieved document sections, explain why and briefly mention a summary of what relevant information you can provide.
            No sources should be provided if the query is not answerable from the retrieved document sections.
            
            Format the response with the answer and the sources:
            Answer: [Answer here within 2-3 sentences only]
            Sources:
            - <Filename of the source document, not the title>

            Example response 1:
            User: What is the average salary of a software engineer in the United States?
            Documents: [Text from a document about the average salary of a software engineer in the United States]
            Answer: The average salary of a software engineer in the United States is $100,000.
            Sources:
            - [Name of the source document]

            Example response 2:
            User: Why is the sky blue?
            Documents: [Text from documents about the sky, but nothing describing what makes the sky blue]
            Answer: I don't know. This question is not answerable from the retrieved documents.
            """
        )
        chain = prompt | self.llm.with_structured_output(ResponseGenerationStructure)
        response = chain.invoke({"query": state["query"], "documents": state["documents"]})

        response_str = f"{response.answer}"
        if response.sources:
            response_str += f"\n\nSources:\n"
            for source in response.sources:
                response_str += f"- {source}\n"

        return {"response": response_str}
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using document RAG.
        
        Should reject queries that are not answerable from the OPM documents.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on document knowledge
        """
        result = self.graph.invoke({"query": message})
        print(result)
        return result["response"]
