"""Part 3 - Corrective RAG-lite implementation using LangGraph.

This implementation focuses on:
- Intelligent routing between document knowledge and web search
- Relevance assessment of document chunks
- Combining multiple knowledge sources
- Handling information conflicts
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from perplexia_ai.core.chat_interface import ChatInterface


class CorrectiveRAGState(TypedDict):
    query: str
    documents: List[Document]
    relevance_score: float
    web_search_results: List[Dict[str, str]]
    response: str


class ResponseGenerationStructure(BaseModel):
    answer: str = Field(description="The answer to the user's query")
    sources: List[str] = Field(description="Names of the source documents that were used to answer the user's query")


class RelevanceAssessmentStructure(BaseModel):
    scores: List[int] = Field(description="Relevance scores of each of the retrieved documents to the user's query")


class CorrectiveRAGChat(ChatInterface):
    """Week 2 Part 3 implementation for Corrective RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings_model = None
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
        
        # Set up Tavily search tool
        self.search_tool = TavilySearchResults(max_results=5)
                
        # Create the graph and define nodes
        graph_builder = StateGraph(CorrectiveRAGState)
        graph_builder.add_node("retrieval", self._create_document_retrieval_node)
        graph_builder.add_node("relevance", self._create_relevance_assessment_node)
        graph_builder.add_node("web_search", self._create_web_search_node)
        graph_builder.add_node("rag_response", self._create_rag_response_node)
        graph_builder.add_node("web_search_response", self._create_web_search_response_node)

        # Define graph with conditional edges
        graph_builder.add_edge(START, "retrieval")
        graph_builder.add_edge("retrieval", "relevance")
        graph_builder.add_conditional_edges(
            "relevance",
            self._should_use_web_search, {
                True: "web_search",
                False: "rag_response",
            },
        )
        graph_builder.add_edge("web_search", "web_search_response")
        graph_builder.add_edge("web_search_response", END)
        graph_builder.add_edge("rag_response", END)

        self.graph = graph_builder.compile()
        
    
    def _load_and_process_documents(self) -> list[str]:
        """Load and process OPM documents."""
        all_pages = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for path in self.document_paths:
            all_pages.extend(PyPDFLoader(path).load())
        return text_splitter.split_documents(all_pages)
    
    def _create_relevance_assessment_node(self, state: CorrectiveRAGState):
        """Create a node that assesses document relevance."""
        prompt = PromptTemplate.from_template(
            """
            You are a strict relevance judge. For each retrieved document chunk, decide whether it directly helps answer the user's query as written.

            Apply this rubric conservatively and base your decision on the query itself:
            1) Extract the query's core facets: subject/entity, domain/role/industry, topic/aspect, task being asked (e.g., explain, compare, list), time period, location, and any explicit constraints.
            2) A document is relevant (1) only if it contains explicit information that directly contributes to answering the task about the same subject/domain and topic.
            3) If the query constrains any facet (e.g., domain/role, time period, location, organization), treat that constraint as required unless the query clearly allows generalization. If the document does not satisfy a required facet, mark 0.
            4) Mark 0 for documents that are adjacent or generic (e.g., background definitions, general policies, or tangential mentions) when they do not address the query's specific task and constrained facets.
            5) Do not infer missing details or assume applicability across domains, time periods, or locations. If uncertain, mark 0.
            6) Only return one rating per document.

            User's query: {query}
            Retrieved documents: {documents}

            Output: Return a list of binary ratings (0 or 1), only one per document.
            """
        )
        chain = prompt | self.llm.with_structured_output(RelevanceAssessmentStructure)
        relevance = chain.invoke({"query": state["query"], "documents": state["documents"]})
        relevance_score = sum(relevance.scores) / len(relevance.scores)
        return {"relevance_score": relevance_score}
    
    def _create_document_retrieval_node(self, state: CorrectiveRAGState):
        """Create a node that retrieves relevant document sections."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        relevant_docs = retriever.invoke(state["query"])
        return {"documents": relevant_docs}
    
    def _create_web_search_node(self, state: CorrectiveRAGState):
        """Create a node that performs web search when needed."""
        results = self.search_tool.invoke({"query": state["query"]})
        return {"web_search_results": results}
    
    def _should_use_web_search(self, state: CorrectiveRAGState) -> bool:
        """Determine if web search should be used based on document relevance."""
        # Use web search if the relevance score is less than 25%
        if state["relevance_score"] < .25:
            return True
        else:
            return False

    def _create_rag_response_node(self, state: CorrectiveRAGState):
        """Create a node that generates a response."""
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that generates a response to a user's query.

            User's query: {query}
            Documents: {documents}

            Format the response with the answer and the sources:
            Answer: [Answer here within 2-3 sentences only]
            Sources:
            - <Filename of the source document, not the title>

            Example response 1:
            User: What is the average salary of a software engineer in the United States?
            Context: [Text from a document about the average salary of a software engineer in the United States]
            - [Name of the source document]
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

    def _create_web_search_response_node(self, state: CorrectiveRAGState):
        """Create a node that generates a response."""
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that generates a response to a user's query.

            User's query: {query}
            Web search results: {web_search_results}

            Provide the response in the following example format below.
            If the answer is factual, keep the response short to 1-2 sentences.
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": state["query"], "web_search_results": state["web_search_results"]})
        return {"response": response}
    
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
        result =self.graph.invoke({"query": message})
        print(result)
        return result["response"]
