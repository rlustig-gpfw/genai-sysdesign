"""Part 2 - Agentic RAG implementation.

This implementation focuses on:
- Building an Agentic RAG system with dynamic search strategy
- Using LangGraph for controlling the RAG workflow
- Evaluating retrieved information quality
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import create_react_agent
from opik.integrations.langchain import OpikTracer
from pydantic import BaseModel, Field

from perplexia_ai.core.chat_interface import ChatInterface


class DocumentEvaluationStructure(BaseModel):
    """Structure for the document evaluation."""
    score: bool = Field(description="Whether the document is relevant to the user query")
    feedback: str = Field(description="The feedback for the system to help improve the query or clarify why the document is not relevant")
    improved_query: str = Field(description="A version of the original query improved or clarified based on additional reasoning or semantic intent")


class SynthesizerStructure(BaseModel):
    """Structure for the synthesizer."""
    response: str = Field(description="The response to the user's query")
    sources: List[str] = Field(description="The sources of the response (either OPM documents or web search results)")


class AgenticRAGState(MessagesState):
    """State for the Agentic RAG system."""
    query: str = Field(description="The user's query")
    improved_query: str = Field(description="A version of the original query improved or clarified based on additional reasoning or semantic intent")
    relevant_docs: List[Document] = Field(description="The retrieved documents")
    search_results: List[Dict[str, str]] = Field(description="The search results from the web")
    score: bool = Field(description="Whether the retrieved documents are relevant to the query")
    feedback: str = Field(description="The feedback for the system to help improve the query or clarify why the document is not relevant")
    num_iterations: int = Field(description="The number of iterations of the query")
    final_response: str = Field(description="The final response to the user's query")



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
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Set paths to OPM documents
        data_dir = Path(os.getcwd()) / "perplexia_ai" / "docs"
        self.document_paths = list(data_dir.glob("*.pdf"))

        # Process documents and create vector store
        docs = self._load_and_process_documents()
        self.vector_store, self.retriever = self._setup_vector_store(docs)
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.document_evaluator = self._create_document_evaluator()
        self.synthesizer = self._create_synthesizer()
        self.workflow = self._create_workflow()

        self.tracer = OpikTracer(graph=self.agent.get_graph(xray=True))
    
    def _load_and_process_documents(self) -> list[Document]:
        """Load and process OPM documents."""
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

        @tool
        def search_tool(query: str) -> List[Dict[str, str]]:
            """Search the web for information."""
            web_search_tool = TavilySearchResults(
                max_results=3,
                include_answer=True,
                include_raw_content=False,
                include_images=False,
                search_depth="advanced",
            )
            results = web_search_tool.invoke(query)
            return results

        return [retriever_tool, search_tool]

    def _create_document_evaluator(self) -> Any:
        """Create a document evaluator that assesses retrieved document quality.
        
        Returns:
            Any: The document evaluator runnable
        """
        evaluation_prompt = PromptTemplate.from_template(
            """
            You are an expert in assessing the relevance of documents retrieved from the Office of Personnel Management (OPM) to a user's query.
            Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

            ## Inputs
            - **User Query**: {query}
            - **Retrieved Document**: {relevant_docs}
            - **Web Search Results (if any)**: {search_results}

            ## Instructions
            - Evaluate whether the retrieved document includes keywords or conveys semantic content directly related to the user's query.
            - Assign a binary score:
            - Return true if the document is relevant to the user's query.
            - Return false if it is not relevant.
            - If the score is 'no', provide detailed, actionable feedback:
            - Clearly interpret the user's query.
            - Analyze its semantic intent.
            - Suggest improvements or clarifications to make the query more specific if needed. IMPORTANT: Direct this feedback to the system (not the user) so the query can be programmatically improved or clarified.
            - The "feedback" will be used specifically to improve the quality of future document retrieval.
            - In all cases, additionally output an improved query in the JSON output, using any additional reasoning or semantic intent identified during the evaluation to clarify or enhance the original query.
            
            After scoring, validate the reasoning in 1-2 sentences and confirm the result is fully justified; self-correct if validation fails.
            
            ## Output Format
            The response must contain the following fields:
            - `score` (boolean): true if relevant, false otherwise.
            - `feedback` (string, optional): Include only if the score is 'no'. Provide specific feedback for the system to help improve the query or clarify why the document is not relevant.
            - `improved_query` (string): A version of the original query improved or clarified based on additional reasoning or semantic intent.
            """
        )
        document_evaluator = evaluation_prompt | self.llm.with_structured_output(DocumentEvaluationStructure)
        return document_evaluator
    
    def _create_document_evaluator2(self) -> Any:
        """Create a document evaluator that assesses retrieved document quality.
        
        Returns:
            Any: The document evaluator runnable
        """
        evaluation_prompt = PromptTemplate.from_template(
            """
            You are an expert at evaluating the relevance of retrieved documents from the Office of Personnel Management (OPM).

            Here is the user query:
            {query}
            Here is the retrieved document:
            {relevant_docs}
            Here are the search results from the web (if any):
            {search_results}

            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            Return a binary score 'yes' or 'no' to indicate if the document is relevant to the user query.
            """
        )

        document_evaluator = evaluation_prompt | self.llm.with_structured_output(DocumentEvaluationStructure)
        return document_evaluator

    def _create_document_evaluator_node(self, state: AgenticRAGState) -> Any:
        """Create a document evaluator that assesses retrieved document quality."""
        response = self.document_evaluator.invoke({
            "query": state["query"],
            "relevant_docs": state["relevant_docs"], 
            "search_results": state["search_results"],
        })

        return {"score": response.score, "feedback": response.feedback, "improved_query": response.improved_query}

    def _rewrite_query(self, state: AgenticRAGState) -> Any:
        """Rewrite the user query to be more specific and clear."""
        rewrite_prompt = PromptTemplate.from_template(
            """
            Role and Objective
            - Interpret and refine the user's query by analyzing its underlying semantic intent, then rewrite it to be clearer and more specific.
            
            Instructions
            - Carefully review the original user query.
            - Think through the intended meaning and context.
            - Begin with a concise checklist (3-7 bullets) of your approach to interpreting and rewriting the query; keep checklist items conceptual.
            - Rewrite the query to be unambiguous, concise, and specific.
            - After rewriting, briefly validate that the reformulated query preserves the intent and improves clarity.
            
            Input
            - The original user query (provided as `{query}`).
            
            Output
            - A reformulated query that improves clarity and specificity while preserving the user's intent.
            """
        )
        chain = rewrite_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": state["query"]})

        # Update the original query with the rewritten query
        return {"query": response}
    
    def _create_synthesizer(self) -> Any:
        """Create a synthesizer that combines retrieved information.
        
        Returns:
            Any: The synthesizer runnable
        """
        synthesizer_prompt = PromptTemplate.from_template(
            """
            Role and Objective: Provide concise, direct answers to user queries by synthesizing information from Office of Personnel Management (OPM) documents and relevant search results.

            Instructions:
            - Begin with a concise checklist (3-5 bullets) of your approach for each user query; keep items conceptual, not implementation-level.
            - Summarize only the most relevant information to address the user's query.
            - Limit your response to a maximum of three sentences when possible.
            - Reference information from OPM documents and web search results as needed, specifying sources when possible.
            - After generating your answer, briefly validate whether all key aspects of the user query have a direct response based on the available information. If not, state what is missing or unclear.
            - Only return the summary of the information, not the checklist or validation steps.

            Context:
            - User query: {query}
            - Retrieved OPM document: {relevant_docs}
            - Web search results: {search_results}
            """
        )
        synthesizer = synthesizer_prompt | self.llm.with_structured_output(SynthesizerStructure)
        return synthesizer

    def _create_synthesizer_node(self, state: AgenticRAGState) -> Any:
        """Create a synthesizer that combines retrieved information."""
        response = self.synthesizer.invoke({
            "query": state["query"],
            "relevant_docs": state["relevant_docs"], 
            "search_results": state["search_results"],
        })
        return {"final_response": response.response, "sources": response.sources}

    def _create_agent(self) -> Any:
        """Create and return the ReAct agent executor.
        
        Returns:
            Any: The agent executor graph or callable
        """
        agent = create_react_agent(
            model=self.llm,
            tools=self.tools
        )
        return agent

    def _create_agent_node(self, state: AgenticRAGState) -> Any:
        """Create the agent node.

        Places the response from the agent into the correct field in AgenticRAGState
        based on which tool is called.
        """
        # Use the improved query if it exists, otherwise use the original query
        if state["improved_query"] != "":
            query = state["improved_query"]
        else:
            query = state["query"]

        input = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""You are a helpful assistant that can answer questions about the Office of Personnel Management (OPM).
                    Prefer the use of tools to answer questions, if applicable.
                    Use both the OPM documents and the web search results to answer the question, if needed

                    Here is the user query:
                    {query}
                    """
                }
            ]
        }

        result = self.agent.invoke(input, config={"callbacks": [self.tracer]})
        messages = result.get("messages", result)

        update = {}

        # Find the last human message index
        user_idx = None
        if isinstance(messages, list):
            for idx in reversed(range(len(messages))):
                msg = messages[idx]
                is_human = (getattr(msg, "type", None) == "human") or (isinstance(msg, dict) and msg.get("type") == "human")
                if is_human:
                    user_idx = idx
                    break

        # Collect the latest tool outputs AFTER the last human message
        last_tool_outputs = {}
        if isinstance(messages, list):
            start_idx = (user_idx + 1) if user_idx is not None else 0
            for idx in range(start_idx, len(messages)):
                msg = messages[idx]
                is_tool = isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool" or (isinstance(msg, dict) and msg.get("type") == "tool")
                if is_tool:
                    tool_name = getattr(msg, "name", None) if not isinstance(msg, dict) else msg.get("name")
                    output = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
                    if tool_name:
                        # Always keep the latest output per tool
                        last_tool_outputs[tool_name] = output

        if "OPM_Retriever" in last_tool_outputs:
            update["relevant_docs"] = last_tool_outputs["OPM_Retriever"]
        if "search_tool" in last_tool_outputs:
            update["search_results"] = last_tool_outputs["search_tool"]

        # If no tool output was captured, fall back to returning the final AI response
        if not update and isinstance(messages, list) and messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", None) if not isinstance(last_msg, dict) else last_msg.get("content")
            if content:
                update["final_response"] = content

        update["num_iterations"] = state["num_iterations"] + 1
        return update

    def _should_synthesize(self, state: AgenticRAGState) -> str:
        """Determine if the agent needs to synthesize."""
        if state["score"] is True or state["num_iterations"] > 3:
            return "synthesizer"
        else:
            return "agent"
    
    def _create_workflow(self) -> Any:
        """Create the agentic RAG workflow using LangGraph.
        
        Returns:
            Any: The compiled workflow
        """
        workflow = StateGraph(AgenticRAGState)

        workflow.add_node("agent", self._create_agent_node)
        workflow.add_node("document_evaluator", self._create_document_evaluator_node)
        #workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("synthesizer", self._create_synthesizer_node)

        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", "document_evaluator")
        workflow.add_conditional_edges(
            "document_evaluator",
            self._should_synthesize,
        )
        workflow.add_edge("synthesizer", END)

        return workflow.compile()
    
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
        result = self.workflow.invoke({
            "query": message,
            "improved_query": "",
            "relevant_docs": [],
            "search_results": [],
            "score": False,
            "feedback": "",
            "num_iterations": 0,
            "final_response": "",
        })

        print(result)
        return result["final_response"]
