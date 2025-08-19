"""Part 3 - Deep Research Multi-Agent System implementation.

This implementation focuses on:
- Building a multi-agent system for comprehensive research
- Using LangGraph for coordinating multiple specialized agents
- Synthesizing research findings into structured reports
"""

from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from perplexia_ai.core.chat_interface import ChatInterface


# NOTE: The TODOs here are ONLY a guideline, feel free to change the structure as you see fit.
class DeepResearchChat(ChatInterface):
    """Week 3 Part 3 implementation focusing on deep research."""
    
    def __init__(self):
        self.llm = None
        self.research_manager = None
        self.specialized_research_agent = None
        self.finalizer = None
        self.workflow = None
        self.tavily_search_tool = None
    
    def initialize(self) -> None:
        """Initialize components for the deep research system.
        
        Students should:
        - Initialize LLM model
        - Create the research manager agent
        - Create specialized research agents
        - Create the finalizer component
        - Set up the multi-agent workflow
        """
        # TODO: Initialize LLM model
        
        # TODO: Create Tavily search tool for agents
        
        # TODO: Create components
        self.research_manager = self._create_research_manager()
        self.specialized_research_agent = self._create_specialized_research_agent()
        self.finalizer = self._create_finalizer()
        
        # TODO: Create the workflow graph using these agents.
        self.workflow = self._create_workflow()
    
    def _create_research_manager(self) -> Any:
        """Create the research manager agent.
        
        This agent is responsible for:
        - Taking a broad research topic
        - Breaking it down into specific research questions
        - Creating a research plan and report structure

        Refer to the assignment description for the exact report structure.
        
        Returns:
            Any: The research manager component
        """
        # TODO: Create a research manager prompt
        # TODO: Define tools for the research manager
        # TODO: Create the research manager agent
    
    def _create_specialized_research_agent(self) -> Any:
        """Create specialized research agents.
        
        These agents are responsible for:
        - Conducting research on specific sections
        - Using web search to find information
        - Summarizing findings
        
        Returns:
            Any: The specialized research agent component
        """
        # TODO: Create a specialized research agent prompt
        # TODO: Define tools for the specialized agent
        # TODO: Create the specialized research agent
    
    def _create_finalizer(self) -> Any:
        """Create the finalizer component.
        
        This component is responsible for:
        - Taking the completed research sections
        - Generating the Executive Summary
        - Identifying Key Findings
        - Adding Limitations and Further Research
        
        Returns:
            Any: The finalizer component
        """
        # TODO: Create a finalizer prompt
        # TODO: Create the finalizer chain
    
    def _create_workflow(self) -> Any:
        """Create the multi-agent deep research workflow.
        
        Returns:
            Any: The compiled workflow
        """
        # TODO: Create a state graph with the research components
        # TODO: Define the state tracking needed for research
        # TODO: Add conditional edges for controlling research flow
        # TODO: Compile and return the workflow
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the deep research system.
        
        Args:
            message: The user's input message (research topic)
            chat_history: List of previous chat messages (not used for deep research)
            
        Returns:
            str: The research report
        """
        # TODO: Create initial state with research topic
        # TODO: Run the workflow and format the result 
        return "Hello world from part 3"
