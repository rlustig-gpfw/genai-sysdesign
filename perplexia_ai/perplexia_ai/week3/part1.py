"""Part 1 - Tool-Using Agent implementation.

This implementation focuses on:
- Converting tools from Assignment 1 to use with LangGraph
- Using the ReAct pattern for autonomous tool selection
- Comparing manual workflow vs agent approaches
"""

from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, START, END

from perplexia_ai.core.chat_interface import ChatInterface


class ToolUsingAgentChat(ChatInterface):
    """Week 3 Part 1 implementation focusing on tool-using agents."""
    
    def __init__(self):
        self.llm = None
        self.agent_executor = None
        self.tools = []
    
    def initialize(self) -> None:
        """Initialize components for the tool-using agent.
        
        Students should:
        - Initialize the chat model
        - Define tools for calculator, DateTime, and weather
        - Create the ReAct agent using LangGraph
        """
        # TODO: Initialize your chat model
        
        # TODO: Create tools using the tool decorator
        self.tools = self._create_tools()
        
        # TODO: Create the ReAct agent
        
        # TODO: Create and compile the graph
    
    def _create_tools(self) -> List[Any]:
        """Create and return the list of tools for the agent.
        
        Students should implement:
        - Calculator tool from Assignment 1
        - [Optional] DateTime tool from Assignment 1
        - Weather tool using Tavily search
        
        Returns:
            List: List of tool objects
        """
        # TODO: Implement calculator tool
        
        # TODO: Implement DateTime tool
        
        # TODO: Implement Weather tool using Tavily
    
    def _create_agent(self) -> Any:
        """Create and return the ReAct agent executor.
        
        Returns:
            Any: The agent executor graph or callable
        """
        # TODO: Create a ReAct agent with access to tools
        
        # TODO: Set up a StateGraph with the agent
        
        # TODO: Define entry point and compile
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the tool-using agent.
        
        Students should:
        - Send the message to the agent
        - Extract and return the agent's response
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages
            
        Returns:
            str: The assistant's response
        """
        # TODO: Prepare input for the agent
        
        # TODO: Run the agent and return the result
        return "Hello world from part 1"
