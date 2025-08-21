"""Part 1 - Tool-Using Agent implementation.

This implementation focuses on:
- Converting tools from Assignment 1 to use with LangGraph
- Using the ReAct pattern for autonomous tool selection
- Comparing manual workflow vs agent approaches
"""

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from opik.integrations.langchain import OpikTracer

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from perplexia_ai.tools.datetime import DateTimeTool
from perplexia_ai.tools.weather import WeatherTool


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
        self.llm = ChatOpenAI(model="gpt-4o")
        
        self.tools = self._create_tools()
        
        self._create_agent()

        self.tracer = OpikTracer(graph=self.agent_executor.get_graph(xray=True))

    def _create_tools(self) -> List[Any]:
        """Create and return the list of tools for the agent.
        
        Students should implement:
        - Calculator tool from Assignment 1
        - [Optional] DateTime tool from Assignment 1
        - Weather tool using Tavily search
        
        Returns:
            List: List of tool objects
        """       
        # Single natural-language DateTime tool + calculator + weather
        tools = [
            Calculator.evaluate_expression,
            DateTimeTool.evaluate,
            WeatherTool.get_weather,
        ]
        return tools
        
    def _create_agent(self) -> Any:
        """Create and return the ReAct agent executor.
        
        Returns:
            Any: The agent executor graph or callable
        """
        style_instructions = (
            """You are a helpful assistant. Prefer the use of tools to answer questions, if applicable.
            Style requirements:
            - Do not use LaTeX or TeX markup in responses.
            - Use plain, readable mathematical symbols and text, only when needed.
            - For example, if we need to show a calculation we can use the symbols +, -, *, /, ^, =, <, >.
            """
        )

        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=style_instructions,
        )

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
        result = self.agent_executor.invoke({"messages": [message]}, config={"callbacks": [self.tracer]})

        # Extract final response from the result
        last_message = result["messages"][-1]
        if hasattr(last_message, "content"):
            response = last_message.content
        else:
            response = last_message

        return response
