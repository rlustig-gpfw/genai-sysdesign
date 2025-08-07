"""Part 2 - Basic Tools implementation.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator

class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.calculator = Calculator()
    
    def initialize(self) -> None:
        """Initialize components for basic tools.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        """
        # TODO: Students implement initialization
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with calculator support.
        
        Students should:
        - Check if calculation needed
        - Use calculator if needed
        - Otherwise, handle as regular query
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 2
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement calculator integration
        return "hello from part 2"