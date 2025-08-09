"""Part 2 - Basic Tools implementation.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

import json
from enum import Enum
from typing import Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator


class QueryType(Enum):
    """Enum for classifying query types."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    CALCULATION = "calculation"


class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        
        # Prebuilt, reusable components
        self.output_parser: StrOutputParser | None = None
        self.query_classifier_chain = None
        self.response_chains: Dict[QueryType, object] = {}
    
    def initialize(self) -> None:
        """Initialize components for basic tools.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        """
        # Initialize the chat model
        self.llm = init_chat_model(
            model="gpt-4o-mini",
            temperature=0,
        )
        
        self._setup_query_classifier_prompts()
        self._setup_response_prompts()

        self._build_chains()

    def _setup_query_classifier_prompts(self) -> None:
        """Sets up the classifier prompts for each query type."""

        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """
                You are a helpful assistant that classifies questions into different types or decides if a tool should be used.
                You will be given a question and you will need to classify it into one of the following types below.
                
                1. Factual questions ask for specific, verifiable information with a clear, objective answer.
                Examples: When was the iPhone released? What's the capital of Japan? How many kilometers in a mile?
                
                2. Analytical questions require breaking down a concept, situation, or idea into parts.
                Examples: Why do startups fail? How does inflation affect consumer behavior? What causes supply chain disruptions?
                
                3. Comparison questions ask to assess similarities, differences, pros/cons, or trade-offs between two or more things.
                Examples: How does Java compare to Python? What's the difference between B2B and B2C sales? Which is better: hybrid or remote work?
                
                4. Definition questions seek to explain the meaning of a term, concept, or acronym, often to clarify understanding.
                Examples: What is an API? Define machine learning. What does ROI mean? Explain a neural network.

                5. Calculation questions ask for a mathematical calculation to be performed.
                Examples: What is 10 + 10? What is the square root of 100? What is 100 * 100? What is 5% of 100?
                
                Your response should only be one of the following: factual, analytical, comparison, definition, or calculation.
                """),
            ("user", "{message}"),
        ])

    def _setup_response_prompts(self) -> None:
        """Sets up the response prompts for each query type."""

        factual_response_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """
                Provide a clear, concise, and verifiable answer to the question. Do not elaborate unless asked.

                - Stick to objective facts; avoid interpretation.
                - Use numbers, dates, names, or data when relevant.
                - Avoid filler language or restating the question.
                - If multiple valid answers exist, list them succinctly.

                Response should be concise and direct.
                """),
            ("user", "{message}"),
        ])

        analytical_response_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """
                Analyze the question by breaking it into components and logically reasoning through the answer.

                - Break down complex concepts into manageable parts.
                - Examine causes, effects, patterns, or implications.
                - Include reasoning, logic, or synthesis—not just facts.
                - Consider multiple valid interpretations or conclusions.
                - Offer evidence or real-world examples to support points.
                - Keep a clear structure (intro → reasoning → conclusion).

                Response should be detailed and include reasoning steps.
                """),
            ("user", "{message}"),
        ])

        comparison_response_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """
                Compare the given items using a clear, structured format. Focus on key differences, similarities, and trade-offs.

                - Use tables or lists to compare items.
                - Highlight key differences and similarities.
                - Consider trade-offs or pros/cons.
                - Keep the comparison focused and concise.
                - End with a brief summary or decision guidance if applicable.
                - Avoid bias unless asked for an opinionated take.
                """),
            ("user", "{message}"),
        ])

        definition_response_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """
                Define the term clearly and provide supporting examples or common use cases.

                - Start with a concise definition in plain language.
                - Add relevant context, industry, or domain usage.
                - Include at least one example and use case.
                - Avoid circular definitions or excessive jargon.
                - Optionally explain why the term is important or commonly used.
                """),
            ("user", "{message}"),
        ])

        calculation_response_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """
                Return the mathematical expression of the question only, nothing else.
                
                - Do not include any other text or comments. 
                - No Latex responses.
                - No markdown responses.
                - If rounding is asked for, determine where the decimal should be rounded to based on the question.
                Examples:
                Question: Round 10.3 to the nearest integer.
                Answer: 10.3 // 1

                Question: What is the a 15% tip of a $70 dollar dinner, rounder to the nearest ten cents.
                Answer:  (((0.15*70)/0.10 + 0.5)//1)*0.10

                Question: Round 2.11 to one decimal place.
                Answer: (2.11*10 // 1)/10
                """),
            ("user", "{message}"),
        ])

        # Initialize the response formatting prompt
        self.response_prompts = {
            QueryType.FACTUAL: factual_response_prompt,
            QueryType.ANALYTICAL: analytical_response_prompt,
            QueryType.COMPARISON: comparison_response_prompt,
            QueryType.DEFINITION: definition_response_prompt,
            QueryType.CALCULATION: calculation_response_prompt
        }

    def _build_chains(self) -> None:
        """Build reusable chains once during initialization."""
        self.output_parser = StrOutputParser()
        self.query_classifier_chain = (
            self.query_classifier_prompt | self.llm | self.output_parser
        )
        self.response_chains = {
            query_type: (prompt | self.llm | self.output_parser)
            for query_type, prompt in self.response_prompts.items()
        }

    def _classify_query_type(self, message: str) -> QueryType:
        """Classifies the query type based on the message."""
        query_type_str = self.query_classifier_chain.invoke(message).lower()
        print(f"Query type: {query_type_str}")
        
        try:
            return QueryType(query_type_str)
        except ValueError:
            # If the classifier returns an invalid query type, default to FACTUAL
            # This is a reasonable fallback as factual queries are the most common
            print(f"Unknown query type '{query_type_str}', defaulting to FACTUAL")
            return QueryType.FACTUAL

    def _generate_response(self, message: str, query_type: QueryType) -> str:
        """Generates a response based on the query type."""
        response_chain = self.response_chains[query_type]
        response = response_chain.invoke(message)
        return response
    
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
        query_type = self._classify_query_type(message)

        response = self._generate_response(message, query_type)
        if query_type == QueryType.CALCULATION:
            response = str(Calculator.evaluate_expression(response))

        return response