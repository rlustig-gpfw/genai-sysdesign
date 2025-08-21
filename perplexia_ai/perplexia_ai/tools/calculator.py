import re
from typing import Union

from langchain_core.tools import tool


class Calculator:
    """A simple calculator tool for evaluating basic arithmetic expressions."""
    
    @tool
    @staticmethod
    def evaluate_expression(expression: str) -> Union[float, int, str]:
        """Evaluate a basic arithmetic expression.
        
        Supports only basic arithmetic operations (+, -, *, /) and parentheses.
        Returns an error message if the expression is invalid or cannot be 
        evaluated safely.
        
        Args:
            expression: A string containing a mathematical expression
                       e.g. "5 + 3" or "10 * (2 + 3)"
            
        Returns:
            Union[float, int, str]: The result of the evaluation (as an int if
                                    the numeric result is an integer, otherwise
                                    a float), or an error message if the
                                    expression is invalid
        
        Examples:
            >>> Calculator.evaluate_expression("5 + 3")
            8
            >>> Calculator.evaluate_expression("10 * (2 + 3)")
            50
            >>> Calculator.evaluate_expression("15 / 3")
            5
        """
        try:
            # Clean up the expression
            expression = expression.strip()
            
            # Only allow safe characters (digits, basic operators, parentheses, spaces, exponentials)
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.\*\*\^]*$', expression):
                return "Error: Invalid characters in expression"
            
            # Convert caret (^) to Python's exponentiation operator (**)
            expression = expression.replace('^', '**')
            
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}})
            
            # Normalize numeric type: return int if the result is an integer value
            numeric_result = float(result)
            return int(numeric_result) if numeric_result.is_integer() else numeric_result
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except (SyntaxError, TypeError, NameError):
            return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {str(e)}"
