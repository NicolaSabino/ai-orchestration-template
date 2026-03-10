"""
Tools for AI agents.

Add your custom tools here using the @tool decorator from LangChain.
Each tool should have a clear docstring - the LLM uses it to understand when to use the tool.
"""

from langchain.tools import tool


# ============================================================================
# TODO: Add your custom tools here
# ============================================================================
# Example template:
#
# @tool
# def my_custom_tool(param1: str, param2: int) -> str:
#     """
#     Brief description of what this tool does.
#
#     Args:
#         param1: Description of first parameter
#         param2: Description of second parameter
#
#     Returns:
#         Description of what the tool returns
#     """
#     # Your implementation here
#     return result


@tool
def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform basic arithmetic operations.

    Use this tool to calculate: addition, subtraction, multiplication, division.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of the arithmetic operation
    """
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else float('inf')
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}. Use: add, subtract, multiply, divide")

    return operations[operation](a, b)


@tool
def text_analyzer(text: str) -> dict:
    """
    Analyze text and return statistics.

    Use this tool to get word count, character count, and sentence count.

    Args:
        text: The text to analyze

    Returns:
        Dictionary with 'chars', 'words', and 'sentences' counts
    """
    chars = len(text)
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')

    return {
        "characters": chars,
        "words": words,
        "sentences": sentences
    }
