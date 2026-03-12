"""
Agent factory functions.

Create your agents here. Each agent is created using LangChain's create_agent() function.
Prompts are loaded from the prompts/ directory as .txt files.
"""

from pathlib import Path
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_prompt(agent_name: str) -> str:
    """
    Load prompt from prompts/ directory.

    Args:
        agent_name: Name of the prompt file (without .txt extension)

    Returns:
        Prompt text content
    """
    prompt_file = Path(__file__).parent / "prompts" / f"{agent_name}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}\n"
            f"Create a file at prompts/{agent_name}.txt"
        )

    return prompt_file.read_text().strip()


def invoke_agent(agent, message: str) -> str:
    """
    Helper function to invoke an agent with a message.

    Args:
        agent: LangChain agent instance
        message: User message to process

    Returns:
        Agent's response as string
    """
    from observability import get_callbacks

    result = agent.invoke(
        {"messages": [HumanMessage(message)]},
        config={"callbacks": get_callbacks()}
    )
    return result["messages"][-1].content


# ============================================================================
# TODO: Add your agent factory functions here
# ============================================================================
# Template for creating a new agent:
#
# def create_my_agent(model, tools=None):
#     """
#     Create my custom agent.
#
#     TODO: Edit prompts/my_agent.txt to customize behavior
#     """
#     return create_agent(
#         model=model,
#         system_prompt=load_prompt("my_agent"),
#         tools=tools or []
#     )


def create_general_agent(model, tools=None):
    """
    Create a general-purpose assistant agent.

    This agent can handle a variety of tasks and questions.

    Args:
        model: LLM model instance
        tools: Optional list of tools for the agent

    Returns:
        Configured LangChain agent

    TODO: Customize behavior by editing prompts/general_agent.txt
    """
    return create_agent(
        model=model,
        system_prompt=load_prompt("general_agent"),
        tools=tools or []
    )


def create_math_agent(model, tools=None):
    """
    Create a mathematical specialist agent.

    This agent is optimized for mathematical calculations and explanations.
    Best used with calculator tools.

    Args:
        model: LLM model instance
        tools: List of tools (should include calculator)

    Returns:
        Configured LangChain agent

    TODO: Customize behavior by editing prompts/math_agent.txt
    """
    return create_agent(
        model=model,
        system_prompt=load_prompt("math_agent"),
        tools=tools or []
    )
