"""
Orchestrator - coordinates multiple specialized agents.

The orchestrator acts as a router that decides which agent(s) to use based on the user's request.
It converts agents into tools and uses a meta-agent to make routing decisions.
"""

from pathlib import Path
from langchain.agents import create_agent
from langchain.tools import tool as langchain_tool
from langchain_core.messages import HumanMessage


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_prompt(name: str) -> str:
    """
    Load prompt from prompts/ directory.

    Args:
        name: Name of the prompt file (without .txt extension)

    Returns:
        Prompt text content
    """
    prompt_file = Path(__file__).parent / "prompts" / f"{name}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}\n"
            f"Create a file at prompts/{name}.txt"
        )

    return prompt_file.read_text().strip()


def invoke_orchestrator(orchestrator, message: str) -> str:
    """
    Helper function to invoke the orchestrator with a message.

    Args:
        orchestrator: Orchestrator agent instance
        message: User message to process

    Returns:
        Orchestrator's response as string
    """
    from observability import get_callbacks

    result = orchestrator.invoke(
        {"messages": [HumanMessage(message)]},
        config={"callbacks": get_callbacks()}
    )
    return result["messages"][-1].content


# ============================================================================
# TODO: Modify this function to add/remove agents
# ============================================================================

def create_orchestrator(model, agents_dict):
    """
    Create an orchestrator that routes requests to specialized agents.

    The orchestrator analyzes the user's request and decides which agent(s)
    to invoke. It can call multiple agents if needed and synthesize their responses.

    Args:
        model: LLM model instance
        agents_dict: Dictionary mapping agent names to agent instances
                     Example: {"general": general_agent, "math": math_agent}

    Returns:
        Configured orchestrator agent

    TODO: Customize routing behavior by editing prompts/orchestrator.txt

    Example usage:
        orchestrator = create_orchestrator(
            model=model,
            agents_dict={
                "general": general_agent,
                "math": math_agent,
                "research": research_agent
            }
        )
    """
    # Convert agents to tools that the orchestrator can use
    tools = []

    for agent_name, agent_instance in agents_dict.items():
        # Create a closure to capture the agent instance
        def make_agent_tool(name: str, agent):
            """Factory function to create a tool from an agent."""

            @langchain_tool
            def agent_tool(request: str) -> str:
                f"""Route request to {name} agent for specialized processing."""
                from observability import get_callbacks

                result = agent.invoke(
                    {"messages": [HumanMessage(request)]},
                    config={"callbacks": get_callbacks()}
                )
                return result["messages"][-1].content

            # Set the tool name dynamically
            agent_tool.__name__ = f"{name}_agent"
            agent_tool.name = f"{name}_agent"

            return agent_tool

        tools.append(make_agent_tool(agent_name, agent_instance))

    # Load orchestrator prompt and replace placeholder with agent names
    agent_names = ", ".join(agents_dict.keys())
    prompt = load_prompt("orchestrator").replace("{agent_names}", agent_names)

    # Create the orchestrator agent
    return create_agent(
        model=model,
        system_prompt=prompt,
        tools=tools
    )
