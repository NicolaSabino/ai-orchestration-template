"""
AI Agent Template - Main Entry Point

Follow the TODO comments to customize this template for your use case.
Edit prompts in the prompts/ directory to change agent behavior.
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# ============================================================================
# STEP 1: Import your modules
# ============================================================================
# TODO: These are your building blocks - keep these imports
from agents import create_general_agent, create_math_agent, invoke_agent
from tools import calculator, text_analyzer
from orchestrator import create_orchestrator, invoke_orchestrator

load_dotenv()


# ============================================================================
# STEP 2: Configure your LLM model
# ============================================================================
# TODO: Change the model provider here if needed
# Examples: ChatOpenAI, ChatAnthropic, etc.

model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=float(os.getenv("TEMPERATURE", "0.7"))
)


# ============================================================================
# STEP 3: Create your agents
# ============================================================================
# TODO: Create agents by calling the factory functions from agents.py
# Edit prompts/agent_name.txt files to customize behavior!

# General agent - no tools (can answer general questions)
general_agent = create_general_agent(
    model=model,
    tools=[]  # TODO: Add tools if needed
)

# Math agent - with calculator tool (specialized for math)
math_agent = create_math_agent(
    model=model,
    tools=[calculator]  # TODO: Add more math-related tools
)

# TODO: Add more agents here as needed
# Example:
# research_agent = create_research_agent(
#     model=model,
#     tools=[web_search, database_query]
# )


# ============================================================================
# STEP 4: Create orchestrator (optional but recommended)
# ============================================================================
# TODO: Add/remove agents from the orchestrator
# The orchestrator will route requests to the appropriate agent

orchestrator = create_orchestrator(
    model=model,
    agents_dict={
        "general": general_agent,  # For general questions
        "math": math_agent,        # For math problems
        # TODO: Add your custom agents here
        # "research": research_agent,
        # "coding": coding_agent,
    }
)


# ============================================================================
# STEP 5: Define your main logic
# ============================================================================
# TODO: Customize the examples below for your use case

def main():
    """Main execution function."""
    print("=" * 70)
    print("AI Agent Orchestration Template")
    print("=" * 70)

    # ========================================================================
    # EXAMPLE 1: Direct Agent Call (General Agent)
    # ========================================================================
    print("\n[Example 1] Using General Agent directly:")
    print("-" * 70)

    # TODO: Replace with your own query
    query = "Explain what Python is in one sentence"

    response = invoke_agent(general_agent, query)
    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # ========================================================================
    # EXAMPLE 2: Direct Agent Call (Math Agent with Tool)
    # ========================================================================
    print("\n[Example 2] Using Math Agent with calculator:")
    print("-" * 70)

    # TODO: Replace with your own calculation
    query = "What is 456 multiplied by 789?"

    response = invoke_agent(math_agent, query)
    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # ========================================================================
    # EXAMPLE 3: Orchestrator (Auto-routing)
    # ========================================================================
    print("\n[Example 3] Using Orchestrator (auto-routes to correct agent):")
    print("-" * 70)

    # TODO: Try different types of queries - orchestrator will route them
    queries = [
        "Calculate 25 + 17",
        "What is artificial intelligence?",
        "Compute the result of 100 divided by 4 and explain Python"
    ]

    for query in queries:
        response = invoke_orchestrator(orchestrator, query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


# ============================================================================
# OPTIONAL: Interactive Mode
# ============================================================================
# TODO: Uncomment to enable interactive chat

def interactive_mode():
    """
    Interactive chat mode - talk directly with the orchestrator.

    Type 'exit' to quit.
    """
    print("=" * 70)
    print("Interactive Mode - Type 'exit' to quit")
    print("=" * 70)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            # TODO: Choose which agent to use (orchestrator recommended)
            response = invoke_orchestrator(orchestrator, user_input)
            # Or use a specific agent:
            # response = invoke_agent(general_agent, user_input)

            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\nError: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # TODO: Choose execution mode

    # Option 1: Run predefined examples
    main()

    # Option 2: Run interactive mode
    # interactive_mode()

    # Option 3: Run both
    # main()
    # print("\n")
    # interactive_mode()
