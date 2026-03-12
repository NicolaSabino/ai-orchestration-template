"""
AI Agent Orchestration - All-in-One Template

Simplified single-file implementation with:
- Prompts as global multiline strings
- Langfuse observability (required)
- Custom tools
- Agent factory functions
- Example execution

Requirements:
- .env file with OLLAMA_MODEL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
- pip install -r requirements.txt

Usage:
    python ai_orchestration.py
"""

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

load_dotenv()


# ============================================================================
# SECTION 1: PROMPT TEMPLATES (GLOBAL VARIABLES)
# ============================================================================

FOO_AGENT_PROMPT = """You are a specialized foo agent.

Your role:
- Handle foo-related tasks using the foo_command tool
- Provide clear and helpful responses

Guidelines:
- Use the foo_command tool when needed
- Explain your actions clearly
"""


# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

# LLM Model Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


# ============================================================================
# SECTION 3: OBSERVABILITY (Langfuse Integration - REQUIRED)
# ============================================================================

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Validate Langfuse configuration
if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    raise ValueError(
        "Langfuse credentials are required. "
        "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env file."
    )

# Initialize Langfuse handler
langfuse_handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

print(f"[Observability] Langfuse tracing enabled: {LANGFUSE_HOST}")


def get_callbacks():
    """Get Langfuse callback handler for LangChain operations."""
    return [langfuse_handler]


# ============================================================================
# SECTION 4: TOOLS
# ============================================================================

@tool
def foo_command(input_text: str) -> str:
    """
    Execute foo command.

    Use this tool to process foo-related requests.

    Args:
        input_text: The input to process

    Returns:
        Processed result
    """
    return f"Foo command executed with input: {input_text}"


# ============================================================================
# SECTION 5: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with examples."""
    print("=" * 70)
    print("AI Agent Orchestration - Foo Agent Example")
    print(f"[Observability] Langfuse tracing: ENABLED ({LANGFUSE_HOST})")
    print("=" * 70)

    # Initialize LLM model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE
    )

    # Create foo agent with foo_command tool
    foo_agent = create_agent(
        model=model,
        system_prompt=FOO_AGENT_PROMPT,
        tools=[foo_command]
    )

    # Example usage
    print("\n[Example] Using Foo Agent with foo_command tool:")
    print("-" * 70)

    query = "Execute foo command with test input"
    result = foo_agent.invoke(
        {"messages": [HumanMessage(query)]},
        config={"callbacks": get_callbacks()}
    )
    response = result["messages"][-1].content

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
