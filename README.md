# AI Orchestration Template

Ultra-minimal single-file template for AI agent systems. **Zero abstractions**, direct LangChain usage, inline prompts. Everything in one file (~156 LOC). Langfuse observability required. Working example included.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with required settings:
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2
# TEMPERATURE=0.7
# LANGFUSE_PUBLIC_KEY=pk-lf-...      # REQUIRED
# LANGFUSE_SECRET_KEY=sk-lf-...      # REQUIRED
# LANGFUSE_HOST=https://cloud.langfuse.com

# Run the example
python ai_orchestration.py
```

**To use different providers**: Edit the `main()` function in `ai_orchestration.py` (e.g., replace `ChatOllama` with `ChatOpenAI`)

## Project Structure

```text
ai-orchestration-template/
├── ai_orchestration.py    # Single all-in-one file (~156 LOC)
│                          # - Prompts as global variables
│                          # - Langfuse integration (required)
│                          # - Tool definitions
│                          # - Agent creation
│                          # - Example execution
├── .env                   # Environment configuration
├── .env.example           # Example configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How It Works

Everything is in `ai_orchestration.py` organized in sections:

**1. Prompts** - Defined as global multiline strings:

```python
FOO_AGENT_PROMPT = """You are a specialized foo agent.

Your role:
- Handle foo-related tasks using the foo_command tool
- Provide clear and helpful responses
"""
```

**2. Tools** - Using `@tool` decorator:

```python
@tool
def foo_command(input_text: str) -> str:
    """Execute foo command. Use this tool to process foo-related requests."""
    return f"Foo command executed with input: {input_text}"
```

**3. Agent Creation** - Explicit in main():

```python
foo_agent = create_agent(
    model=model,
    system_prompt=FOO_AGENT_PROMPT,
    tools=[foo_command]
)
```

**4. Langfuse Integration** - Always enabled, configured via environment variables.

## Example

The template includes a working example with a foo agent and foo_command tool:

```python
# Create the agent
foo_agent = create_agent(
    model=model,
    system_prompt=FOO_AGENT_PROMPT,
    tools=[foo_command]
)

# Invoke the agent
query = "Execute foo command with test input"
result = foo_agent.invoke(
    {"messages": [HumanMessage(query)]},
    config={"callbacks": get_callbacks()}
)
response = result["messages"][-1].content
```

Run with: `python ai_orchestration.py`

## Observability & Tracing

Langfuse integration is **required** for tracing agent executions, tool calls, and LLM interactions.

### Setup Langfuse

1. **Get Langfuse credentials** (choose one):
   - **Cloud**: Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
   - **Self-hosted**: Deploy Langfuse locally ([docs](https://langfuse.com/docs/deployment/self-host))

2. **Configure in .env** (required):

   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
   LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
   LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
   ```

3. **Run your agents** - traces appear automatically in Langfuse dashboard

### What Gets Traced

- Agent invocations
- Tool executions
- LLM calls with prompts, completions, and token usage
- Execution timeline and latency

Access the Langfuse dashboard at your `LANGFUSE_HOST` URL to view traces, token usage, and debugging info.

## Customization

All customization happens in `ai_orchestration.py`:

**Add a Tool** - Add in SECTION 4:

```python
@tool
def my_new_tool(param: str) -> str:
    """Description of what this tool does."""
    return result
```

**Add an Agent** - Add prompt in SECTION 1 and create in main():

```python
# SECTION 1: Add prompt
MY_AGENT_PROMPT = """You are a specialized agent..."""

# In main(): Create agent
my_agent = create_agent(
    model=model,
    system_prompt=MY_AGENT_PROMPT,
    tools=[my_new_tool]
)
```

**Change Behavior** - Edit the prompt strings directly in SECTION 1

## Requirements

- Python 3.8+
- LangChain
- LangChain Ollama (or other model provider)
- python-dotenv
- Langfuse (required for observability)

See `requirements.txt` for full list.

## Key Features

- **Single file**: Everything in one place (~156 LOC)
- **No abstractions**: Direct LangChain usage
- **Inline prompts**: Edit prompts as multiline strings
- **Required observability**: Langfuse tracing always enabled
- **Explicit code**: No helper functions, everything visible in main()
