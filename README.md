# AI Orchestration Template

Ultra-minimal template for AI agent systems. **Zero abstractions**, direct LangChain usage, external prompts. No classes to inherit, just functions. Prompts in `.txt` files. Flat structure (~250 LOC core logic). Working examples included.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings:
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2
# TEMPERATURE=0.7

# Run examples
python main.py
```

**To use different providers**: Edit `main.py` Step 2 (e.g., replace `ChatOllama` with `ChatOpenAI`)

## Project Structure

```text
exercise-1/
├── agents.py              # Agent factory functions (~110 LOC)
├── tools.py               # Tools with @tool decorator (~85 LOC)
├── orchestrator.py        # Orchestrator factory (~110 LOC)
├── main.py                # Entry point with TODO (~190 LOC)
├── prompts/               # External prompt files
│   ├── general_agent.txt  # Prompt for general agent
│   ├── math_agent.txt     # Prompt for math agent
│   └── orchestrator.txt   # Prompt for orchestrator
├── .env.example
└── README.md
```

## How It Works

**Tools** extend agent capabilities using `@tool` decorator:

```python
from langchain.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description - LLM uses this to decide when to use the tool."""
    return result
```

**Agents** are created via factory functions:

```python
def create_my_agent(model, tools=None):
    return create_agent(
        model=model,
        system_prompt=load_prompt("my_agent"),  # Loads from prompts/my_agent.txt
        tools=tools or []
    )
```

**Orchestrator** routes requests to appropriate agents:

```python
orchestrator = create_orchestrator(
    model=model,
    agents_dict={
        "general": general_agent,
        "math": math_agent
    }
)
```

Follow TODO comments in `main.py` for setup steps.

## Examples

### Example 1: Direct Agent Call

```python
from agents import create_general_agent, invoke_agent

agent = create_general_agent(model)
response = invoke_agent(agent, "What is Python?")
```

### Example 2: Agent with Tools

```python
from agents import create_math_agent, invoke_agent
from tools import calculator

math_agent = create_math_agent(model, tools=[calculator])
response = invoke_agent(math_agent, "Calculate 25 * 17")
```

### Example 3: Orchestrator

```python
from orchestrator import create_orchestrator, invoke_orchestrator

orchestrator = create_orchestrator(
    model=model,
    agents_dict={"general": general_agent, "math": math_agent}
)

# Auto-routes to the right agent
response = invoke_orchestrator(orchestrator, "Calculate 100 / 4 and explain Python")
```

## Customization

**Add a Tool** - Edit `tools.py`:

```python
@tool
def my_new_tool(param: str) -> str:
    """Description of what this tool does."""
    return result
```

Then add to agent: `agent = create_general_agent(model, tools=[calculator, my_new_tool])`

**Add an Agent**:

1. Create `prompts/my_agent.txt` with instructions
2. Add factory in `agents.py`:

```python
def create_my_agent(model, tools=None):
    return create_agent(
        model=model,
        system_prompt=load_prompt("my_agent"),
        tools=tools or []
    )
```

3. Use in `main.py`: `my_agent = create_my_agent(model, tools=[...])`

**Change Behavior** - Edit `.txt` files in `prompts/` (no code changes needed!)

**Add to Orchestrator** - Edit `main.py`:

```python
orchestrator = create_orchestrator(
    model=model,
    agents_dict={
        "general": general_agent,
        "math": math_agent,
        "my_new": my_new_agent,  # Add here
    }
)
```

Then update `prompts/orchestrator.txt` to describe when to use it.

## Requirements

- Python 3.8+
- LangChain
- LangChain Ollama (or other model provider)
- python-dotenv

See `requirements.txt` for full list.
