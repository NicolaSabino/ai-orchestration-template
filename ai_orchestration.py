"""
AI Agent Orchestration - All-in-One Template

Simplified single-file implementation with:
- OpenRouter LLM integration (gpt-4o-mini default)
- Langfuse observability with session tracking (required)
- Custom tools
- Agent factory functions
- Connectivity test method
- Example execution

Requirements:
- .env file with OPENROUTER_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST, TEAM_NAME
- pip install -r requirements.txt

Usage:
    python ai_orchestration.py
"""

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
from enum import Enum

load_dotenv()


# ============================================================================
# SECTION 1: PYDANTIC SCHEMAS (STRUCTURED OUTPUTS)
# ============================================================================

# Enums for Risk Levels
class RiskLevel(str, Enum):
    """Risk level classification for fraud detection."""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


# Transaction Analysis Models
class TransactionFeatures(BaseModel):
    """Features extracted from a transaction."""
    transaction_id: str = Field(description="Unique transaction identifier")
    sender_iban: str = Field(description="Sender IBAN")
    receiver_iban: str = Field(description="Receiver IBAN")
    amount: float = Field(description="Transaction amount")
    timestamp: str = Field(description="Transaction timestamp")
    transaction_type: Optional[str] = Field(default=None, description="Type of transaction")
    location: Optional[str] = Field(default=None, description="Transaction location")


class TransactionAnomalyScore(BaseModel):
    """Anomaly scores for transaction analysis."""
    amount_anomaly: float = Field(ge=0.0, le=1.0, description="Amount anomaly score (0-1)")
    frequency_anomaly: float = Field(ge=0.0, le=1.0, description="Frequency anomaly score (0-1)")
    timing_anomaly: float = Field(ge=0.0, le=1.0, description="Timing anomaly score (0-1)")
    recipient_anomaly: float = Field(ge=0.0, le=1.0, description="Recipient anomaly score (0-1)")
    velocity_anomaly: float = Field(ge=0.0, le=1.0, description="Transaction velocity anomaly (0-1)")


class TransactionAnalysisResult(BaseModel):
    """Result from Transaction Analyzer agent."""
    transaction_id: str = Field(description="Transaction ID analyzed")
    risk_level: RiskLevel = Field(description="Overall risk level")
    anomaly_scores: TransactionAnomalyScore = Field(description="Detailed anomaly scores")
    risk_score: float = Field(ge=0.0, le=1.0, description="Combined risk score (0-1)")
    suspicious_indicators: List[str] = Field(description="List of suspicious indicators found")
    reasoning: str = Field(description="Explanation of the analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this analysis (0-1)")


# Behavioral Analysis Models
class UserBehaviorBaseline(BaseModel):
    """Baseline behavioral profile for a user."""
    user_id: str = Field(description="User identifier (IBAN or biotag)")
    avg_transaction_amount: float = Field(description="Average transaction amount")
    std_transaction_amount: float = Field(description="Standard deviation of amounts")
    typical_hours: List[int] = Field(description="Typical transaction hours (0-23)")
    typical_recipients: List[str] = Field(description="Frequent recipient IBANs")
    typical_locations: List[str] = Field(description="Frequent transaction locations")
    transaction_count: int = Field(description="Total transactions observed")
    last_updated: str = Field(description="Last update timestamp")


class BehavioralAnomalyResult(BaseModel):
    """Result from Behavioral Profiler agent."""
    transaction_id: str = Field(description="Transaction ID analyzed")
    risk_level: RiskLevel = Field(description="Behavioral risk level")
    risk_score: float = Field(ge=0.0, le=1.0, description="Behavioral risk score (0-1)")
    deviations: List[str] = Field(description="Detected behavioral deviations")
    phishing_indicators: List[str] = Field(description="Phishing pattern indicators")
    profile_match_score: float = Field(ge=0.0, le=1.0, description="Match with user baseline (0-1, lower=worse)")
    reasoning: str = Field(description="Explanation of behavioral analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this analysis (0-1)")


# Geospatial Analysis Models
class LocationPoint(BaseModel):
    """Geographic location point."""
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    timestamp: Optional[str] = Field(default=None, description="Location timestamp")
    accuracy: Optional[float] = Field(default=None, description="Location accuracy in meters")


class GeospatialAnalysisResult(BaseModel):
    """Result from Geospatial Analyzer agent."""
    transaction_id: str = Field(description="Transaction ID analyzed")
    risk_level: RiskLevel = Field(description="Geospatial risk level")
    risk_score: float = Field(ge=0.0, le=1.0, description="Geospatial risk score (0-1)")
    impossible_travel_detected: bool = Field(description="Whether impossible travel was detected")
    distance_from_last_location_km: Optional[float] = Field(default=None, description="Distance from last GPS location")
    time_since_last_location_hours: Optional[float] = Field(default=None, description="Time since last GPS ping")
    location_anomalies: List[str] = Field(description="Detected location anomalies")
    reasoning: str = Field(description="Explanation of geospatial analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this analysis (0-1)")


# Fraud Decision Models
class FraudEvidence(BaseModel):
    """Single piece of fraud evidence."""
    source: str = Field(description="Source agent (transaction/behavioral/geospatial)")
    evidence_type: str = Field(description="Type of evidence")
    description: str = Field(description="Evidence description")
    weight: float = Field(ge=0.0, le=1.0, description="Weight/importance of this evidence (0-1)")


class FraudDecision(BaseModel):
    """Final fraud detection decision from Orchestrator."""
    transaction_id: str = Field(description="Transaction ID")
    is_fraudulent: bool = Field(description="Final fraud decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in decision (0-1)")
    risk_score: float = Field(ge=0.0, le=1.0, description="Combined risk score (0-1)")
    primary_reasons: List[str] = Field(description="Primary reasons for the decision")
    evidence: List[FraudEvidence] = Field(description="All evidence considered")
    reasoning: str = Field(description="Detailed reasoning for the decision")
    pattern_matches: List[str] = Field(description="Matched fraud patterns from memory")


# Memory & Pattern Models
class FraudPattern(BaseModel):
    """Fraud pattern discovered and stored in memory."""
    pattern_id: str = Field(description="Unique pattern identifier")
    pattern_type: str = Field(description="Type of fraud pattern")
    description: str = Field(description="Human-readable pattern description")
    features: Dict[str, Any] = Field(description="Pattern feature vector")
    success_rate: float = Field(ge=0.0, le=1.0, description="Detection success rate (0-1)")
    occurrences: int = Field(description="Number of times observed")
    level_discovered: int = Field(description="Challenge level where discovered")
    discovered_at: str = Field(description="Discovery timestamp")
    last_seen: str = Field(description="Last occurrence timestamp")


class AdaptiveLearningState(BaseModel):
    """Global adaptive learning state."""
    current_level: int = Field(description="Current challenge level")
    total_transactions_analyzed: int = Field(description="Total transactions processed")
    total_frauds_detected: int = Field(description="Total frauds detected")
    fraud_patterns_count: int = Field(description="Number of patterns in memory")
    user_baselines_count: int = Field(description="Number of user baselines created")
    decision_threshold: float = Field(ge=0.0, le=1.0, description="Current decision threshold")
    last_adaptation_timestamp: str = Field(description="Last adaptation timestamp")
    performance_metrics: Dict[str, float] = Field(description="Performance tracking metrics")


# Agent Communication Models
class AgentTask(BaseModel):
    """Task assigned to a specialized agent."""
    task_id: str = Field(description="Unique task identifier")
    agent_type: Literal["transaction", "behavioral", "geospatial"] = Field(description="Target agent type")
    transaction_data: Dict[str, Any] = Field(description="Transaction data to analyze")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class AgentResponse(BaseModel):
    """Generic agent response wrapper."""
    task_id: str = Field(description="Task identifier")
    agent_type: str = Field(description="Responding agent type")
    success: bool = Field(description="Whether analysis succeeded")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Analysis result")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# SECTION 2: PROMPT TEMPLATES (GLOBAL VARIABLES)
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
# SECTION 3: CONFIGURATION
# ============================================================================

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Validate OpenRouter configuration
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OpenRouter API key is required. "
        "Set OPENROUTER_API_KEY in .env file."
    )




# ============================================================================
# SECTION 4: OBSERVABILITY (Langfuse Integration - REQUIRED)
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

# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

print(f"[Observability] Langfuse tracing enabled: {LANGFUSE_HOST}")


def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    team_name = os.getenv("TEAM_NAME", "default-team")
    return f"{team_name}-{ulid.new().str}"


def run_agent_with_tracking(agent, query, session_id):
    """
    Run agent invocation with Langfuse observability.

    Args:
        agent: The LangChain agent to invoke
        query: The user query string
        session_id: Session ID for grouping traces

    Returns:
        The agent response content
    """
    # Create Langfuse callback handler
    # Session ID will be set via LangChain run metadata
    langfuse_handler = CallbackHandler()

    # Invoke agent with Langfuse tracking
    # Pass session_id and user_id in the config metadata
    result = agent.invoke(
        {"messages": [HumanMessage(query)]},
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "session_id": session_id,
                "user_id": os.getenv("TEAM_NAME", "default-team")
            },
            "run_name": "agent_invocation"
        }
    )

    return result["messages"][-1].content


# ============================================================================
# SECTION 5: TOOLS
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
# SECTION 6: CONNECTIVITY TEST
# ============================================================================

def test_connectivity(session_id):
    """
    Test OpenRouter and Langfuse connectivity.

    Args:
        session_id: Session ID to use for this test

    Performs a simple agent call with a test question and validates:
    - OpenRouter API connection
    - Langfuse trace logging
    - Agent tool functionality
    """
    print("=" * 70)
    print("Connectivity Test: OpenRouter + Langfuse")
    print("=" * 70)

    # Initialize LLM with OpenRouter
    print(f"[OpenRouter] Connecting to: {OPENROUTER_BASE_URL}")
    print(f"[OpenRouter] Using model: {OPENROUTER_MODEL}")

    model = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        temperature=TEMPERATURE
    )

    # Create agent with foo_command tool
    print("[Agent] Creating agent with foo_command tool...")
    agent = create_agent(
        model=model,
        system_prompt=FOO_AGENT_PROMPT,
        tools=[foo_command]
    )

    # Test query
    test_query = "What is the square root of 144?"
    print(f"\n[Test] Sending query: {test_query}")

    try:
        # Run agent with Langfuse tracking
        response = run_agent_with_tracking(agent, test_query, session_id)

        print(f"[Response] {response}")

        # Flush Langfuse to ensure trace is sent
        langfuse_client.flush()

        print(f"\n[Success] ✓ OpenRouter connection successful")
        print(f"[Success] ✓ Langfuse trace sent successfully")

        return True

    except Exception as e:
        print(f"\n[Error] ✗ Connectivity test failed: {e}")
        return False

    finally:
        print("=" * 70)


# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with connectivity test and examples."""
    # Generate a single session ID for the entire execution
    session_id = generate_session_id()

    print("=" * 70)
    print("AI Agent Orchestration - OpenRouter + Langfuse")
    print(f"[Model] {OPENROUTER_MODEL}")
    print(f"[Observability] Langfuse tracing: ENABLED ({LANGFUSE_HOST})")
    print(f"[Session ID] {session_id}")
    print("=" * 70)

    # Run connectivity test with the same session
    if not test_connectivity(session_id):
        print("\n[Error] Connectivity test failed. Please check your configuration.")
        return

    print("\n[Info] Starting main agent example...")
    print("=" * 70)

    # Initialize LLM model with OpenRouter
    model = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        temperature=TEMPERATURE
    )

    # Create foo agent with foo_command tool
    foo_agent = create_agent(
        model=model,
        system_prompt=FOO_AGENT_PROMPT,
        tools=[foo_command]
    )

    # Example usage with tracking
    print("\n[Example] Using Foo Agent with foo_command tool:")
    print("-" * 70)

    query = "Execute foo command with test input"
    response = run_agent_with_tracking(foo_agent, query, session_id)

    print(f"Query: {query}")
    print(f"Response: {response}")

    # Flush to ensure all traces are sent
    langfuse_client.flush()

    print("=" * 70)
    print("Example completed!")
    print(f"View all traces for session: {session_id}")
    print(f"Langfuse URL: {LANGFUSE_HOST}")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
