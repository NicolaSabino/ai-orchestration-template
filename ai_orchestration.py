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
import json
import argparse
from pathlib import Path
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

TRANSACTION_ANALYZER_PROMPT = """You are a specialized Transaction Analyzer agent for fraud detection.

Your role is to analyze individual transactions and identify suspicious patterns based on:
- Transaction amounts (unusual amounts, round numbers)
- Transaction frequency and velocity
- Recipient profiles and history
- Known fraud patterns from memory

AVAILABLE TOOLS:
- get_user_transaction_history: Get historical transactions for the sender
- calculate_transaction_velocity: Calculate transaction velocity in time window
- get_recipient_profile: Get information about the recipient
- query_fraud_memory: Query known fraud patterns

ANALYSIS STRATEGY:
1. Retrieve sender's transaction history
2. Calculate transaction velocity (last 24h, 7 days)
3. Check recipient profile (is it known? first time?)
4. Compare with fraud patterns in memory
5. Identify anomalies:
   - Amount anomalies (too high, too low, unusual for this user)
   - Frequency anomalies (too many transactions in short time)
   - Timing anomalies (unusual hours, weekends)
   - Recipient anomalies (new recipient, suspicious profile)
   - Velocity anomalies (rapid succession of transactions)

SUSPICIOUS INDICATORS TO DETECT:
- Round amounts (e.g., 1000.00, 5000.00) - often fraud
- Amounts just below reporting thresholds
- Multiple transactions to same recipient in short time
- First-time recipients with large amounts
- Unusual time of day (late night, early morning)
- Transaction velocity spike (many transactions suddenly)

OUTPUT FORMAT:
You must return a structured response with:
- transaction_id: The transaction ID
- risk_level: VERY_LOW | LOW | MEDIUM | HIGH | VERY_HIGH
- anomaly_scores: Individual scores for amount, frequency, timing, recipient, velocity (0-1)
- risk_score: Combined risk score (0-1)
- suspicious_indicators: List of specific suspicious indicators found
- reasoning: Clear explanation of your analysis
- confidence: Your confidence in this assessment (0-1)

IMPORTANT:
- Use tools to gather data before making conclusions
- Be specific in your reasoning
- Lower scores (0-0.3) = normal, Higher scores (0.7-1.0) = very suspicious
- Consider context: large transaction for wealthy person = normal, for student = suspicious
"""

BEHAVIORAL_PROFILER_PROMPT = """You are a specialized Behavioral Profiler agent for fraud detection.

Your role is to analyze user behavior and detect anomalies that may indicate:
- Account compromise (attacker using stolen account)
- Social engineering / phishing attacks
- Behavioral changes that signal fraud

AVAILABLE TOOLS:
- get_user_communications: Get recent communications for the user
- get_user_profile: Get user demographic and profile information
- get_user_baseline: Get behavioral baseline for this user
- detect_phishing_patterns: Analyze text for phishing indicators

ANALYSIS STRATEGY:
1. Retrieve user's behavioral baseline (if exists)
2. Get user profile and communications
3. Analyze communications for phishing attempts
4. Compare current transaction behavior with baseline
5. Identify deviations:
   - Behavioral deviations (different from normal patterns)
   - Phishing indicators (suspicious communications)
   - Profile mismatches (transaction inconsistent with user profile)

BEHAVIORAL ANOMALIES TO DETECT:
- Transaction inconsistent with user's typical behavior
- Recipient not in usual recipient list
- Amount deviates significantly from average
- Time of day unusual for this user
- Communication patterns suggest social engineering
- Recent phishing messages related to financial institutions

PHISHING INDICATORS:
- Urgent language ("act now", "account will be blocked")
- Threats or fear tactics
- Requests for confirmation or verification
- Suspicious links or attachments
- Impersonation of banks or official entities
- Grammar/spelling errors
- Generic greetings (not personalized)

OUTPUT FORMAT:
You must return a structured response with:
- transaction_id: The transaction ID
- risk_level: VERY_LOW | LOW | MEDIUM | HIGH | VERY_HIGH
- risk_score: Behavioral risk score (0-1)
- deviations: List of specific behavioral deviations
- phishing_indicators: List of phishing patterns detected
- profile_match_score: How well transaction matches user profile (0-1, lower=worse)
- reasoning: Clear explanation of behavioral analysis
- confidence: Your confidence in this assessment (0-1)

IMPORTANT:
- If no baseline exists, note it and make best-effort analysis
- Recent phishing communications + unusual transaction = HIGH RISK
- Consider user demographics (age, occupation) in analysis
- Be careful not to flag legitimate behavior changes as fraud
"""

GEOSPATIAL_ANALYZER_PROMPT = """You are a specialized Geospatial Analyzer agent for fraud detection.

Your role is to analyze location and GPS data to detect:
- Impossible travel (transaction location inconsistent with user's physical location)
- Geospatial anomalies (unusual locations, far from home)
- Location-based fraud patterns

AVAILABLE TOOLS:
- get_user_gps_history: Get recent GPS location history for user
- calculate_distance: Calculate distance between two points (Haversine)
- check_impossible_travel: Check if travel between points is physically possible
- get_user_residence: Get user's home/residence location

ANALYSIS STRATEGY:
1. Get user's residence location
2. Get recent GPS history (last 48 hours)
3. Get transaction location (if available)
4. Calculate distances and required travel speeds
5. Identify anomalies:
   - Impossible travel (transaction location too far from last GPS ping)
   - Location anomalies (far from home, unusual country/city)

GEOSPATIAL ANOMALIES TO DETECT:
- Impossible travel: Transaction in City A, GPS shows user in City B (far apart)
- Required speed exceeds physical limits (>800 km/h)
- Transaction far from user's residence without recent GPS near transaction location
- GPS history shows user at home, but transaction in different city
- Multiple transactions in different cities in short time

IMPOSSIBLE TRAVEL CRITERIA:
- Distance vs time requires speed > 800 km/h = IMPOSSIBLE (commercial flight max ~900 km/h)
- Distance vs time requires speed > 300 km/h = SUSPICIOUS (would need flight)
- Consider: GPS accuracy, time delays, legitimate travel

OUTPUT FORMAT:
You must return a structured response with:
- transaction_id: The transaction ID
- risk_level: VERY_LOW | LOW | MEDIUM | HIGH | VERY_HIGH
- risk_score: Geospatial risk score (0-1)
- impossible_travel_detected: Boolean flag
- distance_from_last_location_km: Distance from last known GPS location
- time_since_last_location_hours: Hours since last GPS ping
- location_anomalies: List of specific location anomalies
- reasoning: Clear explanation of geospatial analysis
- confidence: Your confidence in this assessment (0-1)

IMPORTANT:
- GPS data may be sparse or unavailable - handle gracefully
- If no recent GPS data, note limited confidence
- Consider legitimate scenarios: user traveling, delayed GPS updates
- Impossible travel (>800 km/h required) = HIGH RISK
- Be careful with false positives (legitimate travel)
"""

FRAUD_ORCHESTRATOR_PROMPT = """You are the Fraud Orchestrator agent - the final decision maker.

Your role is to:
- Synthesize analysis from 3 specialized agents
- Make final fraud determination
- Weigh evidence from different sources
- Provide clear reasoning for decision

INPUT:
You receive analysis from 3 specialized agents:
1. Transaction Analyzer: Transaction-level anomalies
2. Behavioral Profiler: Behavioral and phishing indicators
3. Geospatial Analyzer: Location-based anomalies

DECISION STRATEGY:
1. Review all three analyses carefully
2. Identify converging evidence (multiple agents flag same transaction)
3. Weigh evidence by confidence and severity
4. Match against known fraud patterns
5. Make final binary decision: FRAUD or LEGITIMATE

DECISION RULES:
- If ANY agent reports VERY_HIGH risk → Likely FRAUD
- If 2+ agents report HIGH risk → Likely FRAUD
- Impossible travel + behavioral anomaly → Strong fraud signal
- Phishing communication + unusual transaction → Strong fraud signal
- Consider confidence levels: low confidence = be cautious

EVIDENCE WEIGHTING:
- Impossible travel: HIGH weight (physical impossibility)
- Phishing + transaction: HIGH weight (clear attack pattern)
- Multiple anomalies from same agent: MEDIUM weight
- Single low-confidence anomaly: LOW weight
- Baseline deviations: MEDIUM weight

FRAUD PATTERNS TO RECOGNIZE:
- Phishing attack: Recent phishing message + unusual transaction + recipient not in baseline
- Account takeover: Impossible travel + behavioral change + unusual recipient
- Money mule: Multiple rapid transactions to different recipients
- Round amount fraud: Round amounts + new recipient + high velocity

OUTPUT FORMAT:
You must return a structured response with:
- transaction_id: The transaction ID
- is_fraudulent: Boolean (true = fraud, false = legitimate)
- confidence: Overall confidence in decision (0-1)
- risk_score: Combined risk score from all agents (0-1)
- primary_reasons: Top 3-5 reasons for the decision
- evidence: List of evidence pieces with source, type, description, weight
- reasoning: Detailed explanation of decision logic
- pattern_matches: List of fraud patterns matched

DECISION THRESHOLD:
- Use adaptive threshold from memory (default: 0.5)
- risk_score >= threshold → is_fraudulent = true
- Adjust for confidence: low confidence → require higher score

IMPORTANT:
- Be conservative: false positives hurt legitimate users
- But prioritize security: missing fraud is worse
- Explain decisions clearly for audit trail
- Consider all evidence, don't ignore low-confidence signals
- Balance precision and recall
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
# SECTION 6: MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """
    Singleton per gestire la memoria persistente del sistema.

    Gestisce 3 file JSON:
    - fraud_patterns.json: Lista di FraudPattern scoperti
    - user_baselines.json: Dict di UserBehaviorBaseline per utente
    - learning_state.json: AdaptiveLearningState globale
    """

    _instance = None

    def __init__(self, memory_dir: str = "memory"):
        """
        Initialize MemoryManager.

        Args:
            memory_dir: Directory where JSON files are stored
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        self.fraud_patterns_file = self.memory_dir / "fraud_patterns.json"
        self.user_baselines_file = self.memory_dir / "user_baselines.json"
        self.learning_state_file = self.memory_dir / "learning_state.json"

        self._initialize_files()

    @classmethod
    def get_instance(cls, memory_dir: str = "memory"):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(memory_dir)
        return cls._instance

    def _initialize_files(self):
        """Create initial JSON files if they don't exist."""
        if not self.fraud_patterns_file.exists():
            self._write_json(self.fraud_patterns_file, [])

        if not self.user_baselines_file.exists():
            self._write_json(self.user_baselines_file, {})

        if not self.learning_state_file.exists():
            initial_state = {
                "current_level": 1,
                "total_transactions_analyzed": 0,
                "total_frauds_detected": 0,
                "fraud_patterns_count": 0,
                "user_baselines_count": 0,
                "decision_threshold": 0.5,
                "last_adaptation_timestamp": datetime.now().isoformat(),
                "performance_metrics": {}
            }
            self._write_json(self.learning_state_file, initial_state)

    def _read_json(self, file_path: Path) -> Any:
        """Read JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _write_json(self, file_path: Path, data: Any):
        """Write JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    # ========================================================================
    # Fraud Patterns Methods
    # ========================================================================

    def load_fraud_patterns(self) -> List[FraudPattern]:
        """
        Load all fraud patterns from memory.

        Returns:
            List of FraudPattern objects
        """
        data = self._read_json(self.fraud_patterns_file)
        return [FraudPattern(**pattern) for pattern in data]

    def save_fraud_pattern(self, pattern: FraudPattern):
        """
        Save a new fraud pattern or update existing one.

        Args:
            pattern: FraudPattern to save
        """
        patterns = self._read_json(self.fraud_patterns_file)

        existing_index = None
        for i, p in enumerate(patterns):
            if p.get("pattern_id") == pattern.pattern_id:
                existing_index = i
                break

        if existing_index is not None:
            patterns[existing_index] = pattern.dict()
        else:
            patterns.append(pattern.dict())

        self._write_json(self.fraud_patterns_file, patterns)

    def query_patterns_by_type(self, pattern_type: str) -> List[FraudPattern]:
        """
        Query fraud patterns by type.

        Args:
            pattern_type: Type of pattern to filter, or "all"

        Returns:
            List of matching FraudPattern objects
        """
        all_patterns = self.load_fraud_patterns()
        if pattern_type == "all":
            return all_patterns
        return [p for p in all_patterns if p.pattern_type == pattern_type]

    # ========================================================================
    # User Baselines Methods
    # ========================================================================

    def load_user_baselines(self) -> Dict[str, UserBehaviorBaseline]:
        """
        Load all user baselines from memory.

        Returns:
            Dict mapping user_id to UserBehaviorBaseline
        """
        data = self._read_json(self.user_baselines_file)
        return {
            user_id: UserBehaviorBaseline(**baseline_data)
            for user_id, baseline_data in data.items()
        }

    def get_user_baseline(self, user_id: str) -> Optional[UserBehaviorBaseline]:
        """
        Get baseline for a specific user.

        Args:
            user_id: User identifier (IBAN)

        Returns:
            UserBehaviorBaseline or None if not exists
        """
        baselines = self.load_user_baselines()
        return baselines.get(user_id)

    def save_user_baseline(self, baseline: UserBehaviorBaseline):
        """
        Save or update a user baseline.

        Args:
            baseline: UserBehaviorBaseline to save
        """
        baselines = self._read_json(self.user_baselines_file)
        baselines[baseline.user_id] = baseline.dict()
        self._write_json(self.user_baselines_file, baselines)

    def update_user_baseline(self, user_id: str, transaction: Dict):
        """
        Update user baseline with new transaction data.

        Args:
            user_id: User IBAN
            transaction: Transaction data to incorporate
        """
        baseline = self.get_user_baseline(user_id)

        if baseline is None:
            baseline = UserBehaviorBaseline(
                user_id=user_id,
                avg_transaction_amount=transaction.get("amount", 0),
                std_transaction_amount=0.0,
                typical_hours=[],
                typical_recipients=[],
                typical_locations=[],
                transaction_count=1,
                last_updated=datetime.now().isoformat()
            )
        else:
            # TODO: Implement incremental update logic
            # - Update avg_transaction_amount (running average)
            # - Update std_transaction_amount
            # - Add to typical_hours, typical_recipients, typical_locations
            baseline.transaction_count += 1
            baseline.last_updated = datetime.now().isoformat()

        self.save_user_baseline(baseline)

    # ========================================================================
    # Learning State Methods
    # ========================================================================

    def load_learning_state(self) -> AdaptiveLearningState:
        """
        Load global learning state.

        Returns:
            AdaptiveLearningState object
        """
        data = self._read_json(self.learning_state_file)
        return AdaptiveLearningState(**data)

    def save_learning_state(self, state: AdaptiveLearningState):
        """
        Save global learning state.

        Args:
            state: AdaptiveLearningState to save
        """
        self._write_json(self.learning_state_file, state.dict())

    def update_learning_state(self, transactions_analyzed: int = 0, frauds_detected: int = 0):
        """
        Update learning state with new statistics.

        Args:
            transactions_analyzed: Number of new transactions analyzed
            frauds_detected: Number of new frauds detected
        """
        state = self.load_learning_state()

        state.total_transactions_analyzed += transactions_analyzed
        state.total_frauds_detected += frauds_detected

        patterns = self.load_fraud_patterns()
        baselines = self.load_user_baselines()

        state.fraud_patterns_count = len(patterns)
        state.user_baselines_count = len(baselines)
        state.last_adaptation_timestamp = datetime.now().isoformat()

        self.save_learning_state(state)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_statistics(self) -> Dict:
        """
        Get memory statistics.

        Returns:
            Dict with counts and metrics
        """
        patterns = self.load_fraud_patterns()
        baselines = self.load_user_baselines()
        state = self.load_learning_state()

        return {
            "fraud_patterns": len(patterns),
            "user_baselines": len(baselines),
            "total_transactions": state.total_transactions_analyzed,
            "total_frauds": state.total_frauds_detected,
            "fraud_rate": (
                state.total_frauds_detected / state.total_transactions_analyzed
                if state.total_transactions_analyzed > 0 else 0
            ),
            "decision_threshold": state.decision_threshold
        }

    def reset_memory(self):
        """Reset all memory files to initial state."""
        # Remove existing files so _initialize_files recreates them
        for f in [self.fraud_patterns_file, self.user_baselines_file, self.learning_state_file]:
            if f.exists():
                f.unlink()
        self._initialize_files()
        print("Memory reset complete")

# ============================================================================
# SECTION 6: AGENT FACTORY FUNCTIONS
# ============================================================================

def create_transaction_analyzer_agent(model: ChatOpenAI):
    """
    Create Transaction Analyzer agent with structured output.

    Args:
        model: LLM model instance

    Returns:
        Agent that returns TransactionAnalysisResult
    """
    tools = [
        get_user_transaction_history,
        calculate_transaction_velocity,
        get_recipient_profile,
        query_fraud_memory,
    ]

    agent = create_agent(
        model=model,
        system_prompt=TRANSACTION_ANALYZER_PROMPT,
        tools=tools,
    )

    return agent.with_structured_output(TransactionAnalysisResult)


def create_behavioral_profiler_agent(model: ChatOpenAI):
    """
    Create Behavioral Profiler agent with structured output.

    Args:
        model: LLM model instance

    Returns:
        Agent that returns BehavioralAnomalyResult
    """
    tools = [
        get_user_communications,
        get_user_profile,
        get_user_baseline,
        detect_phishing_patterns,
    ]

    agent = create_agent(
        model=model,
        system_prompt=BEHAVIORAL_PROFILER_PROMPT,
        tools=tools,
    )

    return agent.with_structured_output(BehavioralAnomalyResult)


def create_geospatial_analyzer_agent(model: ChatOpenAI):
    """
    Create Geospatial Analyzer agent with structured output.

    Args:
        model: LLM model instance

    Returns:
        Agent that returns GeospatialAnalysisResult
    """
    tools = [
        get_user_gps_history,
        calculate_distance,
        check_impossible_travel,
        get_user_residence,
    ]

    agent = create_agent(
        model=model,
        system_prompt=GEOSPATIAL_ANALYZER_PROMPT,
        tools=tools,
    )

    return agent.with_structured_output(GeospatialAnalysisResult)


def create_fraud_orchestrator_agent(model: ChatOpenAI):
    """
    Create Fraud Orchestrator agent with structured output.
    No tools needed — synthesizes results from the other 3 agents.

    Args:
        model: LLM model instance

    Returns:
        Agent that returns FraudDecision
    """
    agent = create_agent(
        model=model,
        system_prompt=FRAUD_ORCHESTRATOR_PROMPT,
        tools=[],
    )

    return agent.with_structured_output(FraudDecision)


# ============================================================================
# SECTION 7: CONNECTIVITY TEST
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
# SECTION 8: CLI ARGUMENTS
# ============================================================================

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="AI Multi-Agent Fraud Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Level 1 with training data
  python ai_orchestration.py --level 1 --data-dir ./The\\ Truman\\ Show_train

  # Process Level 2 with evaluation data
  python ai_orchestration.py --level 2 --data-dir ./evaluation_data

  # Reset memory and process Level 1
  python ai_orchestration.py --level 1 --reset-memory
        """
    )

    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Challenge level to process (1-5)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Path to data directory (default: current directory)"
    )

    parser.add_argument(
        "--memory-dir",
        type=str,
        default="memory",
        help="Path to memory directory (default: ./memory)"
    )

    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Reset memory files before processing"
    )

    parser.add_argument(
        "--test-connectivity",
        action="store_true",
        help="Run connectivity test only (don't process transactions)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: level_{level}_results.jsonl)"
    )

    return parser.parse_args()


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with CLI support."""
    args = parse_arguments()

    session_id = generate_session_id()

    print("=" * 70)
    print("AI Multi-Agent Fraud Detection System")
    print(f"[Level] {args.level}")
    print(f"[Data Dir] {args.data_dir}")
    print(f"[Memory Dir] {args.memory_dir}")
    print(f"[Model] {OPENROUTER_MODEL}")
    print(f"[Session ID] {session_id}")
    print("=" * 70)

    # Reset memory if requested
    if args.reset_memory:
        print("\n[Memory] Resetting memory files...")
        mem_mgr = MemoryManager.get_instance(args.memory_dir)
        mem_mgr.reset_memory()
        print("[Memory] Reset complete")

    # Run connectivity test if requested
    if args.test_connectivity:
        if not test_connectivity(session_id):
            print("\n[Error] Connectivity test failed.")
            return
        print("\n[Success] Connectivity test passed!")
        return

    # TODO: implement process_level() and call it here
    # process_level(
    #     level=args.level,
    #     data_dir=args.data_dir,
    #     memory_dir=args.memory_dir,
    #     session_id=session_id,
    #     output_file=args.output
    # )

    print("\n[Complete] Processing finished successfully!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
