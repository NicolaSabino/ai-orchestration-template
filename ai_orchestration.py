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


# ============================================================================
# SECTION 4.5: DATA MANAGER (Data Access Layer)
# ============================================================================

class DataManager:
    """
    Singleton class to manage all CSV/JSON data loading and caching.

    Handles:
    - Transactions (CSV)
    - Users (JSON)
    - GPS locations (JSON)
    - Communications (mails + sms JSON)
    """
    _instance = None

    def __init__(self, data_dir: str):
        """
        Initialize DataManager.

        Args:
            data_dir: Root directory containing data files
        """
        self.data_dir = data_dir
        self._transactions = []  # List[Dict]
        self._users = {}  # {iban: Dict}
        self._users_by_name = {}  # {name: Dict} for biotag lookup
        self._locations = []  # List[Dict] with biotag GPS data
        self._locations_by_biotag = {}  # {biotag: List[Dict]}
        self._mails = []  # List[Dict]
        self._sms = []  # List[Dict]
        self._communications_by_recipient = {}  # {email/phone: List[Dict]}

    @classmethod
    def get_instance(cls, data_dir: str = "."):
        """
        Get singleton instance of DataManager.

        Args:
            data_dir: Root directory containing data files

        Returns:
            DataManager instance
        """
        if cls._instance is None:
            cls._instance = cls(data_dir)
        return cls._instance

    def load_all_data(self):
        """Load all data files (transactions, users, locations, communications)."""
        print("[DataManager] Loading all data...")
        self.load_transactions()
        self.load_users()
        self.load_locations()
        self.load_communications()
        print("[DataManager] All data loaded successfully")

    def load_transactions(self) -> List[Dict]:
        """
        Load transactions from CSV file.

        Returns:
            List of transaction dictionaries
        """
        import csv

        csv_path = os.path.join(self.data_dir, "public", "transactions.csv")

        if not os.path.exists(csv_path):
            print(f"[DataManager] Warning: Transactions file not found: {csv_path}")
            return []

        self._transactions = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert amount to float
                if 'amount' in row:
                    try:
                        row['amount'] = float(row['amount'])
                    except (ValueError, TypeError):
                        row['amount'] = 0.0

                # Convert balance_after to float
                if 'balance_after' in row:
                    try:
                        row['balance_after'] = float(row['balance_after'])
                    except (ValueError, TypeError):
                        row['balance_after'] = 0.0

                self._transactions.append(row)

        print(f"[DataManager] Loaded {len(self._transactions)} transactions")
        return self._transactions

    def load_users(self) -> Dict[str, Dict]:
        """
        Load users from JSON file and index by IBAN.

        Returns:
            Dictionary of users indexed by IBAN
        """
        import json

        json_path = os.path.join(self.data_dir, "public", "users.json")

        if not os.path.exists(json_path):
            print(f"[DataManager] Warning: Users file not found: {json_path}")
            return {}

        with open(json_path, 'r', encoding='utf-8') as f:
            users_list = json.load(f)

        # Index by IBAN
        self._users = {}
        self._users_by_name = {}

        for user in users_list:
            if 'iban' in user:
                self._users[user['iban']] = user

                # Also index by full name for biotag lookup
                if 'first_name' in user and 'last_name' in user:
                    full_name = f"{user['first_name']} {user['last_name']}"
                    self._users_by_name[full_name] = user

        print(f"[DataManager] Loaded {len(self._users)} users")
        return self._users

    def load_locations(self) -> Dict[str, List[Dict]]:
        """
        Load GPS locations from JSON file and group by biotag.

        Returns:
            Dictionary of GPS points grouped by biotag
        """
        import json

        json_path = os.path.join(self.data_dir, "public", "locations.json")

        if not os.path.exists(json_path):
            print(f"[DataManager] Warning: Locations file not found: {json_path}")
            return {}

        with open(json_path, 'r', encoding='utf-8') as f:
            self._locations = json.load(f)

        # Group by biotag and sort by timestamp
        self._locations_by_biotag = {}
        for loc in self._locations:
            biotag = loc.get('biotag')
            if biotag:
                if biotag not in self._locations_by_biotag:
                    self._locations_by_biotag[biotag] = []
                self._locations_by_biotag[biotag].append(loc)

        # Sort each biotag's locations by timestamp
        for biotag in self._locations_by_biotag:
            self._locations_by_biotag[biotag].sort(
                key=lambda x: x.get('timestamp', '')
            )

        print(f"[DataManager] Loaded {len(self._locations)} GPS points for {len(self._locations_by_biotag)} biotags")
        return self._locations_by_biotag

    def load_communications(self) -> Dict[str, List[Dict]]:
        """
        Load communications (mails + sms) from JSON files and group by recipient.

        Returns:
            Dictionary of communications grouped by recipient identifier
        """
        import json
        import re

        # Load mails
        mails_path = os.path.join(self.data_dir, "public", "mails.json")
        if os.path.exists(mails_path):
            with open(mails_path, 'r', encoding='utf-8') as f:
                self._mails = json.load(f)
            print(f"[DataManager] Loaded {len(self._mails)} mails")
        else:
            print(f"[DataManager] Warning: Mails file not found: {mails_path}")
            self._mails = []

        # Load SMS
        sms_path = os.path.join(self.data_dir, "public", "sms.json")
        if os.path.exists(sms_path):
            with open(sms_path, 'r', encoding='utf-8') as f:
                self._sms = json.load(f)
            print(f"[DataManager] Loaded {len(self._sms)} SMS")
        else:
            print(f"[DataManager] Warning: SMS file not found: {sms_path}")
            self._sms = []

        # Group by recipient
        self._communications_by_recipient = {}

        # Process mails - extract recipient email from "To:" header
        for mail in self._mails:
            mail_text = mail.get('mail', '')
            # Extract email from To: line
            to_match = re.search(r'To:\s*"([^"]+)"\s*<([^>]+)>', mail_text)
            if to_match:
                recipient_email = to_match.group(2).strip()
                if recipient_email not in self._communications_by_recipient:
                    self._communications_by_recipient[recipient_email] = []
                self._communications_by_recipient[recipient_email].append({
                    'type': 'mail',
                    'content': mail_text,
                    'data': mail
                })

        # Process SMS - extract recipient phone from "To:" line
        for sms in self._sms:
            sms_text = sms.get('sms', '')
            # Extract phone from To: line
            to_match = re.search(r'To:\s*(\+?\d+)', sms_text)
            if to_match:
                recipient_phone = to_match.group(1).strip()
                if recipient_phone not in self._communications_by_recipient:
                    self._communications_by_recipient[recipient_phone] = []
                self._communications_by_recipient[recipient_phone].append({
                    'type': 'sms',
                    'content': sms_text,
                    'data': sms
                })

        total_comms = sum(len(v) for v in self._communications_by_recipient.values())
        print(f"[DataManager] Indexed {total_comms} communications for {len(self._communications_by_recipient)} recipients")

        return self._communications_by_recipient

    # Getter methods

    def get_transactions(self) -> List[Dict]:
        """Get all loaded transactions."""
        return self._transactions

    def get_user(self, iban: str) -> Optional[Dict]:
        """
        Get user by IBAN.

        Args:
            iban: User IBAN

        Returns:
            User dict or None if not found
        """
        return self._users.get(iban)

    def get_user_by_name(self, name: str) -> Optional[Dict]:
        """
        Get user by full name.

        Args:
            name: Full name (first + last)

        Returns:
            User dict or None if not found
        """
        return self._users_by_name.get(name)

    def get_all_users(self) -> Dict[str, Dict]:
        """Get all users indexed by IBAN."""
        return self._users

    def get_user_gps(self, biotag: str) -> List[Dict]:
        """
        Get GPS history for a user by biotag.

        Args:
            biotag: User biotag

        Returns:
            List of GPS points sorted by timestamp
        """
        return self._locations_by_biotag.get(biotag, [])

    def get_user_communications(self, recipient_id: str) -> List[Dict]:
        """
        Get communications for a recipient (email or phone).

        Args:
            recipient_id: Email address or phone number

        Returns:
            List of communication dictionaries
        """
        return self._communications_by_recipient.get(recipient_id, [])

    def get_all_locations(self) -> List[Dict]:
        """Get all GPS locations."""
        return self._locations

    def get_all_mails(self) -> List[Dict]:
        """Get all emails."""
        return self._mails

    def get_all_sms(self) -> List[Dict]:
        """Get all SMS."""
        return self._sms


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

# Transaction Analyzer Tools

@tool
def get_user_transaction_history(user_iban: str, limit: int = 50) -> str:
    """
    Get transaction history for a user.

    Args:
        user_iban: User IBAN to query
        limit: Max number of transactions to return (default: 50)

    Returns:
        JSON string with transaction history
    """
    import json
    from datetime import datetime

    data_mgr = DataManager.get_instance()
    all_transactions = data_mgr.get_transactions()

    # Filter transactions where user is the sender
    user_transactions = [
        tx for tx in all_transactions
        if tx.get('sender_iban') == user_iban
    ]

    # Sort by timestamp (most recent first)
    user_transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    # Limit results
    user_transactions = user_transactions[:limit]

    result = {
        "user_iban": user_iban,
        "transaction_count": len(user_transactions),
        "transactions": user_transactions
    }

    return json.dumps(result, indent=2)


@tool
def calculate_transaction_velocity(user_iban: str, time_window_hours: int = 24) -> str:
    """
    Calculate transaction velocity for a user.

    Args:
        user_iban: User IBAN
        time_window_hours: Time window in hours (default: 24)

    Returns:
        JSON with velocity metrics: {count, total_amount, avg_amount, max_amount}
    """
    import json
    from datetime import datetime, timedelta

    data_mgr = DataManager.get_instance()
    all_transactions = data_mgr.get_transactions()

    # Filter transactions by user
    user_transactions = [
        tx for tx in all_transactions
        if tx.get('sender_iban') == user_iban
    ]

    if not user_transactions:
        return json.dumps({
            "user_iban": user_iban,
            "time_window_hours": time_window_hours,
            "count": 0,
            "total_amount": 0.0,
            "avg_amount": 0.0,
            "max_amount": 0.0,
            "message": "No transactions found for this user"
        })

    # Find most recent transaction timestamp
    timestamps = [
        datetime.fromisoformat(tx.get('timestamp', '').replace('Z', '+00:00'))
        for tx in user_transactions
        if tx.get('timestamp')
    ]

    if not timestamps:
        return json.dumps({
            "user_iban": user_iban,
            "time_window_hours": time_window_hours,
            "count": 0,
            "total_amount": 0.0,
            "avg_amount": 0.0,
            "max_amount": 0.0,
            "message": "No valid timestamps found"
        })

    most_recent = max(timestamps)
    cutoff_time = most_recent - timedelta(hours=time_window_hours)

    # Filter transactions within time window
    recent_transactions = []
    for tx in user_transactions:
        try:
            tx_time = datetime.fromisoformat(tx.get('timestamp', '').replace('Z', '+00:00'))
            if tx_time >= cutoff_time:
                recent_transactions.append(tx)
        except (ValueError, AttributeError):
            continue

    # Calculate metrics
    amounts = [tx.get('amount', 0.0) for tx in recent_transactions]
    total_amount = sum(amounts)
    count = len(recent_transactions)
    avg_amount = total_amount / count if count > 0 else 0.0
    max_amount = max(amounts) if amounts else 0.0

    result = {
        "user_iban": user_iban,
        "time_window_hours": time_window_hours,
        "count": count,
        "total_amount": round(total_amount, 2),
        "avg_amount": round(avg_amount, 2),
        "max_amount": round(max_amount, 2),
        "most_recent_timestamp": most_recent.isoformat()
    }

    return json.dumps(result, indent=2)


@tool
def get_recipient_profile(recipient_iban: str) -> str:
    """
    Get profile of recipient IBAN.

    Args:
        recipient_iban: Recipient IBAN

    Returns:
        JSON with recipient info and statistics
    """
    import json

    data_mgr = DataManager.get_instance()

    # Get user info if recipient is a registered user
    user = data_mgr.get_user(recipient_iban)

    # Count transactions to this recipient
    all_transactions = data_mgr.get_transactions()
    incoming_transactions = [
        tx for tx in all_transactions
        if tx.get('recipient_iban') == recipient_iban
    ]

    # Get unique senders
    unique_senders = set(tx.get('sender_iban') for tx in incoming_transactions if tx.get('sender_iban'))

    # Calculate statistics
    amounts = [tx.get('amount', 0.0) for tx in incoming_transactions]
    total_received = sum(amounts)
    avg_received = total_received / len(amounts) if amounts else 0.0
    max_received = max(amounts) if amounts else 0.0

    result = {
        "recipient_iban": recipient_iban,
        "is_registered_user": user is not None,
        "user_info": {
            "name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() if user else None,
            "job": user.get('job') if user else None,
            "birth_year": user.get('birth_year') if user else None,
            "residence": user.get('residence') if user else None
        } if user else None,
        "transaction_statistics": {
            "total_received_count": len(incoming_transactions),
            "unique_senders_count": len(unique_senders),
            "total_amount_received": round(total_received, 2),
            "avg_amount_received": round(avg_received, 2),
            "max_amount_received": round(max_received, 2)
        }
    }

    return json.dumps(result, indent=2)


@tool
def query_fraud_memory(pattern_type: str = "all") -> str:
    """
    Query fraud patterns from memory.

    Args:
        pattern_type: Type of pattern to query ("all", "behavioral", "geospatial", "transaction")

    Returns:
        JSON string with matching fraud patterns
    """
    import json

    # NOTE: This will be integrated with Alfonso's MemoryManager later
    # For now, return a placeholder structure that demonstrates the interface

    result = {
        "pattern_type": pattern_type,
        "patterns": [],
        "message": "MemoryManager integration pending - will be connected to Alfonso's Task 8"
    }

    # Placeholder: In production, this would call:
    # from memory_manager import MemoryManager
    # mem_mgr = MemoryManager.get_instance()
    # patterns = mem_mgr.load_fraud_patterns()
    # Filter by pattern_type if not "all"

    return json.dumps(result, indent=2)


# Behavioral Profiler Tools

@tool
def get_user_communications(user_biotag: str, limit: int = 20) -> str:
    """
    Get recent communications for a user.

    Args:
        user_biotag: User biotag
        limit: Max communications to return (default: 20)

    Returns:
        JSON string with communications
    """
    import json

    data_mgr = DataManager.get_instance()

    # Get user by biotag (biotags are in format: XXXX-XXXX-XXX-CITY-N)
    # We need to find the user first to get their email/phone
    all_users = data_mgr.get_all_users()

    # Search for user with matching biotag pattern or name
    user = None
    user_email = None
    user_phone = None

    # For now, return empty if we can't directly match
    # In production, we'd need a biotag-to-user mapping
    communications = []

    result = {
        "user_biotag": user_biotag,
        "communication_count": len(communications),
        "communications": communications[:limit],
        "message": "Note: Biotag-to-communication mapping requires user identification"
    }

    return json.dumps(result, indent=2)


@tool
def get_user_profile(user_iban: str) -> str:
    """
    Get user profile information.

    Args:
        user_iban: User IBAN

    Returns:
        JSON with user profile (name, age, location, etc.)
    """
    import json
    from datetime import datetime

    data_mgr = DataManager.get_instance()
    user = data_mgr.get_user(user_iban)

    if not user:
        return json.dumps({
            "user_iban": user_iban,
            "found": False,
            "message": "User not found"
        })

    # Calculate age if birth_year is available
    age = None
    if user.get('birth_year'):
        current_year = datetime.now().year
        age = current_year - user.get('birth_year')

    result = {
        "user_iban": user_iban,
        "found": True,
        "profile": {
            "first_name": user.get('first_name'),
            "last_name": user.get('last_name'),
            "full_name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip(),
            "birth_year": user.get('birth_year'),
            "age": age,
            "job": user.get('job'),
            "salary": user.get('salary'),
            "residence": user.get('residence'),
            "description": user.get('description')
        }
    }

    return json.dumps(result, indent=2)


@tool
def get_user_baseline(user_iban: str) -> str:
    """
    Get behavioral baseline for a user from memory.

    Args:
        user_iban: User IBAN

    Returns:
        JSON with user baseline or "null" if not exists
    """
    import json

    # NOTE: This will be integrated with Alfonso's MemoryManager later
    # For now, return a placeholder structure

    result = {
        "user_iban": user_iban,
        "baseline_exists": False,
        "baseline": None,
        "message": "MemoryManager integration pending - will be connected to Alfonso's Task 8"
    }

    # Placeholder: In production, this would call:
    # from memory_manager import MemoryManager
    # mem_mgr = MemoryManager.get_instance()
    # baselines = mem_mgr.load_user_baselines()
    # baseline = baselines.get(user_iban)

    return json.dumps(result, indent=2)


@tool
def detect_phishing_patterns(communication_text: str) -> str:
    """
    Detect phishing patterns in communication text.

    Args:
        communication_text: Text to analyze

    Returns:
        JSON with detected patterns: {has_urgency, has_threat, has_link, score}
    """
    import json
    import re

    # Define phishing indicators
    urgency_keywords = [
        "urgente", "urgent", "immediately", "subito", "ora", "now",
        "entro", "scadenza", "expir", "deadline", "ultima chance",
        "last chance", "act now", "limited time"
    ]

    threat_keywords = [
        "bloccare", "block", "suspend", "chiudere", "close",
        "terminare", "terminate", "cancellare", "cancel",
        "penalt", "multa", "fine", "legal action", "azione legale"
    ]

    action_keywords = [
        "clicca", "click", "conferma", "confirm", "verifica", "verify",
        "aggiorna", "update", "scarica", "download", "installa", "install"
    ]

    suspicious_phrases = [
        "conferma i tuoi dati", "verify your account", "unusual activity",
        "attività sospetta", "urgent action required", "azione richiesta",
        "account will be closed", "il tuo account", "your account"
    ]

    # Convert to lowercase for matching
    text_lower = communication_text.lower()

    # Check for indicators
    has_urgency = any(keyword in text_lower for keyword in urgency_keywords)
    has_threat = any(keyword in text_lower for keyword in threat_keywords)
    has_action = any(keyword in text_lower for keyword in action_keywords)
    has_suspicious = any(phrase in text_lower for phrase in suspicious_phrases)

    # Check for URLs (potential phishing links)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, communication_text)
    has_link = len(urls) > 0

    # Check for suspicious domains
    suspicious_domains = [
        'paypa1', 'amaz0n', 'g00gle', 'microsof1', 'app1e',
        'bank-secure', 'verify-account', 'secure-login',
        '.net/', '.org/', 'bit.ly', 'tinyurl'
    ]
    suspicious_urls = [
        url for url in urls
        if any(domain in url.lower() for domain in suspicious_domains)
    ]

    # Calculate phishing score (0-1)
    score = 0.0
    if has_urgency:
        score += 0.2
    if has_threat:
        score += 0.25
    if has_action:
        score += 0.15
    if has_suspicious:
        score += 0.2
    if suspicious_urls:
        score += 0.3
    elif has_link:
        score += 0.1

    score = min(score, 1.0)  # Cap at 1.0

    result = {
        "has_urgency": has_urgency,
        "has_threat": has_threat,
        "has_action_request": has_action,
        "has_suspicious_phrases": has_suspicious,
        "has_link": has_link,
        "link_count": len(urls),
        "suspicious_urls": suspicious_urls,
        "phishing_score": round(score, 2),
        "risk_level": "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
    }

    return json.dumps(result, indent=2)


# Geospatial Analyzer Tools

@tool
def get_user_gps_history(user_biotag: str, last_n_hours: int = 48) -> str:
    """
    Get GPS history for a user.

    Args:
        user_biotag: User biotag
        last_n_hours: Hours of history to retrieve (default: 48)

    Returns:
        JSON string with GPS points
    """
    import json
    from datetime import datetime, timedelta

    data_mgr = DataManager.get_instance()
    gps_data = data_mgr.get_user_gps(user_biotag)

    if not gps_data:
        return json.dumps({
            "user_biotag": user_biotag,
            "last_n_hours": last_n_hours,
            "gps_points": [],
            "message": "No GPS data found for this biotag"
        })

    # Filter by time window
    try:
        # Find most recent timestamp
        timestamps = [
            datetime.fromisoformat(point.get('timestamp', '').replace('Z', '+00:00'))
            for point in gps_data
            if point.get('timestamp')
        ]

        if not timestamps:
            return json.dumps({
                "user_biotag": user_biotag,
                "last_n_hours": last_n_hours,
                "gps_points": gps_data,
                "message": "No valid timestamps, returning all points"
            })

        most_recent = max(timestamps)
        cutoff_time = most_recent - timedelta(hours=last_n_hours)

        # Filter points
        recent_points = []
        for point in gps_data:
            try:
                point_time = datetime.fromisoformat(point.get('timestamp', '').replace('Z', '+00:00'))
                if point_time >= cutoff_time:
                    recent_points.append(point)
            except (ValueError, AttributeError):
                continue

    except Exception as e:
        recent_points = gps_data  # Return all if filtering fails

    result = {
        "user_biotag": user_biotag,
        "last_n_hours": last_n_hours,
        "total_points": len(gps_data),
        "filtered_points": len(recent_points),
        "gps_points": recent_points
    }

    return json.dumps(result, indent=2)


@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """
    Calculate distance between two points using Haversine formula.

    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point

    Returns:
        JSON with distance in km
    """
    import json
    from math import radians, cos, sin, asin, sqrt

    # Haversine formula
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c  # Earth radius in km

    result = {
        "point1": {"lat": lat1, "lon": lon1},
        "point2": {"lat": lat2, "lon": lon2},
        "distance_km": round(km, 2),
        "distance_miles": round(km * 0.621371, 2)
    }

    return json.dumps(result, indent=2)


@tool
def check_impossible_travel(
    prev_lat: float, prev_lon: float, prev_time: str,
    curr_lat: float, curr_lon: float, curr_time: str
) -> str:
    """
    Check if travel between two points is physically impossible.

    Args:
        prev_lat: Previous latitude
        prev_lon: Previous longitude
        prev_time: Previous timestamp (ISO format)
        curr_lat: Current latitude
        curr_lon: Current longitude
        curr_time: Current timestamp (ISO format)

    Returns:
        JSON with analysis: {is_impossible, distance_km, time_hours, required_speed_kmh}
    """
    import json
    from math import radians, cos, sin, asin, sqrt
    from datetime import datetime

    # Calculate distance using Haversine
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [prev_lat, prev_lon, curr_lat, curr_lon])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    distance_km = 6371 * c

    # Calculate time difference
    try:
        prev_dt = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
        curr_dt = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
        time_diff = curr_dt - prev_dt
        time_hours = time_diff.total_seconds() / 3600

        if time_hours <= 0:
            return json.dumps({
                "error": "Current time is before or equal to previous time",
                "prev_time": prev_time,
                "curr_time": curr_time
            })

        # Calculate required speed
        required_speed_kmh = distance_km / time_hours

        # Threshold: >800 km/h is considered impossible
        # (Max commercial flight speed ~900 km/h, but accounting for boarding, taxi, etc.)
        is_impossible = required_speed_kmh > 800

        result = {
            "is_impossible": is_impossible,
            "distance_km": round(distance_km, 2),
            "time_hours": round(time_hours, 2),
            "required_speed_kmh": round(required_speed_kmh, 2),
            "assessment": "IMPOSSIBLE" if is_impossible else "POSSIBLE",
            "details": {
                "previous_location": {"lat": prev_lat, "lon": prev_lon, "time": prev_time},
                "current_location": {"lat": curr_lat, "lon": curr_lon, "time": curr_time}
            }
        }

    except (ValueError, AttributeError) as e:
        result = {
            "error": f"Invalid timestamp format: {str(e)}",
            "prev_time": prev_time,
            "curr_time": curr_time
        }

    return json.dumps(result, indent=2)


@tool
def get_user_residence(user_iban: str) -> str:
    """
    Get user's residence location.

    Args:
        user_iban: User IBAN

    Returns:
        JSON with residence info (location_name, lat, lon)
    """
    import json

    data_mgr = DataManager.get_instance()
    user = data_mgr.get_user(user_iban)

    if not user:
        return json.dumps({
            "user_iban": user_iban,
            "found": False,
            "message": "User not found"
        })

    residence = user.get('residence', {})

    result = {
        "user_iban": user_iban,
        "found": True,
        "residence": {
            "city": residence.get('city'),
            "latitude": float(residence.get('lat')) if residence.get('lat') else None,
            "longitude": float(residence.get('lng')) if residence.get('lng') else None,
            "coordinates": f"{residence.get('lat')}, {residence.get('lng')}" if residence.get('lat') else None
        }
    }

    return json.dumps(result, indent=2)


# Example/Test Tool (can be removed later)

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
