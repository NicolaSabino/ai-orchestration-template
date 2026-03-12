# TODO NICOLA - Path Critico (Data Access + Orchestration)

**Ruolo**: Core implementation - Data access layer + Orchestration logic
**Dipendenze**: Path critico - gli altri aspettano alcuni tuoi task
**Priorità**: ALTA - Il tuo lavoro sblocca Laura e Alfonso

---

## 📊 Progress

- **Completati**: 1/8
- **In corso**: 1/8
- **Da fare**: 6/8

---

## ✅ Task Completati

### Task 1: Pydantic schemas ✅
**Status**: COMPLETATO
- 15+ modelli creati in `ai_orchestration.py` (Section 1)
- Tutti i modelli testati e funzionanti

---

## 🔄 Task In Corso

### Task 2: DataManager class 🔄
**File**: `ai_orchestration.py`
**Status**: IN PROGRESS
**Priorità**: 🔥 URGENTE - sblocca Task 3, 4, 5

**Obiettivo**: Classe singleton per caricare e cachare tutti i dati CSV

**Implementazione**:
```python
class DataManager:
    """Singleton per gestire tutti i dati CSV."""
    _instance = None

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._transactions = {}  # {level: List[Dict]}
        self._users = {}         # {iban: Dict}
        self._locations = {}     # {location_id: Dict}
        self._communications = {} # {biotag: List[Dict]}
        self._gps_data = {}      # {biotag: List[Dict]}

    @classmethod
    def get_instance(cls, data_dir: str = "."):
        if cls._instance is None:
            cls._instance = cls(data_dir)
        return cls._instance

    def load_all_data(self, level: int):
        """Load all data for a specific level."""
        self.load_transactions(level)
        self.load_users()
        self.load_locations()
        self.load_communications()
        self.load_gps_data()

    def load_transactions(self, level: int) -> List[Dict]:
        """Load transactions CSV for a level."""
        # Read CSV from f"{self.data_dir}/level_{level}_transactions.csv"
        # Cache in self._transactions[level]
        pass

    def load_users(self) -> Dict[str, Dict]:
        """Load users CSV."""
        # Read CSV from f"{self.data_dir}/users.csv"
        # Index by IBAN
        pass

    def load_locations(self) -> Dict[str, Dict]:
        """Load locations CSV."""
        # Read CSV from f"{self.data_dir}/locations.csv"
        pass

    def load_communications(self) -> Dict[str, List[Dict]]:
        """Load communications CSV, grouped by biotag."""
        # Read CSV from f"{self.data_dir}/communications.csv"
        # Group by sender biotag
        pass

    def load_gps_data(self) -> Dict[str, List[Dict]]:
        """Load GPS CSV, grouped by biotag."""
        # Read CSV from f"{self.data_dir}/gps.csv"
        # Group by biotag, sort by timestamp
        pass

    def get_transactions(self, level: int) -> List[Dict]:
        return self._transactions.get(level, [])

    def get_user(self, iban: str) -> Optional[Dict]:
        return self._users.get(iban)

    def get_user_communications(self, biotag: str) -> List[Dict]:
        return self._communications.get(biotag, [])

    def get_user_gps(self, biotag: str) -> List[Dict]:
        return self._gps_data.get(biotag, [])
```

**Files da leggere**:
- `level_{level}_transactions.csv`
- `users.csv`
- `locations.csv`
- `communications.csv`
- `gps.csv`

**Posizione nel file**: Dopo Section 4 (Observability), prima di Section 5 (Tools)

---

## ⏳ Task Da Fare

### Task 3: @tool functions per Transaction Analyzer
**File**: `ai_orchestration.py` (Section 5)
**Dipende da**: Task 2 (DataManager)
**Priorità**: ALTA
**Tempo stimato**: 30-45 min

**Tools da implementare**:

```python
@tool
def get_user_transaction_history(user_iban: str, limit: int = 50) -> str:
    """
    Get transaction history for a user.

    Args:
        user_iban: User IBAN to query
        limit: Max number of transactions to return

    Returns:
        JSON string with transaction history
    """
    data_mgr = DataManager.get_instance()
    # Filter transactions where sender_iban == user_iban
    # Return last N transactions as JSON string
    pass

@tool
def calculate_transaction_velocity(user_iban: str, time_window_hours: int = 24) -> str:
    """
    Calculate transaction velocity for a user.

    Args:
        user_iban: User IBAN
        time_window_hours: Time window in hours

    Returns:
        JSON with velocity metrics: {count, total_amount, avg_amount, max_amount}
    """
    # Count transactions in time window
    # Calculate stats
    pass

@tool
def get_recipient_profile(recipient_iban: str) -> str:
    """
    Get profile of recipient IBAN.

    Args:
        recipient_iban: Recipient IBAN

    Returns:
        JSON with recipient info and statistics
    """
    data_mgr = DataManager.get_instance()
    # Get user info
    # Count how many times this IBAN received money
    # Return as JSON string
    pass

@tool
def query_fraud_memory(pattern_type: str = "all") -> str:
    """
    Query fraud patterns from memory.

    Args:
        pattern_type: Type of pattern to query ("all", "behavioral", "geospatial", etc.)

    Returns:
        JSON string with matching fraud patterns
    """
    from memory_manager import MemoryManager  # Alfonso lo farà
    mem_mgr = MemoryManager.get_instance()
    patterns = mem_mgr.load_fraud_patterns()
    # Filter by pattern_type if not "all"
    # Return as JSON string
    pass
```

---

### Task 4: @tool functions per Behavioral Profiler
**File**: `ai_orchestration.py` (Section 5)
**Dipende da**: Task 2 (DataManager)
**Priorità**: ALTA
**Tempo stimato**: 30-45 min

**Tools da implementare**:

```python
@tool
def get_user_communications(user_biotag: str, limit: int = 20) -> str:
    """
    Get recent communications for a user.

    Args:
        user_biotag: User biotag
        limit: Max communications to return

    Returns:
        JSON string with communications
    """
    data_mgr = DataManager.get_instance()
    comms = data_mgr.get_user_communications(user_biotag)
    # Return last N as JSON string
    pass

@tool
def get_user_profile(user_iban: str) -> str:
    """
    Get user profile information.

    Args:
        user_iban: User IBAN

    Returns:
        JSON with user profile (name, age, location, etc.)
    """
    data_mgr = DataManager.get_instance()
    user = data_mgr.get_user(user_iban)
    # Return as JSON string
    pass

@tool
def get_user_baseline(user_iban: str) -> str:
    """
    Get behavioral baseline for a user from memory.

    Args:
        user_iban: User IBAN

    Returns:
        JSON with user baseline or "null" if not exists
    """
    from memory_manager import MemoryManager
    mem_mgr = MemoryManager.get_instance()
    baselines = mem_mgr.load_user_baselines()
    baseline = baselines.get(user_iban)
    # Return as JSON string
    pass

@tool
def detect_phishing_patterns(communication_text: str) -> str:
    """
    Detect phishing patterns in communication text.

    Args:
        communication_text: Text to analyze

    Returns:
        JSON with detected patterns: {has_urgency, has_threat, has_link, score}
    """
    # Simple rule-based detection
    # Keywords: "urgente", "bloccare", "confermare", "clicca", URLs, etc.
    # Return JSON string with results
    pass
```

---

### Task 5: @tool functions per Geospatial Analyzer
**File**: `ai_orchestration.py` (Section 5)
**Dipende da**: Task 2 (DataManager)
**Priorità**: ALTA
**Tempo stimato**: 30-45 min

**Tools da implementare**:

```python
from math import radians, cos, sin, asin, sqrt

@tool
def get_user_gps_history(user_biotag: str, last_n_hours: int = 48) -> str:
    """
    Get GPS history for a user.

    Args:
        user_biotag: User biotag
        last_n_hours: Hours of history to retrieve

    Returns:
        JSON string with GPS points
    """
    data_mgr = DataManager.get_instance()
    gps_data = data_mgr.get_user_gps(user_biotag)
    # Filter by timestamp (last N hours)
    # Return as JSON string
    pass

@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """
    Calculate distance between two points using Haversine formula.

    Returns:
        JSON with distance in km
    """
    # Haversine formula
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return json.dumps({"distance_km": round(km, 2)})

@tool
def check_impossible_travel(
    prev_lat: float, prev_lon: float, prev_time: str,
    curr_lat: float, curr_lon: float, curr_time: str
) -> str:
    """
    Check if travel between two points is physically impossible.

    Returns:
        JSON with analysis: {is_impossible, distance_km, time_hours, required_speed_kmh}
    """
    from datetime import datetime
    # Calculate distance
    # Calculate time difference
    # Calculate required speed
    # Threshold: >800 km/h = impossible (max commercial flight ~900 km/h)
    pass

@tool
def get_user_residence(user_iban: str) -> str:
    """
    Get user's residence location.

    Args:
        user_iban: User IBAN

    Returns:
        JSON with residence info (location_name, lat, lon)
    """
    data_mgr = DataManager.get_instance()
    user = data_mgr.get_user(user_iban)
    # Get location_id from user
    # Look up in locations
    # Return as JSON string
    pass
```

---

### Task 10: analyze_transaction function (Orchestrazione)
**File**: `ai_orchestration.py`
**Dipende da**: Task 7 (Laura), Task 8 (Alfonso)
**Priorità**: MEDIA (aspetta Task 7, 8)
**Tempo stimato**: 45-60 min

**Implementazione**:

```python
def analyze_transaction(
    transaction: Dict,
    session_id: str,
    agents: Dict[str, Any]
) -> FraudDecision:
    """
    Orchestrate fraud analysis for a single transaction.

    Args:
        transaction: Transaction data
        session_id: Langfuse session ID
        agents: Dict with all 4 agents

    Returns:
        FraudDecision from orchestrator
    """
    # Step 1: Invoke Transaction Analyzer
    trans_query = f"Analyze this transaction: {json.dumps(transaction)}"
    trans_result = run_agent_with_tracking(
        agents["transaction"],
        trans_query,
        session_id
    )

    # Step 2: Invoke Behavioral Profiler
    behav_query = f"Analyze behavioral patterns: {json.dumps(transaction)}"
    behav_result = run_agent_with_tracking(
        agents["behavioral"],
        behav_query,
        session_id
    )

    # Step 3: Invoke Geospatial Analyzer
    geo_query = f"Analyze geospatial data: {json.dumps(transaction)}"
    geo_result = run_agent_with_tracking(
        agents["geospatial"],
        geo_query,
        session_id
    )

    # Step 4: Invoke Orchestrator with all results
    orchestrator_query = f"""
    Make fraud decision based on these analyses:

    Transaction Analysis: {trans_result}
    Behavioral Analysis: {behav_result}
    Geospatial Analysis: {geo_result}

    Transaction: {json.dumps(transaction)}
    """

    decision = run_agent_with_tracking(
        agents["orchestrator"],
        orchestrator_query,
        session_id
    )

    # Step 5: Update memory (fraud patterns, baselines)
    # TODO: Extract patterns, update baselines

    return decision
```

---

### Task 11: process_level function (Main execution loop)
**File**: `ai_orchestration.py`
**Dipende da**: Task 10
**Priorità**: BASSA (fine pipeline)
**Tempo stimato**: 30 min

```python
def process_level(level: int, data_dir: str, session_id: str):
    """
    Process all transactions for a level.

    Args:
        level: Challenge level (1-5)
        data_dir: Data directory path
        session_id: Langfuse session ID
    """
    print(f"Processing Level {level}")
    print("=" * 70)

    # Load data
    data_mgr = DataManager.get_instance(data_dir)
    data_mgr.load_all_data(level)
    transactions = data_mgr.get_transactions(level)

    print(f"Loaded {len(transactions)} transactions")

    # Create agents
    model = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        temperature=TEMPERATURE
    )

    agents = {
        "transaction": create_transaction_analyzer_agent(model),
        "behavioral": create_behavioral_profiler_agent(model),
        "geospatial": create_geospatial_analyzer_agent(model),
        "orchestrator": create_fraud_orchestrator_agent(model)
    }

    # Process transactions
    results = []
    for i, transaction in enumerate(transactions, 1):
        print(f"\n[{i}/{len(transactions)}] Analyzing transaction {transaction['id']}...")

        decision = analyze_transaction(transaction, session_id, agents)
        results.append(decision.dict())

        fraud_status = "🔴 FRAUD" if decision.is_fraudulent else "🟢 LEGIT"
        print(f"  Result: {fraud_status} (confidence: {decision.confidence:.2f})")

    # Save results
    output_file = f"level_{level}_results.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print stats
    fraud_count = sum(1 for r in results if r['is_fraudulent'])
    fraud_rate = fraud_count / len(results) * 100
    avg_confidence = sum(r['confidence'] for r in results) / len(results)

    print("\n" + "=" * 70)
    print(f"Level {level} Complete!")
    print(f"Frauds detected: {fraud_count}/{len(results)} ({fraud_rate:.1f}%)")
    print(f"Avg confidence: {avg_confidence:.2f}")
    print(f"Output: {output_file}")
    print("=" * 70)
```

---

### Task 13: Testing su Level 1
**Dipende da**: Tutti i task precedenti
**Priorità**: FINALE
**Tempo stimato**: 20 min

**Test plan**:
1. Run: `python ai_orchestration.py --level 1 --data-dir ./The\ Truman\ Show_train`
2. Verify:
   - No errors/crashes
   - All agents invoke successfully
   - Output file generated
   - Fraud rate reasonable (10-30%)
   - Langfuse traces visible
3. Review quality of decisions
4. Debug and fix issues

---

## 🔗 Dipendenze da Altri

**Da Laura**:
- Task 7 (Agent factories) - necessario per Task 10

**Da Alfonso**:
- Task 8 (MemoryManager) - necessario per Task 10
- Task 12 (CLI args) - necessario per Task 11

---

## 💡 Note Importanti

1. **DataManager è CRITICO**: Laura e Alfonso aspettano che tu finisca Task 2 per poter testare i loro tool

2. **Tool returns**: Tutti i tool devono ritornare JSON strings (non Dict), perché gli agenti LLM si aspettano stringhe

3. **Error handling**: Gestisci gracefully i casi in cui i dati non esistono (user not found, etc.)

4. **Testing incrementale**: Testa ogni tool appena lo scrivi con piccoli script

5. **Coordinate con il team**:
   - Avvisa quando Task 2 è pronto (sblocca Task 3, 4, 5)
   - Aspetta Task 7 e 8 prima di Task 10

---

**Prossimo task**: Finire Task 2 (DataManager) ASAP!
