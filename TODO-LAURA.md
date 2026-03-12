# TODO LAURA - Agent Configuration (Prompts + Factories)

**Ruolo**: AI Agent Expert - Prompts engineering + Agent factory setup
**Dipendenze**: Indipendente - puoi iniziare SUBITO
**Priorità**: ALTA - I tuoi prompts servono per Task 7 e poi per l'orchestrazione

---

## 📊 Progress

- **Completati**: 0/2
- **Da fare**: 2/2

---

## ⏳ Task Da Fare

### Task 6: Creare prompts per i 4 agenti 🔥
**File**: `ai_orchestration.py` (Section 2)
**Dipende da**: NIENTE - puoi iniziare SUBITO
**Priorità**: 🔥 URGENTE - serve per Task 7
**Tempo stimato**: 60-90 min

**Obiettivo**: Creare 4 system prompts dettagliati per gli agenti specializzati

**Posizione nel file**: Section 2 (PROMPT TEMPLATES) - sostituire `FOO_AGENT_PROMPT`

---

#### Prompt 1: TRANSACTION_ANALYZER_PROMPT

```python
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
```

---

#### Prompt 2: BEHAVIORAL_PROFILER_PROMPT

```python
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
```

---

#### Prompt 3: GEOSPATIAL_ANALYZER_PROMPT

```python
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
```

---

#### Prompt 4: FRAUD_ORCHESTRATOR_PROMPT

```python
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
```

---

### Task 7: Implementare agent factory functions 🔥
**File**: `ai_orchestration.py`
**Dipende da**: Task 6 (i tuoi prompts), Task 3-5 (tools di Nicola)
**Priorità**: ALTA - necessario per orchestrazione
**Tempo stimato**: 30-45 min

**Obiettivo**: Creare 4 factory functions che istanziano agenti con structured output

**Nota**: Puoi scrivere le factories SUBITO usando i tool di esempio. Quando Nicola finisce Task 3-5, basta aggiornare i tool array.

**Implementazione**:

```python
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
        query_fraud_memory
    ]

    agent = create_agent(
        model=model,
        system_prompt=TRANSACTION_ANALYZER_PROMPT,
        tools=tools
    )

    # Add structured output
    agent_with_output = agent.with_structured_output(TransactionAnalysisResult)

    return agent_with_output


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
        detect_phishing_patterns
    ]

    agent = create_agent(
        model=model,
        system_prompt=BEHAVIORAL_PROFILER_PROMPT,
        tools=tools
    )

    agent_with_output = agent.with_structured_output(BehavioralAnomalyResult)

    return agent_with_output


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
        get_user_residence
    ]

    agent = create_agent(
        model=model,
        system_prompt=GEOSPATIAL_ANALYZER_PROMPT,
        tools=tools
    )

    agent_with_output = agent.with_structured_output(GeospatialAnalysisResult)

    return agent_with_output


def create_fraud_orchestrator_agent(model: ChatOpenAI):
    """
    Create Fraud Orchestrator agent with structured output.

    Args:
        model: LLM model instance

    Returns:
        Agent that returns FraudDecision
    """
    # Orchestrator doesn't need tools - it just synthesizes
    agent = create_agent(
        model=model,
        system_prompt=FRAUD_ORCHESTRATOR_PROMPT,
        tools=[]
    )

    agent_with_output = agent.with_structured_output(FraudDecision)

    return agent_with_output
```

**Posizione nel file**: Dopo Section 5 (Tools), creare nuova Section "Agent Factory Functions"

---

## 🔗 Dipendenze

**Aspetta da Nicola**:
- Task 3, 4, 5 (Tools) - per completare l'array di tools in Task 7
- Ma puoi già scrivere le factories con tool placeholder!

**Alfonso** non ha dipendenze da te

---

## ✅ Testing delle Factories

Quando hai finito Task 7, testa le factories:

```python
# Test script
if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL
    )

    # Test creation
    trans_agent = create_transaction_analyzer_agent(model)
    behav_agent = create_behavioral_profiler_agent(model)
    geo_agent = create_geospatial_analyzer_agent(model)
    orch_agent = create_fraud_orchestrator_agent(model)

    print("✅ All agents created successfully")

    # Test invocation (simple test)
    test_query = "Analyze this test transaction: {}"
    try:
        result = trans_agent.invoke({"messages": [HumanMessage(test_query)]})
        print(f"✅ Transaction agent works: {type(result)}")
    except Exception as e:
        print(f"❌ Error: {e}")
```

---

## 💡 Tips per i Prompts

1. **Be specific**: Gli agenti devono sapere esattamente cosa fare
2. **List tools**: Specifica i tool disponibili nel prompt
3. **Give examples**: Indica esempi di anomalie da cercare
4. **Set thresholds**: Specifica quando qualcosa è "suspicious" vs "normal"
5. **Output format**: Specifica chiaramente il formato di output
6. **Edge cases**: Menziona casi limite (no data, missing fields, etc.)

---

## 📋 Checklist

Task 6:
- [ ] Scritto TRANSACTION_ANALYZER_PROMPT
- [ ] Scritto BEHAVIORAL_PROFILER_PROMPT
- [ ] Scritto GEOSPATIAL_ANALYZER_PROMPT
- [ ] Scritto FRAUD_ORCHESTRATOR_PROMPT
- [ ] Prompts testati per chiarezza e completezza

Task 7:
- [ ] Scritto create_transaction_analyzer_agent()
- [ ] Scritto create_behavioral_profiler_agent()
- [ ] Scritto create_geospatial_analyzer_agent()
- [ ] Scritto create_fraud_orchestrator_agent()
- [ ] Testato che le factories creino agenti correttamente
- [ ] Verificato structured output funzioni

---

**Prossimo task**: Inizia con Task 6 (Prompts) - è URGENTE!
