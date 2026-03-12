# TODO List - Fraud Detection Multi-Agent System

**Progetto**: Implementazione sistema multi-agente per fraud detection
**Fonte**: fraud-detection-plan.md
**Data creazione**: 2026-03-12

---

## Progress Overview

- **Completati**: 1/13
- **In corso**: 1/13
- **Da fare**: 11/13

---

## Task Status

### ✅ Task Completati

#### Task 1: Creare Pydantic schemas per structured outputs (15+ models)
**Status**: ✅ COMPLETATO
**File**: `ai_orchestration.py` (Section 1)

**Modelli creati**:
- `RiskLevel` (Enum)
- `TransactionFeatures`
- `TransactionAnomalyScore`
- `TransactionAnalysisResult`
- `UserBehaviorBaseline`
- `BehavioralAnomalyResult`
- `LocationPoint`
- `GeospatialAnalysisResult`
- `FraudEvidence`
- `FraudDecision`
- `FraudPattern`
- `AdaptiveLearningState`
- `AgentTask`
- `AgentResponse`

---

### 🔄 Task In Corso

#### Task 2: Implementare DataManager class per caricamento dati
**Status**: 🔄 IN PROGRESS
**File**: `ai_orchestration.py`

**Obiettivo**: Classe singleton per caricare e cachare tutti i dati all'avvio:
- Transactions CSV
- Users CSV
- Locations CSV
- Communications CSV
- GPS data CSV

**Metodi da implementare**:
- `load_transactions(level: int) -> List[Dict]`
- `load_users() -> Dict[str, Dict]`
- `load_locations() -> Dict[str, Dict]`
- `load_communications() -> Dict[str, List[Dict]]`
- `load_gps_data() -> Dict[str, List[Dict]]`

---

### ⏳ Task Da Fare

### Foundation - Data Access & Tools

#### Task 3: Implementare @tool functions per Transaction Analyzer (4 tools)
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py` (Section 5)

**Tools da creare**:
1. `get_user_transaction_history(user_iban: str) -> List[Dict]`
2. `calculate_transaction_velocity(user_iban: str, time_window_hours: int) -> Dict`
3. `get_recipient_profile(recipient_iban: str) -> Dict`
4. `query_fraud_memory(pattern_type: str) -> List[FraudPattern]`

---

#### Task 4: Implementare @tool functions per Behavioral Profiler (4 tools)
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py` (Section 5)

**Tools da creare**:
1. `get_user_communications(user_biotag: str) -> List[Dict]`
2. `get_user_profile(user_iban: str) -> Dict`
3. `get_user_baseline(user_iban: str) -> Optional[UserBehaviorBaseline]`
4. `detect_phishing_patterns(communication_text: str) -> Dict`

---

#### Task 5: Implementare @tool functions per Geospatial Analyzer (4 tools)
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py` (Section 5)

**Tools da creare**:
1. `get_user_gps_history(user_biotag: str, last_n_hours: int) -> List[LocationPoint]`
2. `calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float`
3. `check_impossible_travel(prev_location: LocationPoint, curr_location: LocationPoint) -> Dict`
4. `get_user_residence(user_iban: str) -> Optional[Dict]`

---

### Agent Configuration

#### Task 6: Creare prompts per i 4 agenti
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py` (Section 2)

**Prompts da creare**:
1. `TRANSACTION_ANALYZER_PROMPT`
2. `BEHAVIORAL_PROFILER_PROMPT`
3. `GEOSPATIAL_ANALYZER_PROMPT`
4. `FRAUD_ORCHESTRATOR_PROMPT`

**Requisiti**: Ogni prompt deve specificare:
- Ruolo dell'agente
- Tools disponibili
- Output format (structured)
- Strategie di analisi
- Edge cases da considerare

---

#### Task 7: Implementare agent factory functions con structured output
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py`

**Factory functions da creare**:
1. `create_transaction_analyzer_agent() -> Agent`
   - Output: `TransactionAnalysisResult`
2. `create_behavioral_profiler_agent() -> Agent`
   - Output: `BehavioralAnomalyResult`
3. `create_geospatial_analyzer_agent() -> Agent`
   - Output: `GeospatialAnalysisResult`
4. `create_fraud_orchestrator_agent() -> Agent`
   - Output: `FraudDecision`

**Pattern da usare**: `.with_structured_output(Schema)`

---

### Memory & Persistence

#### Task 8: Implementare MemoryManager class per persistence JSON
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py`

**Classe singleton per gestire**:
- `fraud_patterns.json`: Lista di FraudPattern
- `user_baselines.json`: Dict[user_id, UserBehaviorBaseline]
- `learning_state.json`: AdaptiveLearningState

**Metodi da implementare**:
- `load_fraud_patterns() -> List[FraudPattern]`
- `save_fraud_pattern(pattern: FraudPattern) -> None`
- `load_user_baselines() -> Dict[str, UserBehaviorBaseline]`
- `save_user_baseline(baseline: UserBehaviorBaseline) -> None`
- `load_learning_state() -> AdaptiveLearningState`
- `save_learning_state(state: AdaptiveLearningState) -> None`

---

#### Task 9: Creare directory memory/ con file JSON iniziali
**Status**: ⏳ PENDING
**Directory**: `memory/`

**Files da creare**:
```
memory/
  ├── fraud_patterns.json      # []
  ├── user_baselines.json      # {}
  └── learning_state.json      # initial state
```

**Initial learning_state.json**:
```json
{
  "current_level": 1,
  "total_transactions_analyzed": 0,
  "total_frauds_detected": 0,
  "fraud_patterns_count": 0,
  "user_baselines_count": 0,
  "decision_threshold": 0.5,
  "last_adaptation_timestamp": "2026-03-12T00:00:00Z",
  "performance_metrics": {}
}
```

---

### Orchestration & Execution

#### Task 10: Implementare analyze_transaction function (orchestrazione)
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py`

**Signature**:
```python
def analyze_transaction(
    transaction: Dict,
    session_id: str
) -> FraudDecision:
```

**Flow**:
1. Invoke Transaction Analyzer agent → `TransactionAnalysisResult`
2. Invoke Behavioral Profiler agent → `BehavioralAnomalyResult`
3. Invoke Geospatial Analyzer agent → `GeospatialAnalysisResult`
4. Invoke Fraud Orchestrator agent with all results → `FraudDecision`
5. Update memory (fraud patterns, user baselines, learning state)
6. Return `FraudDecision`

---

#### Task 11: Implementare process_level function (main execution loop)
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py`

**Signature**:
```python
def process_level(
    level: int,
    data_dir: str,
    session_id: str
) -> None:
```

**Flow**:
1. Load transactions for level
2. For each transaction:
   - Call `analyze_transaction()`
   - Collect results
   - Show progress
3. Generate output file: `level_{level}_results.jsonl`
4. Calculate and print metrics (fraud rate, confidence avg, etc.)

**Output format**: JSONL with one FraudDecision per line

---

#### Task 12: Aggiungere CLI arguments parser (--level, --data-dir)
**Status**: ⏳ PENDING
**File**: `ai_orchestration.py`

**CLI Interface**:
```bash
python ai_orchestration.py --level 1 --data-dir ./data
```

**Arguments**:
- `--level`: Challenge level to process (1-5)
- `--data-dir`: Path to data directory (default: current dir)

**Update main()** per gestire CLI args e chiamare `process_level()`

---

### Testing

#### Task 13: Testare dry run su training data Level 1
**Status**: ⏳ PENDING

**Test Plan**:
1. Run: `python ai_orchestration.py --level 1 --data-dir ./training_data`
2. Verify:
   - All agents invoke successfully
   - Structured outputs are valid
   - Memory files update correctly
   - Output file generated
   - Fraud rate ~10-30% (reasonable)
   - No crashes/errors
3. Check Langfuse traces
4. Review output quality

---

## Ordine di Implementazione Raccomandato

### Phase 1: Foundation (Tasks 1-5)
- ✅ Task 1: Pydantic schemas
- 🔄 Task 2: DataManager
- Task 3: Transaction Analyzer tools
- Task 4: Behavioral Profiler tools
- Task 5: Geospatial Analyzer tools

### Phase 2: Agents & Memory (Tasks 6-9)
- Task 6: Agent prompts
- Task 7: Agent factory functions
- Task 8: MemoryManager class
- Task 9: Memory directory setup

### Phase 3: Execution (Tasks 10-12)
- Task 10: analyze_transaction orchestration
- Task 11: process_level main loop
- Task 12: CLI arguments

### Phase 4: Validation (Task 13)
- Task 13: End-to-end testing

---

## Note di Implementazione

### Principi chiave
1. **Single file architecture**: Tutto in `ai_orchestration.py`
2. **Structured outputs**: Usare `.with_structured_output()` per tutti gli agenti
3. **Observability**: Langfuse tracking su tutte le invocazioni
4. **Memory persistence**: Salvare patterns e baselines dopo ogni transazione
5. **Minimalism**: Implementare solo ciò che è necessario per PDR

### Vincoli
- No database esterni (solo JSON files)
- No async/parallel processing (sequential per semplicità)
- No complex ML models (solo rule-based + LLM reasoning)

---

## Resources

- **Plan Document**: `fraud-detection-plan.md`
- **Main File**: `ai_orchestration.py`
- **Data Directory**: `./training_data/` o `./evaluation_data/`
- **Memory Directory**: `./memory/`
- **Challenge PDF**: `AIAgentChallenge_2026.pdf`

---

**Last Updated**: 2026-03-12
**Status**: 1/13 tasks completed (7.7%)
