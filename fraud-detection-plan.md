# Piano di Implementazione: Sistema Multi-Agente per Fraud Detection

## Obiettivo
Creare un sistema di agenti orchestrati per rilevare frodi finanziarie in ambienti con pattern evolutivi (concept drift), utilizzando dati multimodali strutturati e memoria adattiva.

## Contesto del Challenge
- **Challenge**: Reply Mirror AI Agent - Fraud Detection 2026
- **Obiettivo**: Identificare transazioni fraudolente su 5 livelli di complessità crescente
- **Vincolo critico**: Solo 1 submission per level (no retry)
- **Scoring**: Accuracy + stabilità temporale + adattabilità + costi + velocità

## Requisiti Confermati
- **LLM**: OpenRouter + OpenAI (non Ollama)
- **Persistence**: JSON files + Pydantic schema validation
- **Structured Outputs**: Tutti gli scambi inter-agente usano Pydantic models
- **Agenti**: Transaction Analyzer, Behavioral Profiler, Geospatial Analyzer, Fraud Orchestrator
- **Apprendimento**: Memory-based learning (pattern memory che evolve)
- **Decisione**: LLM-based reasoning (Orchestrator analizza e decide)
- **Baseline**: Cold start (nessun baseline pre-calcolato, costruiti on-the-fly)
- **Data Access**: Smart lazy loading con cache
- **Validation**: No validation interna (submission diretta)

## Architettura del Sistema

### 1. Componenti Principali

```
┌─────────────────────────────────────────────────────────────┐
│              FRAUD ORCHESTRATOR AGENT (LLM)                 │
│  - Riceve analisi da 3 agenti specializzati                │
│  - Ragiona su evidenze aggregate                           │
│  - Decide: fraudulent/legitimate + confidence              │
│  - Aggiorna pattern memory                                 │
└───────┬──────────────────────────────────┬─────────────────┘
        │ Pydantic Messages                │
┌───────▼──────────┬──────────────┬────────▼─────────────────┐
│ Transaction      │ Behavioral   │ Geospatial              │
│ Analyzer         │ Profiler     │ Analyzer                │
│ (LLM + Tools)    │ (LLM + Tools)│ (LLM + Tools)           │
└───────┬──────────┴──────┬───────┴────────┬────────────────┘
        │ Tools            │                │
┌───────▼──────────────────▼────────────────▼────────────────┐
│            DATA ACCESS LAYER (Smart Lazy Loading)          │
│  - DataManager: cache + lazy loading                       │
│  - Tools: @tool functions per ogni agente                  │
│  - CrossReferencer: IBAN → User → BioTag mapping          │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│          MEMORY LAYER (JSON Persistence)                   │
│  - fraud_patterns.json: pattern scoperti                   │
│  - user_baselines.json: profili comportamentali dinamici   │
│  - learning_state.json: stato apprendimento globale        │
└────────────────────────────────────────────────────────────┘
```

### 2. Flusso di Elaborazione

**Per ogni transazione:**

1. **Invocazione Sequenziale Agenti**: Orchestrator chiama i 3 agenti uno dopo l'altro
   - Transaction Analyzer: anomalie quantitative (importi, frequenza, timing)
   - Behavioral Profiler: deviazioni comportamentali (SMS/email/profilo utente)
   - Geospatial Analyzer: impossibilità spaziali (GPS vs transaction location)

2. **Structured Output**: Ogni agente ritorna Pydantic model:
   - `TransactionAnalysisResult`
   - `BehavioralAnomalyResult`
   - `GeospatialAnalysisResult`

3. **LLM-Based Decision**: Orchestrator riceve i 3 risultati e:
   - Analizza evidenze con reasoning LLM
   - Consulta pattern memory per match
   - Produce `FraudDecision` (is_fraudulent, confidence, reasoning)

4. **Memory Update**: Se frode rilevata o nuovo pattern scoperto:
   - Estrae pattern features
   - Salva in `fraud_patterns.json`
   - Aggiorna baseline utente in `user_baselines.json`

## File da Modificare/Creare

### File Principale
- **`ai_orchestration.py`** (già esistente): Espandere con tutta l'architettura multi-agente
  - Sezione Pydantic Schemas (15+ models)
  - Sezione Agent Prompts (4 prompts dettagliati)
  - Sezione Tools (12+ @tool functions)
  - Sezione Memory Management (MemoryManager class)
  - Sezione Agent Creation (4 factory functions con structured output)
  - Sezione Main Execution (analyze_transaction, process_level)

### Directory da Creare
- **`memory/`**: Directory per persistence JSON
  - `fraud_patterns.json`
  - `user_baselines.json`
  - `learning_state.json`

### Data Sources (già presenti, read-only)
- `The Truman Show_train/public/transactions.csv`
- `The Truman Show_train/public/users.json`
- `The Truman Show_train/public/locations.json`
- `The Truman Show_train/public/sms.json`
- `The Truman Show_train/public/mails.json`

## Implementazione Dettagliata

### STEP 1: Pydantic Schemas (Structured Outputs)

Definire 15+ Pydantic models in `ai_orchestration.py`:

**Core Models:**
- `TransactionFeatures`: dati estratti da CSV
- `TransactionAnomalyScore`: score anomalie quantitative
- `TransactionAnalysisResult`: output Transaction Analyzer
- `UserBehaviorBaseline`: profilo comportamentale utente
- `BehavioralAnomalyResult`: output Behavioral Profiler
- `LocationPoint`: punto GPS
- `GeospatialAnalysisResult`: output Geospatial Analyzer
- `FraudEvidence`: singola evidenza di frode
- `FraudDecision`: decisione finale orchestrator

**Memory Models:**
- `FraudPattern`: pattern fraudolento scoperto e memorizzato
- `AdaptiveLearningState`: stato globale apprendimento

**Communication Models:**
- `AgentTask`: task assegnato ad agente
- `AgentResponse`: risposta generica agente

### STEP 2: Data Access Layer (Simple Loading)

**DataManager Class:**
```python
class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.transactions_df = None
        self.users_dict = {}
        self.locations_list = []
        self.sms_list = []
        self.mails_list = []

    def load_all_data(self):
        """Load all data at startup - simple approach"""
        self.transactions_df = pd.read_csv(f"{self.data_dir}/transactions.csv")

        with open(f"{self.data_dir}/users.json") as f:
            users = json.load(f)
            self.users_dict = {u['iban']: u for u in users}

        with open(f"{self.data_dir}/locations.json") as f:
            self.locations_list = json.load(f)

        with open(f"{self.data_dir}/sms.json") as f:
            self.sms_list = json.load(f)

        with open(f"{self.data_dir}/mails.json") as f:
            self.mails_list = json.load(f)

    def get_user_profile(self, iban: str) -> Dict:
        return self.users_dict.get(iban, {})

    def get_user_gps_history(self, biotag: str, days_back: int = 7) -> List[Dict]:
        # Simple filter - no caching
        return [loc for loc in self.locations_list if loc.get('biotag') == biotag]

    def get_user_communications(self, user_id: str) -> Dict:
        # Simple filter - no caching
        user_sms = [s for s in self.sms_list if user_id in str(s)]
        user_mails = [m for m in self.mails_list if user_id in str(m)]
        return {"sms": user_sms, "mails": user_mails}
```

**Tools (@tool functions):**

Per Transaction Analyzer:
- `get_user_transaction_history(user_id, days_back)`
- `get_recipient_profile(recipient_id)`
- `calculate_transaction_velocity(user_id, time_window_hours)`
- `query_fraud_memory(pattern_features)`

Per Behavioral Profiler:
- `get_user_communications(user_id, type="all")`
- `get_user_profile(iban)`
- `get_user_baseline(user_id)`
- `detect_phishing_patterns(communication_text)`

Per Geospatial Analyzer:
- `get_user_gps_history(biotag, days_back)`
- `calculate_distance(lat1, lng1, lat2, lng2)` (Haversine formula)
- `check_impossible_travel(biotag, txn_location, txn_time)`
- `get_user_residence(iban)`

### STEP 3: Agent Creation con Structured Output

Usare `ChatOpenAI.with_structured_output()` per forzare output Pydantic:

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

def get_llm(model="gpt-4", temp=0.1):
    return ChatOpenAI(
        model=model,
        temperature=temp,
        api_key=os.getenv("OPENAI_API_KEY")  # o OPENROUTER_API_KEY
    )

def create_transaction_analyzer_agent():
    llm = get_llm("gpt-4", temp=0.1)
    tools = [get_user_transaction_history, calculate_transaction_velocity, ...]

    agent = create_agent(
        model=llm.with_structured_output(TransactionAnalysisResult),
        system_prompt=TRANSACTION_ANALYZER_PROMPT,
        tools=tools
    )
    return agent
```

**Prompt Engineering per Agents:**

Ogni agent prompt deve:
1. Definire ruolo e obiettivo
2. Elencare fraud indicators specifici
3. Spiegare come usare i tools
4. Enfatizzare structured output requirement
5. Dare esempi di reasoning

### STEP 4: Memory & Adaptive Learning

**MemoryManager Class:**
```python
class MemoryManager:
    def __init__(self, memory_dir="./memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

    def load_fraud_patterns(self) -> List[FraudPattern]:
        # Load fraud_patterns.json

    def save_fraud_patterns(self, patterns: List[FraudPattern]):
        # Save to fraud_patterns.json

    def load_user_baselines(self) -> Dict[str, UserBehaviorBaseline]:
        # Load user_baselines.json

    def save_user_baselines(self, baselines: Dict):
        # Save to user_baselines.json

    def load_learning_state(self) -> AdaptiveLearningState:
        # Load learning_state.json (init if not exists)

    def save_learning_state(self, state: AdaptiveLearningState):
        # Save to learning_state.json
```

**Adaptive Learning Strategy (Cold Start):**
- Inizia senza baseline
- Per ogni utente visto, accumula statistiche on-the-fly:
  - Media/std importi transazioni
  - Orari tipici
  - Destinatari frequenti
  - Location tipiche
- Dopo N transazioni per utente, crea baseline dinamico
- Aggiorna baseline incrementalmente (exponential moving average)

**Pattern Discovery:**
- Quando 3+ frodi hanno feature simili → estrai pattern
- Pattern = {tipo, description, features, success_rate, level_discovered}
- Usa pattern in query_fraud_memory per matching

**Concept Drift Handling:**
- Pattern hanno timestamp di discovery
- Pattern vecchi (>2 levels) vengono pesati meno o rimossi
- Performance tracking: se pattern smette di funzionare → depreca

### STEP 5: Orchestrator con LLM-Based Reasoning

**Fraud Orchestrator Agent:**

Invece di weighted voting, usa LLM reasoning:

```python
FRAUD_ORCHESTRATOR_PROMPT = """You are the Fraud Orchestrator.

Your role:
- Receive structured analysis from 3 specialized agents
- Reason about the evidence holistically
- Make final fraud detection decision
- Explain your reasoning clearly

You will receive:
1. TransactionAnalysisResult from Transaction Analyzer
2. BehavioralAnomalyResult from Behavioral Profiler
3. GeospatialAnalysisResult from Geospatial Analyzer

Decision guidelines:
- Consider ALL evidence from all agents
- Weight higher confidence assessments more heavily
- Look for consensus: if 2+ agents flag high risk → strong signal
- Balance false positives vs false negatives (false negatives more costly)
- Use fraud pattern memory to enhance detection
- Provide clear reasoning for your decision

Decision threshold (adaptive):
- Start at: combined risk > 0.7 → fraudulent
- Or: 2+ agents HIGH/VERY_HIGH risk + combined risk > 0.6 → fraudulent

Output: FraudDecision with is_fraudulent, confidence, risk_score, primary_reasons
"""

def create_fraud_orchestrator_agent():
    llm = get_llm("gpt-4", temp=0.0)  # Zero temperature for consistency
    tools = [query_fraud_memory, store_fraud_pattern]

    agent = create_agent(
        model=llm.with_structured_output(FraudDecision),
        system_prompt=FRAUD_ORCHESTRATOR_PROMPT,
        tools=tools
    )
    return agent
```

**analyze_transaction() function:**
```python
def analyze_transaction(txn_id: str, txn_data: Dict, agents: Dict, memory: MemoryManager) -> FraudDecision:
    # 1. Invoke 3 agents sequentially (simple approach)
    txn_result = agents['transaction'].invoke({"transaction": txn_data})
    behav_result = agents['behavioral'].invoke({"transaction": txn_data, "user_id": txn_data['sender_id']})
    geo_result = agents['geospatial'].invoke({"transaction": txn_data, "user_id": txn_data['sender_id']})

    # 2. Package results for orchestrator
    orchestrator_input = {
        "transaction_id": txn_id,
        "transaction_analysis": txn_result.model_dump(),
        "behavioral_analysis": behav_result.model_dump(),
        "geospatial_analysis": geo_result.model_dump(),
        "fraud_patterns_memory": [p.model_dump() for p in memory.load_fraud_patterns()]
    }

    # 3. Orchestrator makes LLM-based decision
    decision = agents['orchestrator'].invoke(orchestrator_input)

    # 4. Update memory if fraud detected
    if decision.is_fraudulent:
        # Extract pattern features and store
        pattern = extract_pattern_from_decision(decision, txn_data)
        memory.add_fraud_pattern(pattern)

    # 5. Update user baseline
    memory.update_user_baseline(txn_data['sender_id'], txn_data)

    return decision
```

### STEP 6: Main Execution (process_level)

```python
def process_level(level_num: int, data_dir: str):
    """Process entire challenge level"""

    # Initialize
    data_manager = DataManager(data_dir)
    memory = MemoryManager()

    # Load memory from previous level
    learning_state = memory.load_learning_state()
    learning_state.current_level = level_num

    # Create agents with Langfuse callbacks
    agents = {
        'transaction': create_transaction_analyzer_agent(),
        'behavioral': create_behavioral_profiler_agent(),
        'geospatial': create_geospatial_analyzer_agent(),
        'orchestrator': create_fraud_orchestrator_agent()
    }

    # Load transactions
    transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")

    # Process each transaction
    fraud_decisions = []
    for idx, row in transactions_df.iterrows():
        txn_data = row.to_dict()

        decision = analyze_transaction(
            txn_id=row['transaction_id'],
            txn_data=txn_data,
            agents=agents,
            memory=memory,
            config={"callbacks": get_callbacks()}  # Langfuse
        )

        fraud_decisions.append(decision)

        # Progress
        if idx % 50 == 0:
            print(f"[Level {level_num}] Processed {idx}/{len(transactions_df)}")

    # Extract fraudulent transaction IDs
    fraudulent_ids = [d.transaction_id for d in fraud_decisions if d.is_fraudulent]

    # Validate output (15% < fraud_rate < 100%)
    fraud_rate = len(fraudulent_ids) / len(transactions_df)
    if fraud_rate < 0.15 or fraud_rate >= 1.0:
        print(f"WARNING: Fraud rate {fraud_rate:.2%} outside valid range!")

    # Save output
    output_file = f"level_{level_num}_output.txt"
    with open(output_file, 'w') as f:
        for txn_id in fraudulent_ids:
            f.write(f"{txn_id}\n")

    # Update and persist memory
    learning_state.total_transactions_analyzed += len(transactions_df)
    learning_state.last_adaptation_timestamp = datetime.now()
    memory.save_learning_state(learning_state)
    memory.save_fraud_patterns(memory.load_fraud_patterns())
    memory.save_user_baselines(memory.load_user_baselines())

    print(f"[Level {level_num}] Complete: {len(fraudulent_ids)} fraudulent transactions detected")
    print(f"[Level {level_num}] Output saved to: {output_file}")
```

## Configurazione LLM

### OpenRouter Setup
```python
# In .env file:
# LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=sk-or-...

def get_llm(model="openai/gpt-4", temp=0.1):
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openrouter":
        return ChatOpenAI(
            model=model,  # e.g. "openai/gpt-4", "anthropic/claude-3-sonnet"
            temperature=temp,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    else:
        return ChatOpenAI(
            model=model,  # e.g. "gpt-4", "gpt-3.5-turbo"
            temperature=temp,
            api_key=os.getenv("OPENAI_API_KEY")
        )
```

### Temperature Settings
- Transaction Analyzer: 0.1 (analisi quantitativa precisa)
- Behavioral Profiler: 0.2 (analisi testuale, slightly creative)
- Geospatial Analyzer: 0.1 (calcoli precisi)
- Fraud Orchestrator: 0.0 (decisioni consistenti, no randomness)

## Design Philosophy: Semplicità Prima

### Approccio Semplificato
- **NO parallel execution**: Agenti invocati sequenzialmente
- **NO batch processing**: Una transazione alla volta
- **NO caching**: Load all data all'inizio, accesso diretto
- **Sequential processing**: Semplice loop su transactions

### Ottimizzazioni Future (Post-MVP)
Queste ottimizzazioni saranno implementate in un secondo momento:
1. Parallel agent execution (asyncio)
2. Batch memory updates
3. Smart caching layer
4. Lazy loading strategies

### Costi (Approccio Semplice)
1. **Model Selection**: GPT-4 per tutti gli agenti (consistency)
2. **Context Window**: Include solo dati necessari per ogni tool call
3. **Structured Output**: Reduce parsing failures → less retries

### Adaptivity (Concept Drift)
1. **Pattern Decay**: Vecchi pattern pesati meno (exponential decay)
2. **Performance Monitoring**: Track pattern success rate per level
3. **Dynamic Thresholds**: Se troppi FP → raise threshold; se troppi FN → lower
4. **Incremental Baselines**: Update baselines con exponential moving average

## Output Format

**Per ogni level:**
File: `level_N_output.txt`
Format: Una riga per transaction ID fraudolenta
```
4a92ab00-8a27-4623-ab1d-56ac85fcd6b0
8830a720-ff34-4dce-a578-e5b8006b2976
...
```

**Validation:**
- Almeno 15% delle frodi correttamente identificate
- Non tutto fraudolento
- Non nulla fraudolento

## Testing & Debug

Poiché non facciamo validation interna:

1. **Dry Run su Training**: Test sistema su training data, check output format
2. **Log Decisions**: Langfuse tracing per ogni decision → debug reasoning
3. **Sanity Checks**:
   - Fraud rate in range ragionevole (20-40%)
   - High confidence decisions hanno reasoning solido
   - Pattern memory sta crescendo
4. **Manual Review**: Sample random decisions e verifica reasoning

## Sequenza di Esecuzione

```bash
# Setup
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...  # o OPENROUTER_API_KEY
export LANGFUSE_PUBLIC_KEY=pk-...
export LANGFUSE_SECRET_KEY=sk-...

# Run Level 1
python ai_orchestration.py --level 1 --data-dir "./The Truman Show_train/public"

# Output: level_1_output.txt

# Run Levels 2-5 (memoria persiste tra levels)
python ai_orchestration.py --level 2 --data-dir "./Level2/public"
...
```

## File Critici

1. **ai_orchestration.py** - File principale (tutti i componenti)
2. **memory/** - Directory per persistence JSON (creata automaticamente)
3. **The Truman Show_train/public/** - Data sources (read-only)
4. **.env** - Configurazione API keys
5. **requirements.txt** - Dependencies (già presente, verify langchain_openai)

## Metriche di Successo

- **Accuracy**: >85% detection rate su frodi reali
- **False Positive Rate**: <10% (minimize legitimate blocks)
- **Adaptivity**: Performance stabile o in crescita attraverso i 5 levels
- **Cost**: <$10 per level (optimize LLM calls)
- **Speed**: <2 minutes per 1000 transactions

## Risk Mitigation

1. **Single Submission Risk**: NO validation → DRY RUN MANDATORY prima di submit
2. **Concept Drift**: Pattern memory + adaptive thresholds
3. **False Positives**: LLM reasoning + high confidence threshold
4. **Cost Overrun**: Model selection + caching + batch processing
5. **Latency**: Parallel agents + lazy loading + caching

---

## Prossimi Passi per Implementation

1. Expand `ai_orchestration.py` con tutte le sezioni
2. Implementare Pydantic schemas (15+ models)
3. Implementare DataManager + Tools Layer
4. Implementare MemoryManager + Adaptive Learning
5. Implementare Agent Creation functions
6. Implementare analyze_transaction + process_level
7. Test dry run su training data
8. Submit level 1
9. Iterate per levels 2-5
