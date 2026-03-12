# TODO ALFONSO - Memory System + CLI Setup

**Ruolo**: Memory & Infrastructure - Persistence layer + CLI configuration
**Dipendenze**: Task 1 (già completato) - puoi iniziare SUBITO
**Priorità**: ALTA - MemoryManager serve per l'orchestrazione

---

## 📊 Progress

- **Completati**: 0/3
- **Da fare**: 3/3

---

## ⏳ Task Da Fare

### Task 8: Implementare MemoryManager class 🔥
**File**: `ai_orchestration.py`
**Dipende da**: Task 1 (Pydantic schemas - già fatto)
**Priorità**: 🔥 URGENTE - serve per Task 10 (orchestrazione)
**Tempo stimato**: 45-60 min

**Obiettivo**: Classe singleton per gestire persistence JSON dei fraud patterns, user baselines, e learning state

**Posizione nel file**: Dopo Section 5 (Tools), prima delle factory functions

---

#### Implementazione MemoryManager

```python
import json
from pathlib import Path
from typing import Dict, List, Optional

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

        # Initialize files if they don't exist
        self._initialize_files()

    @classmethod
    def get_instance(cls, memory_dir: str = "memory"):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(memory_dir)
        return cls._instance

    def _initialize_files(self):
        """Create initial JSON files if they don't exist."""
        # Fraud patterns
        if not self.fraud_patterns_file.exists():
            self._write_json(self.fraud_patterns_file, [])

        # User baselines
        if not self.user_baselines_file.exists():
            self._write_json(self.user_baselines_file, {})

        # Learning state
        if not self.learning_state_file.exists():
            from datetime import datetime
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

    def _read_json(self, file_path: Path) -> any:
        """Read JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _write_json(self, file_path: Path, data: any):
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

        # Check if pattern already exists (by pattern_id)
        existing_index = None
        for i, p in enumerate(patterns):
            if p.get("pattern_id") == pattern.pattern_id:
                existing_index = i
                break

        if existing_index is not None:
            # Update existing pattern
            patterns[existing_index] = pattern.dict()
        else:
            # Add new pattern
            patterns.append(pattern.dict())

        self._write_json(self.fraud_patterns_file, patterns)

    def query_patterns_by_type(self, pattern_type: str) -> List[FraudPattern]:
        """
        Query fraud patterns by type.

        Args:
            pattern_type: Type of pattern to filter

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

    def update_user_baseline(
        self,
        user_id: str,
        transaction: Dict
    ):
        """
        Update user baseline with new transaction data.

        Args:
            user_id: User IBAN
            transaction: Transaction data to incorporate
        """
        baseline = self.get_user_baseline(user_id)

        if baseline is None:
            # Create new baseline
            from datetime import datetime
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
            # Update existing baseline
            # TODO: Implement incremental update logic
            # - Update avg_transaction_amount (running average)
            # - Update std_transaction_amount
            # - Add to typical_hours, typical_recipients, typical_locations
            # - Increment transaction_count
            from datetime import datetime
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

    def update_learning_state(
        self,
        transactions_analyzed: int = 0,
        frauds_detected: int = 0
    ):
        """
        Update learning state with new statistics.

        Args:
            transactions_analyzed: Number of new transactions analyzed
            frauds_detected: Number of new frauds detected
        """
        state = self.load_learning_state()

        state.total_transactions_analyzed += transactions_analyzed
        state.total_frauds_detected += frauds_detected

        # Update counts from actual data
        patterns = self.load_fraud_patterns()
        baselines = self.load_user_baselines()

        state.fraud_patterns_count = len(patterns)
        state.user_baselines_count = len(baselines)

        from datetime import datetime
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
        self._initialize_files()
        print("Memory reset complete")
```

---

### Task 9: Creare directory memory/ con file JSON iniziali
**Dipende da**: Task 8 (MemoryManager)
**Priorità**: ALTA
**Tempo stimato**: 5-10 min

**Obiettivo**: Creare la struttura di directory e file JSON iniziali

**Steps**:

1. Crea directory `memory/`:
```bash
mkdir -p memory
```

2. Crea `memory/fraud_patterns.json`:
```json
[]
```

3. Crea `memory/user_baselines.json`:
```json
{}
```

4. Crea `memory/learning_state.json`:
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

5. Test che MemoryManager li carichi correttamente:
```python
# Test script
mem_mgr = MemoryManager.get_instance()
print(mem_mgr.get_statistics())
# Should print all zeros initially
```

---

### Task 12: Aggiungere CLI arguments parser
**File**: `ai_orchestration.py`
**Dipende da**: NIENTE - indipendente
**Priorità**: MEDIA (serve solo a fine pipeline)
**Tempo stimato**: 15-20 min

**Obiettivo**: Aggiungere argparse per gestire CLI arguments

**Implementazione**:

```python
import argparse

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
  python ai_orchestration.py --level 1 --data-dir ./The\ Truman\ Show_train

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


# Update main() function:
def main():
    """Main execution function with CLI support."""
    # Parse arguments
    args = parse_arguments()

    # Generate session ID
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

    # Process level
    process_level(
        level=args.level,
        data_dir=args.data_dir,
        memory_dir=args.memory_dir,
        session_id=session_id,
        output_file=args.output
    )

    print("\n[Complete] Processing finished successfully!")
```

**Posizione nel file**: Aggiungi `parse_arguments()` prima di `main()`, poi modifica `main()`

---

## ✅ Testing

### Test Task 8 (MemoryManager):

```python
# Test script
if __name__ == "__main__":
    from datetime import datetime

    # Initialize
    mem_mgr = MemoryManager.get_instance()

    # Test fraud pattern
    pattern = FraudPattern(
        pattern_id="test-pattern-1",
        pattern_type="phishing",
        description="Test phishing pattern",
        features={"urgency": True, "link": True},
        success_rate=0.85,
        occurrences=10,
        level_discovered=1,
        discovered_at=datetime.now().isoformat(),
        last_seen=datetime.now().isoformat()
    )
    mem_mgr.save_fraud_pattern(pattern)

    # Test user baseline
    baseline = UserBehaviorBaseline(
        user_id="IT60X0542811101000000123456",
        avg_transaction_amount=250.0,
        std_transaction_amount=50.0,
        typical_hours=[9, 10, 14, 17],
        typical_recipients=["IT60X0542811101000000654321"],
        typical_locations=["Rome"],
        transaction_count=25,
        last_updated=datetime.now().isoformat()
    )
    mem_mgr.save_user_baseline(baseline)

    # Test learning state update
    mem_mgr.update_learning_state(
        transactions_analyzed=10,
        frauds_detected=2
    )

    # Print statistics
    stats = mem_mgr.get_statistics()
    print(json.dumps(stats, indent=2))

    print("✅ All MemoryManager tests passed")
```

### Test Task 12 (CLI):

```bash
# Test help
python ai_orchestration.py --help

# Test with arguments
python ai_orchestration.py --level 1 --data-dir ./test_data

# Test reset memory
python ai_orchestration.py --level 1 --reset-memory

# Test connectivity only
python ai_orchestration.py --level 1 --test-connectivity
```

---

## 🔗 Dipendenze

**Da Nicola**:
- Task 10 (Orchestration) userà il tuo MemoryManager

**Da Laura**:
- Nessuna dipendenza diretta

**I tuoi task sbloccano**:
- Task 10 (Nicola può implementare orchestrazione)
- Task 11 (main loop può usare CLI args)

---

## 💡 Note Importanti

1. **MemoryManager è critico**: Testa bene tutti i metodi load/save

2. **JSON format**: Assicurati che i file JSON siano ben formattati (indent=2)

3. **Singleton pattern**: MemoryManager deve essere singleton per evitare conflitti

4. **Error handling**: Gestisci gracefully file mancanti o corrotti

5. **Incremental updates**: Il metodo `update_user_baseline()` può essere semplificato per ora (calcolo running average basic)

---

## 📋 Checklist

Task 8:
- [ ] Scritto MemoryManager class
- [ ] Implementati metodi load/save per fraud patterns
- [ ] Implementati metodi load/save per user baselines
- [ ] Implementati metodi load/save per learning state
- [ ] Testato singleton pattern
- [ ] Testato che file JSON vengano creati correttamente
- [ ] Testato metodo get_statistics()

Task 9:
- [ ] Creata directory `memory/`
- [ ] Creato `fraud_patterns.json`
- [ ] Creato `user_baselines.json`
- [ ] Creato `learning_state.json`
- [ ] Verificato che MemoryManager carichi i file

Task 12:
- [ ] Implementato parse_arguments()
- [ ] Aggiornato main() per usare CLI args
- [ ] Testato `--help`
- [ ] Testato `--level` e `--data-dir`
- [ ] Testato `--reset-memory`
- [ ] Testato `--test-connectivity`

---

**Prossimo task**: Inizia con Task 8 (MemoryManager) - è il più importante!
