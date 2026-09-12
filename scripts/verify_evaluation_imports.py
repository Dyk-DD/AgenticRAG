# verify_evaluation_imports.py - Verify all evaluation module files import correctly
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, "app")

errors = []

def try_import(phase, name, code_str):
    try:
        exec(code_str)
        print(f"  [OK] {phase} - {name}")
    except Exception as e:
        msg = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"  [FAIL] {phase} - {name}: {msg}")
        errors.append((name, msg))

# Phase 1: Core data classes
try_import("Phase 1", "models.py",
    "from rag_modules.evaluation.models import EvalConfig, EvalMode, EvalStage, TestQuery, EvalSample, EvalReport")

# Phase 2: Dataset
try_import("Phase 2", "csv_dataset.py",
    "from rag_modules.evaluation.dataset.csv_dataset import CSVTestDataset")
try_import("Phase 2", "online_dataset.py",
    "from rag_modules.evaluation.dataset.online_dataset import OnlineDataset")

# Phase 3: Metrics
try_import("Phase 3", "retrieval_metrics",
    "from rag_modules.evaluation.metrics.retrieval_metrics import compute_retrieval_metrics")
try_import("Phase 3", "diversity_metrics",
    "from rag_modules.evaluation.metrics.diversity_metrics import compute_diversity_metrics")
try_import("Phase 3", "routing_metrics",
    "from rag_modules.evaluation.metrics.routing_metrics import compute_routing_metrics")
try_import("Phase 3", "generation_metrics",
    "from rag_modules.evaluation.metrics.generation_metrics import compute_generation_metrics")
try_import("Phase 3", "e2e_metrics",
    "from rag_modules.evaluation.metrics.e2e_metrics import compute_e2e_metrics")

# Phase 4: Judges
try_import("Phase 4", "relevance_judge",
    "from rag_modules.evaluation.judges.relevance_judge import RelevanceJudge")
try_import("Phase 4", "faithfulness_judge",
    "from rag_modules.evaluation.judges.faithfulness_judge import FaithfulnessJudge")
try_import("Phase 4", "hallucination_judge",
    "from rag_modules.evaluation.judges.hallucination_judge import HallucinationJudge")
try_import("Phase 4", "completeness_judge",
    "from rag_modules.evaluation.judges.completeness_judge import CompletenessJudge")

# Phase 5: Evaluator
try_import("Phase 5", "evaluator",
    "from rag_modules.evaluation.evaluator import Evaluator")

# Phase 6: Integration
try_import("Phase 6", "integration",
    "from rag_modules.evaluation.integration import EvalIntegration")

# Phase 7: Reporters
try_import("Phase 7", "json_reporter",
    "from rag_modules.evaluation.reporter.json_reporter import JSONReporter")
try_import("Phase 7", "console_reporter",
    "from rag_modules.evaluation.reporter.console_reporter import ConsoleReporter")
try_import("Phase 7", "viz_reporter",
    "from rag_modules.evaluation.reporter.viz_reporter import VizReporter")

# Final: __init__ exports
try_import("Final", "__init__ exports",
    "from rag_modules.evaluation import EvalConfig, EvalMode, EvalStage, EvalReport, TestQuery, EvalSample, Evaluator, EvalIntegration")

print()
if errors:
    print(f"ERRORS: {len(errors)} failures:")
    for name, msg in errors:
        print(f"  {name}: {msg}")
else:
    print("ALL 18 MODULE FILES IMPORTED SUCCESSFULLY")
