# test_evaluation_end_to_end.py - Integration test for the evaluation module
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, "app")
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import json
import time
from rag_modules.evaluation.models import (
    EvalConfig, EvalMode, EvalStage, EvalSample, EvalReport, TestQuery,
)
from rag_modules.evaluation.metrics.retrieval_metrics import compute_retrieval_metrics
from rag_modules.evaluation.metrics.diversity_metrics import compute_diversity_metrics
from rag_modules.evaluation.metrics.routing_metrics import compute_routing_metrics
from rag_modules.evaluation.metrics.generation_metrics import compute_generation_metrics
from rag_modules.evaluation.metrics.e2e_metrics import compute_e2e_metrics
from rag_modules.evaluation.reporter.json_reporter import JSONReporter
from rag_modules.evaluation.reporter.console_reporter import ConsoleReporter
from rag_modules.evaluation.reporter.viz_reporter import VizReporter

# --- Test data: mock EvalSamples ---
SAMPLE_SIZE = 20
samples = []

for i in range(SAMPLE_SIZE):
    query = TestQuery(
        query_id=f"test-{i:03d}",
        query_text=f"测试查询 {i} 相关内容",
        department="心血管科" if i < 10 else "呼吸科",
        ground_truth_node_ids=[f"node_{i}"],
        expected_routing_strategy="hybrid" if i % 2 == 0 else "graph_rag",
    )
    # Vary relevance flags to simulate different outcomes
    num_docs = 5
    if i < 5:
        # High relevance
        flags = [True, True, True, True, False]
    elif i < 10:
        # Medium relevance
        flags = [True, True, False, False, False]
    elif i < 15:
        # Low relevance
        flags = [True, False, False, False, False]
    else:
        # No relevance
        flags = [False, False, False, False, False]

    sample = EvalSample(
        test_query=query,
        strategy_used="hybrid" if i % 2 == 0 else "graph_rag",
        query_analysis={"intent": "diagnosis"},
        retrieved_docs=[],
        retrieval_latency_ms=150 + i * 10,
        generation_latency_ms=200 + i * 5,
        total_latency_ms=350 + i * 15,
        generated_answer=f"这是关于{query.query_text}的AI回答。",
        doc_relevance_flags=flags,
        faithfulness_score=max(0, 1.0 - i * 0.03),
        hallucination_score=min(1.0, i * 0.02),
        completeness_score=max(0, 1.0 - i * 0.02),
        answer_relevance_score=max(0, 1.0 - i * 0.015),
        error=None,
    )
    samples.append(sample)

# --- Compute metrics ---
config = EvalConfig(
    mode=EvalMode.OFFLINE,
    stages=[EvalStage.RETRIEVAL, EvalStage.ROUTING, EvalStage.GENERATION, EvalStage.END_TO_END],
    top_k_values=[1, 3, 5],
    strategies=["hybrid", "graph_rag", "combined", "router"],
)

print(f"Testing with {len(samples)} mock samples...")
print()

# Retrieval metrics
ret_metrics = compute_retrieval_metrics(samples, top_k_values=[1, 3, 5])
print("[TEST] Retrieval metrics computed")

# Diversity metrics
div_metrics = compute_diversity_metrics(samples, top_k=5)
ret_metrics.update(div_metrics)
print("[TEST] Diversity metrics computed")

# Routing metrics
route_metrics = compute_routing_metrics(samples)
print("[TEST] Routing metrics computed")

# Generation metrics
gen_metrics = compute_generation_metrics(samples)
print("[TEST] Generation metrics computed")

# E2E metrics
e2e_metrics = compute_e2e_metrics(samples)
print("[TEST] E2E metrics computed")

# Strategy breakdown
strategy_breakdown = {}
for strat in ["hybrid", "graph_rag", "combined", "router"]:
    strat_samples = [s for s in samples if s.strategy_used == strat]
    if strat_samples:
        hits = sum(1 for s in strat_samples if any(s.doc_relevance_flags))
        strategy_breakdown[strat] = {
            "count": len(strat_samples),
            "hit_count": hits,
            "hit_rate": hits / len(strat_samples),
            "avg_latency_ms": sum(s.retrieval_latency_ms for s in strat_samples) / len(strat_samples),
        }
print("[TEST] Strategy breakdown computed")

# Department breakdown
dept_breakdown = {}
for dept in set(s.test_query.department for s in samples):
    dept_s = [s for s in samples if s.test_query.department == dept]
    flags = [f for s in dept_s for f in s.doc_relevance_flags]
    hit_count = sum(any(s.doc_relevance_flags) for s in dept_s)
    dept_breakdown[dept] = {
        "count": len(dept_s),
        "hit_count": hit_count,
        "Recall@5": sum(flags) / len(flags) if flags else 0,
    }
print("[TEST] Department breakdown computed")

# Build report
report = EvalReport(
    config=config,
    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    system_info={"test": "verification"},
    retrieval_metrics=ret_metrics,
    routing_metrics=route_metrics,
    generation_metrics=gen_metrics,
    e2e_metrics=e2e_metrics,
    strategy_breakdown=strategy_breakdown,
    department_breakdown=dept_breakdown,
    samples=samples,
)

# --- Reporter tests ---
print()
print("=" * 50)
print("ConsoleReporter output:")
print("=" * 50)
ConsoleReporter.print_report(report, detailed=True)

print()
print("=" * 50)
print("JSONReporter output (serialization test):")
print("=" * 50)
json_data = JSONReporter.generate(report)
# Verify key structure
assert "meta" in json_data
assert "retrieval" in json_data
assert "routing" in json_data
assert "generation" in json_data
assert "end_to_end" in json_data
assert "samples" in json_data
json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
print(f"  JSON structure valid ({len(json_data['samples'])} samples)")
print(f"  Retrieval keys: {list(json_data['retrieval'].keys())}")
print(f"  Routing keys: {list(json_data['routing'].keys())}")
print(f"  Generation keys: {list(json_data['generation'].keys())}")

# --- VizReporter test ---
print()
print("=" * 50)
print("VizReporter output:")
print("=" * 50)
radar = VizReporter.generate_radar_data(report)
print(f"  Radar dimensions: {radar['dimensions']}")
print(f"  Radar values: {radar['metrics']['value']}")

comparison = VizReporter.generate_comparison_data(report)
print(f"  Comparison strategies: {comparison.get('strategies', [])}")

dept_chart = VizReporter.generate_department_chart(report)
print(f"  Department chart: {list(zip(dept_chart['departments'], dept_chart['recall_at_5']))}")

pie = VizReporter.generate_routing_pie(report)
print(f"  Routing pie: {pie}")

# --- Metric value assertions ---
print()
print("=" * 50)
print("Key metrics validation:")
print("=" * 50)
r = report.retrieval_metrics
print(f"  Recall@1 = {r.get('Recall@1', 0):.2%} (expected ~{5/20:.2%})")
print(f"  Recall@5 = {r.get('Recall@5', 0):.2%}")
print(f"  MRR      = {r.get('MRR', 0):.4f}")
print(f"  HitRate  = {r.get('HitRate', 0):.2%} (expected ~{15/20:.2%})")
assert r.get("Recall@1", 0) > 0
assert r.get("MRR", 0) > 0
assert r.get("HitRate", 0) > 0
print()
print("ALL TESTS PASSED")
