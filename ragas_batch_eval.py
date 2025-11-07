#!/usr/bin/env python3
"""
Batch evaluator that runs RAGAS metrics for a dataset of
(question, answer, contexts) triples.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import ragas_evaluator


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load dataset entries from JSON or JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    entries: List[Dict[str, Any]] = []

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = data.get("samples", [])
            else:
                raise ValueError("Unsupported dataset format. Expect list or JSONL.")

    if not entries:
        raise ValueError("Dataset is empty; provide at least one sample.")
    return entries


def normalize_contexts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def evaluate_batch(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_question: List[Dict[str, Any]] = []
    aggregates: Dict[str, List[float]] = defaultdict(list)

    for idx, entry in enumerate(entries, start=1):
        question = entry.get("question") or entry.get("user_input") or ""
        answer = entry.get("answer") or entry.get("response") or ""
        contexts = (
            entry.get("contexts")
            or entry.get("retrieved_contexts")
            or entry.get("reference_contexts")
            or []
        )
        contexts = normalize_contexts(contexts)

        metrics = ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=contexts,
        )

        sample_id = entry.get("id") or f"sample_{idx}"
        per_question.append(
            {
                "id": sample_id,
                "question": question,
                "metrics": metrics,
            }
        )

        if "error" not in metrics:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    aggregates[metric_name].append(float(value))

    summary = {}
    for metric_name, values in aggregates.items():
        summary[metric_name] = {
            "count": len(values),
            "mean": mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {"per_question": per_question, "summary": summary}


def print_report(report: Dict[str, Any]) -> None:
    print("Per-question metrics:")
    for entry in report["per_question"]:
        print(f"- {entry['id']}:")
        metrics = entry["metrics"]
        if "error" in metrics:
            print(f"    ERROR: {metrics['error']}")
        else:
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")

    print("\nAggregate summary:")
    if not report["summary"]:
        print("  No successful evaluations to summarize.")
        return

    for metric_name, stats in report["summary"].items():
        print(
            f"  {metric_name}: mean={stats['mean']:.4f} "
            f"(n={stats['count']}, min={stats['min']:.4f}, max={stats['max']:.4f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluate RAG answers using RAGAS metrics."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSON or JSONL file containing question/answer/context entries.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the evaluation report as JSON.",
    )
    args = parser.parse_args()

    entries = load_dataset(Path(args.dataset))
    report = evaluate_batch(entries)
    print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved detailed report to {output_path}")


if __name__ == "__main__":
    main()
