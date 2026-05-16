#!/usr/bin/env python3
"""
run_full_system_check.py — Production runtime stability audit (no UI, no DB writes).

Usage:
    python scripts/run_full_system_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_agent.runtime_health import simulate_full_run


MALE_PATIENT = {
    "id": "system-check-male",
    "name": "System Check Male",
    "age": 45,
    "gender": "Male",
    "diabetes_duration": 5,
}

FEMALE_PATIENT = {
    "id": "system-check-female",
    "name": "System Check Female",
    "age": 30,
    "gender": "Female",
    "diabetes_duration": 2,
}


def main() -> int:
    print("=" * 60)
    print("  Diabetic Complication RAG — Runtime System Check")
    print("  Mode: in-memory | No Streamlit | No DB writes")
    print("=" * 60)

    results = []
    for label, patient in [("Scenario A (Male)", MALE_PATIENT), ("Scenario B (Female)", FEMALE_PATIENT)]:
        print(f"\n--- {label} ---")
        outcome = simulate_full_run(patient)
        results.append(outcome)
        status = "PASS" if outcome.passed else "FAIL"
        print(f"  Result: {status}")
        print(f"  Nodes executed: {len(outcome.nodes_executed)}")
        print(f"  Secondary ran: {outcome.secondary_ran}")
        print(f"  Report sections 10/11: {outcome.report_sections_ok}")
        print(f"  PDN isolated: {outcome.pdn_isolated}")
        if outcome.errors:
            print("  Errors:")
            for err in outcome.errors:
                print(f"    - {err}")

    all_passed = all(r.passed for r in results)
    print("\n" + "=" * 60)
    print(f"  SYSTEM STATUS: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)

    summary_path = ROOT / "Evaluation" / "outputs" / "runtime_system_check.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Summary written to: {summary_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
