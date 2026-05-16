"""Verify gestational diabetes and heart risk integration coverage."""
from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Any, Dict, List

EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT))

from config import ROOT, OUTPUT_DIR
from utils import read_file, save_json


def evaluate() -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    score = 10.0

    graph_path = ROOT / "multi_agent" / "graph.py"
    state_path = ROOT / "multi_agent" / "state.py"
    service_path = ROOT / "core" / "services" / "diagnostic_service.py"
    questionnaire_path = ROOT / "core" / "questionnaire.py"
    fusion_agent_path = ROOT / "multi_agent" / "agents.py"

    if not graph_path.exists():
        findings.append({
            "issue": "multi_agent/graph.py missing",
            "severity": "CRITICAL",
            "recommendation": "Restore graph orchestrator.",
        })
        score -= 5
    else:
        graph_text = read_file(graph_path)
        if "secondary_assessment_node" not in graph_text or "NODE_SECONDARY" not in graph_text:
            findings.append({
                "file": "multi_agent/graph.py",
                "issue": "secondary_assessment_node not integrated in graph",
                "severity": "HIGH",
                "recommendation": "Add NODE_SECONDARY and _run_secondary_assessment_node after fusion.",
            })
            score -= 3
        if "get_eligible_questions" not in graph_text:
            findings.append({
                "file": "multi_agent/graph.py",
                "issue": "Graph does not filter questionnaire via get_eligible_questions",
                "severity": "HIGH",
                "recommendation": "Use eligible_questions for all questionnaire iteration.",
            })
            score -= 2
        if "run_secondary_assessments" not in graph_text:
            findings.append({
                "file": "multi_agent/graph.py",
                "issue": "Graph does not call DiagnosticService.run_secondary_assessments",
                "severity": "HIGH",
                "recommendation": "Invoke service from secondary assessment node only.",
            })
            score -= 2

    if state_path.exists():
        state_text = read_file(state_path)
        for field in (
            "gestational_results",
            "heart_risk_results",
            "secondary_assessments_complete",
            "skipped_assessments",
        ):
            if field not in state_text:
                findings.append({
                    "file": "multi_agent/state.py",
                    "issue": f"MultiAgentState missing {field}",
                    "severity": "MEDIUM",
                    "recommendation": f"Add {field} to state dataclass.",
                })
                score -= 1

    if service_path.exists():
        service_text = read_file(service_path)
        if "def run_secondary_assessments" not in service_text:
            findings.append({
                "file": "core/services/diagnostic_service.py",
                "issue": "run_secondary_assessments not implemented",
                "severity": "HIGH",
                "recommendation": "Add orchestration method for secondary pipelines.",
            })
            score -= 3
        if 'gender == "Female"' not in service_text and "gender\") == \"Female\"" not in service_text:
            findings.append({
                "file": "core/services/diagnostic_service.py",
                "issue": "Female-only gestational guard may be missing",
                "severity": "MEDIUM",
                "recommendation": "Gate gestational save on gender == Female.",
            })
            score -= 1

    if questionnaire_path.exists():
        q_text = read_file(questionnaire_path)
        if 'gender == "Male"' not in q_text and "GESTATIONAL" in q_text:
            findings.append({
                "file": "core/questionnaire.py",
                "issue": "Male gender gating for gestational section not found",
                "severity": "HIGH",
                "recommendation": "Exclude GESTATIONAL section when patient gender is Male.",
            })
            score -= 2

    if fusion_agent_path.exists():
        fusion_text = read_file(fusion_agent_path)
        fusion_class_start = fusion_text.find("class FusionDecisionAgent")
        fusion_class_end = fusion_text.find("class ReflectionAgent", fusion_class_start)
        fusion_block = fusion_text[fusion_class_start:fusion_class_end]
        if "ml_gestational" in fusion_block or "heart_risk" in fusion_block:
            findings.append({
                "file": "multi_agent/agents.py",
                "issue": "FusionDecisionAgent may include secondary assessment logic",
                "severity": "HIGH",
                "recommendation": "Keep PDN fusion isolated; secondary logic belongs in graph node only.",
            })
            score -= 3
        if "compute_fusion" not in fusion_block:
            findings.append({
                "file": "multi_agent/agents.py",
                "issue": "FusionDecisionAgent may not call DiagnosticService.compute_fusion",
                "severity": "CRITICAL",
                "recommendation": "Restore PDN fusion via DiagnosticService.compute_fusion.",
            })
            score -= 4

    migration = ROOT / "database" / "supabase" / "migrations" / "20260517120000_add_secondary_assessments.sql"
    if not migration.exists():
        findings.append({
            "issue": "Secondary assessments migration file missing",
            "severity": "MEDIUM",
            "recommendation": "Add 20260517120000_add_secondary_assessments.sql migration.",
        })
        score -= 1

    score = max(0.0, min(10.0, score))
    result = {
        "score": round(score, 2),
        "findings": findings,
        "checks": {
            "secondary_node": "secondary_assessment_node" in read_file(graph_path) if graph_path.exists() else False,
            "gender_gating": 'gender == "Male"' in read_file(questionnaire_path) if questionnaire_path.exists() else False,
            "heart_risk_always": "save_heart_risk_assessment" in read_file(service_path) if service_path.exists() else False,
        },
    }
    save_json("assessment_coverage", result)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
