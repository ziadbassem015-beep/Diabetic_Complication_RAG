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
    validator_path = ROOT / "multi_agent" / "state_validator.py"
    tests_dir = ROOT / "tests"

    graph_text = read_file(graph_path) if graph_path.exists() else ""
    agents_text = read_file(fusion_agent_path) if fusion_agent_path.exists() else ""
    service_text = read_file(service_path) if service_path.exists() else ""

    if not graph_path.exists():
        findings.append({
            "issue": "multi_agent/graph.py missing",
            "severity": "CRITICAL",
            "recommendation": "Restore graph orchestrator.",
        })
        score -= 5
    else:
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

        required_log_events = (
            "questionnaire_start",
            "questionnaire_end",
            "ml_inference_done",
            "fusion_completed",
            "secondary_assessment_started",
        )
        for event in required_log_events:
            if f'"event": "{event}"' not in graph_text and f"'event': '{event}'" not in graph_text:
                findings.append({
                    "file": "multi_agent/graph.py",
                    "issue": f"Missing structured log event: {event}",
                    "severity": "MEDIUM",
                    "recommendation": f"Add logger.info with event={event}.",
                })
                score -= 0.5

    if service_path.exists():
        for event in ("gestational_saved", "heart_risk_saved"):
            if f'"event": "{event}"' not in service_text and f"'event': '{event}'" not in service_text:
                findings.append({
                    "file": "core/services/diagnostic_service.py",
                    "issue": f"Missing structured log event: {event}",
                    "severity": "MEDIUM",
                    "recommendation": f"Log {event} in run_secondary_assessments.",
                })
                score -= 0.5

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

    if not validator_path.exists():
        findings.append({
            "issue": "multi_agent/state_validator.py missing",
            "severity": "MEDIUM",
            "recommendation": "Add read-only state validation module.",
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
        fusion_class_start = agents_text.find("class FusionDecisionAgent")
        fusion_class_end = agents_text.find("class ReflectionAgent", fusion_class_start)
        fusion_block = agents_text[fusion_class_start:fusion_class_end]
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

        report_start = agents_text.find("class ReportGeneratorAgent")
        report_block = agents_text[report_start:] if report_start >= 0 else ""
        if "10." not in report_block and "Gestational Diabetes Screening" not in report_block:
            findings.append({
                "file": "multi_agent/agents.py",
                "issue": "Report missing section 10 (Gestational Diabetes)",
                "severity": "MEDIUM",
                "recommendation": "Add section 10 to ReportGeneratorAgent prompt.",
            })
            score -= 1
        if "11." not in report_block and "Heart Risk Assessment" not in report_block:
            findings.append({
                "file": "multi_agent/agents.py",
                "issue": "Report missing section 11 (Heart Risk)",
                "severity": "MEDIUM",
                "recommendation": "Add section 11 to ReportGeneratorAgent prompt.",
            })
            score -= 1
        if 'Gestational: Not applicable' not in report_block:
            findings.append({
                "file": "multi_agent/agents.py",
                "issue": "Report missing male gestational not-applicable text",
                "severity": "LOW",
                "recommendation": 'Use "Gestational: Not applicable" for male patients.',
            })
            score -= 0.5

    if service_path.exists():
        if "def run_secondary_assessments" not in service_text:
            findings.append({
                "file": "core/services/diagnostic_service.py",
                "issue": "run_secondary_assessments not implemented",
                "severity": "HIGH",
                "recommendation": "Add orchestration method for secondary pipelines.",
            })
            score -= 3
        if 'save_gestational_assessment' in service_text:
            run_sec_start = service_text.find("def run_secondary_assessments")
            run_sec_block = service_text[run_sec_start:run_sec_start + 1200] if run_sec_start >= 0 else ""
            if 'gender == "Female"' not in run_sec_block:
                findings.append({
                    "file": "core/services/diagnostic_service.py",
                    "issue": "Male patients may trigger gestational save",
                    "severity": "HIGH",
                    "recommendation": 'Gate gestational save on gender == "Female" only.',
                })
                score -= 2

    migration = ROOT / "database" / "supabase" / "migrations" / "20260517120000_add_secondary_assessments.sql"
    if not migration.exists():
        findings.append({
            "issue": "Secondary assessments migration file missing",
            "severity": "MEDIUM",
            "recommendation": "Add 20260517120000_add_secondary_assessments.sql migration.",
        })
        score -= 1

    required_tests = (
        "test_gender_gating.py",
        "test_secondary_flow.py",
        "test_pdn_regression.py",
        "test_e2e_diagnostic_flow.py",
        "test_state_validator.py",
    )
    for test_file in required_tests:
        if not (tests_dir / test_file).exists():
            findings.append({
                "issue": f"Missing test file: tests/{test_file}",
                "severity": "MEDIUM",
                "recommendation": f"Add tests/{test_file}.",
            })
            score -= 0.5

    score = max(0.0, min(10.0, score))
    result = {
        "score": round(score, 2),
        "findings": findings,
        "checks": {
            "secondary_node": "secondary_assessment_node" in graph_text,
            "gender_gating": 'gender == "Male"' in read_file(questionnaire_path) if questionnaire_path.exists() else False,
            "heart_risk_always": "save_heart_risk_assessment" in service_text,
            "structured_logs": "questionnaire_start" in graph_text and "heart_risk_saved" in service_text,
            "report_sections_10_11": "Gestational Diabetes Screening" in agents_text and "Heart Risk Assessment" in agents_text,
        },
    }
    save_json("assessment_coverage", result)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
