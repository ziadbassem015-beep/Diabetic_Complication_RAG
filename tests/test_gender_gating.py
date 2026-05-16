"""Gender-based questionnaire filtering tests."""
from core.questionnaire import QUESTIONNAIRE, get_eligible_questions


def test_male_excludes_gestational_questions(male_patient):
    eligible = get_eligible_questions(QUESTIONNAIRE, male_patient)
    sections = {q["section"] for q in eligible}
    assert "GESTATIONAL" not in sections
    assert "HEART_RISK" in sections


def test_female_includes_gestational_questions(female_patient):
    eligible = get_eligible_questions(QUESTIONNAIRE, female_patient)
    sections = {q["section"] for q in eligible}
    assert "GESTATIONAL" in sections
    assert "HEART_RISK" in sections


def test_female_has_four_more_questions_than_male(male_patient, female_patient):
    male_q = get_eligible_questions(QUESTIONNAIRE, male_patient)
    female_q = get_eligible_questions(QUESTIONNAIRE, female_patient)
    assert len(female_q) - len(male_q) == 4


def test_graph_init_skips_gestational_for_male(male_patient):
    from unittest.mock import patch

    with patch("multi_agent.graph.HybridMemory"):
        from multi_agent.graph import DiagnosticGraph

        graph = DiagnosticGraph(male_patient)
        assert "gestational_diabetes" in graph.state.skipped_assessments
        assert all(q.get("section") != "GESTATIONAL" for q in graph.eligible_questions)
