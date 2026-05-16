"""Orchestrator: generate meta-analysis report from evaluation outputs."""
from pathlib import Path
import json
from Evaluation import results as res
from .analyze_scores import analyze_scores
from .weakness_detector import detect_weaknesses
from .risk_interpreter import interpret_risks
from .improvement_recommender import recommend_improvements
from .overengineering_detector import detect_overengineering
from .production_readiness import assess_production_readiness


OUT = Path(__file__).resolve().parents[1] / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)


def generate():
    category_scores = res.CATEGORY_SCORES or {}
    module_results = res.MODULE_RESULTS or []
    risk_summary = res.RISK_SUMMARY or []

    score_analysis = analyze_scores(category_scores)
    weaknesses = detect_weaknesses(risk_summary, module_results)
    interpreted = interpret_risks(weaknesses, {'anomalies': score_analysis.anomalies})
    recommendations = recommend_improvements(interpreted)
    over = detect_overengineering(category_scores, Path(__file__).resolve())
    readiness = assess_production_readiness(category_scores, interpreted)

    report = {
        'executive_summary': {
            'final_score_weighted': score_analysis.weighted,
            'final_score_mean': score_analysis.mean,
            'production_readiness': readiness,
        },
        'system_strengths': {
            'top_categories': sorted(category_scores.items(), key=lambda x: -x[1])[:3],
        },
        'critical_weaknesses': weaknesses,
        'interpreted_risks': interpreted,
        'overengineering': over,
        'recommendations': recommendations,
        'raw': {
            'category_scores': category_scores,
            'module_results': module_results,
            'risk_summary': risk_summary,
        }
    }

    # write JSON and markdown
    (OUT / 'meta_analysis_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    # Build markdown
    md_lines = []
    md_lines.append('# Meta-Analysis Technical Audit')
    md_lines.append('')
    md_lines.append('## Executive Summary')
    md_lines.append(f"- Weighted score: {score_analysis.weighted}/10")
    md_lines.append(f"- Mean score: {score_analysis.mean}/10")
    md_lines.append(f"- Production readiness: {readiness['classification']} ({readiness['score']}/10)")
    md_lines.append('')

    md_lines.append('## System Strengths')
    for k, v in report['system_strengths']['top_categories']:
        md_lines.append(f"- **{k}**: {v}/10")
    md_lines.append('')

    md_lines.append('## Critical Weaknesses')
    if weaknesses:
        for w in weaknesses[:20]:
            md_lines.append(f"- **{w.get('title')}** ({w.get('severity', 'MEDIUM')}): {w.get('description')} — files: {', '.join([f for f in (w.get('files') or []) if f])}")
    else:
        md_lines.append('- No critical weaknesses detected')
    md_lines.append('')

    md_lines.append('## Overengineering Analysis')
    if over.get('overengineered'):
        md_lines.append(f"- Likely overengineered: agents={over['agents_count']}, graph_complexity={over['graph_complexity']}")
        for r in over.get('reasons', []):
            md_lines.append(f"  - {r}")
    else:
        md_lines.append('- No clear overengineering detected')
    md_lines.append('')

    md_lines.append('## RAG Interpretation')
    rag_score = category_scores.get('rag')
    md_lines.append(f"- RAG score: {rag_score}/10")
    md_lines.append('- Review retrieval grounding and reranking; check memory contamination risks in `risk_summary.json`.')
    md_lines.append('')

    md_lines.append('## Agentic Workflow Interpretation')
    md_lines.append(f"- Agents score: {category_scores.get('agents')}/10")
    md_lines.append('- Check routing complexity and recursion in `multi_agent/graph.py` if orchestration issues appear.')
    md_lines.append('')

    md_lines.append('## Production Readiness Interpretation')
    md_lines.append(f"- Readiness classification: {readiness['classification']}")
    md_lines.append(f"- Readiness score: {readiness['score']}/10 (base {readiness['base']}, penalty {readiness['penalty']})")
    md_lines.append('')

    md_lines.append('## Medical AI Safety Interpretation')
    saf = category_scores.get('safety')
    md_lines.append(f"- Safety score: {saf}/10")
    md_lines.append('- Ensure emergency escalation and clinician-in-the-loop for high-risk suggestions.')
    md_lines.append('')

    md_lines.append('## Technical Debt Analysis')
    md_lines.append('- Review legacy shims and duplicate data access code reported in module outputs (see `module_results.json`).')
    md_lines.append('')

    md_lines.append('## Top Refactoring Priorities')
    for r in recommendations[:10]:
        md_lines.append(f"- {r['priority']}: {r['action']} — {r['rationale']}")
    md_lines.append('')

    md_lines.append('## Suggested Next Enhancements')
    md_lines.append('- Add reranking and grounding checks for RAG pipeline.')
    md_lines.append('- Add emergency escalation and confidence calibration for medical outputs.')
    md_lines.append('- Add monitoring and observability dashboards for agent flows.')
    md_lines.append('')

    md_lines.append('## Final Verdict')
    md_lines.append(f"- Classification: {readiness['classification']}")
    md_lines.append(f"- Score: {readiness['score']}/10")
    md_lines.append('')

    (OUT / 'meta_analysis_report.md').write_text('\n'.join(md_lines), encoding='utf-8')
    return report


if __name__ == '__main__':
    r = generate()
    print('Wrote meta-analysis reports to', OUT)
