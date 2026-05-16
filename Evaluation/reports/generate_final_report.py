"""Run all evaluation modules and aggregate outputs into a final report."""
import importlib
from pathlib import Path
from evaluation.config import OUTPUT_DIR
import json


MODULES = [
    'evaluation.architecture.architecture_score',
    'evaluation.rag.retrieval_quality',
    'evaluation.agents.agent_flow_analysis',
    'evaluation.prompts.prompt_safety_analysis',
    'evaluation.safety.medical_safety_check',
    'evaluation.observability.telemetry_check',
]


def run_all():
    aggregated = {}
    for mod in MODULES:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, 'run'):
                aggregated[mod] = m.run()
            else:
                aggregated[mod] = {'error': 'no run()'}
        except Exception as e:
            aggregated[mod] = {'error': str(e)}

    # write aggregated JSON and a short markdown summary
    out_json = OUTPUT_DIR / 'final_audit.json'
    out_json.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding='utf-8')

    md = ['# Final Audit Report', '']
    for k, v in aggregated.items():
        md.append(f'## {k}')
        md.append('```json')
        md.append(json.dumps(v, indent=2, ensure_ascii=False))
        md.append('```')

    out_md = OUTPUT_DIR / 'final_audit.md'
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print('Wrote', out_json, out_md)


if __name__ == '__main__':
    run_all()
