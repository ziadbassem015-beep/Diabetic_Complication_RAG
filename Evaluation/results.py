"""Convenience loader for evaluation outputs.

Imports JSON results produced by the evaluation runner and exposes them as
module-level variables for quick access in scripts or notebooks.
"""
from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / 'outputs'


def _load(name: str):
    p = _OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        try:
            return json.loads(p.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            return None


MODULE_RESULTS = _load('module_results.json')
CATEGORY_SCORES = _load('category_scores.json')
FINAL_AUDIT = _load('final_audit.json')
RISK_SUMMARY = _load('risk_summary.json')

__all__ = ['MODULE_RESULTS', 'CATEGORY_SCORES', 'FINAL_AUDIT', 'RISK_SUMMARY']


if __name__ == '__main__':
    from pprint import pprint
    print('Module results:')
    pprint(MODULE_RESULTS)
    print('\nCategory scores:')
    pprint(CATEGORY_SCORES)
    print('\nFinal audit summary:')
    pprint(FINAL_AUDIT)
