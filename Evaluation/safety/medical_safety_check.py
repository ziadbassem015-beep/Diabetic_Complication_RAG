"""Basic static safety checks for medical content.

This script performs read-only pattern checks; it's not a substitute for clinical review.
"""
from evaluation.utils import repo_files, read_file, save_json


def run() -> dict:
    issues = []
    for f in repo_files():
        text = read_file(f).lower()
        if 'diagnose' in text and 'not a doctor' not in text:
            issues.append({'file': str(f), 'issue': 'uses word diagnose without clinician-disclaimer'})
    result = {'issues': len(issues), 'details': issues}
    save_json('medical_safety', result)
    return result


if __name__ == '__main__':
    print(run())
