"""Analyze multi-agent flow for node ordering and missing handlers."""
from evaluation.utils import repo_files, read_file, save_json, parse_imports
from evaluation.config import ROOT


def run() -> dict:
    findings = []
    # Inspect multi_agent files
    files = [f for f in repo_files() if 'multi_agent' in str(f)]
    for f in files:
        text = read_file(f)
        if 'DiagnosticGraph' in text and 'initialize' not in text:
            findings.append({'file': str(f), 'issue': 'DiagnosticGraph missing initialize()'})

    result = {'files_checked': len(files), 'findings': findings}
    save_json('agent_flow', result)
    return result


if __name__ == '__main__':
    print(run())
