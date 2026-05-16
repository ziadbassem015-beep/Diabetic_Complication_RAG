"""Simple retrieval quality checks for RAG components (read-only)."""
from pathlib import Path
from evaluation.utils import repo_files, read_file, save_json
from evaluation.config import ROOT


def run() -> dict:
    findings = []
    # Look for match_memory RPC usage
    for f in repo_files():
        text = read_file(f)
        if 'match_memory' in text:
            findings.append({'file': str(f), 'hint': 'Uses match_memory RPC'})

    result = {'matches_found': len(findings), 'details': findings}
    save_json('rag_retrieval', result)
    return result


if __name__ == '__main__':
    print(run())
