"""Check for observability hooks (logging, metrics) presence."""
from evaluation.utils import repo_files, read_file, save_json


def run() -> dict:
    found = {'logging': 0, 'metrics': 0}
    for f in repo_files():
        text = read_file(f)
        if 'import logging' in text or 'logging.' in text:
            found['logging'] += 1
        if 'prometheus' in text or 'meter' in text:
            found['metrics'] += 1
    save_json('observability', found)
    return found


if __name__ == '__main__':
    print(run())
