"""Search for prompt patterns and potential unsafe medical assertions in prompts."""
from evaluation.utils import repo_files, read_file, save_json


def run() -> dict:
    hits = []
    for f in repo_files():
        text = read_file(f)
        if 'diagnostic' in text.lower() and 'should' in text.lower():
            hits.append({'file': str(f)})
    result = {'hits': len(hits), 'details': hits}
    save_json('prompt_safety', result)
    return result


if __name__ == '__main__':
    print(run())
