"""Run all evaluation modules, aggregate results, and generate reports.

Usage:
    python run_all_evaluations.py [--json-only] [--markdown-only] [--verbose] [--category CATEGORY]

This script dynamically discovers evaluation modules under the Evaluation folder,
executes their exported evaluation functions safely, aggregates findings, computes
heuristic scores, and writes JSON + Markdown reports to `Evaluation/outputs/`.
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
import traceback
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from config import OUTPUT_DIR, EVAL_ROOT, ROOT
from utils import repo_files, read_file, parse_imports

# Create a runtime shim so modules importing `evaluation.*` succeed even when
# this script is executed as a top-level file. This maps `evaluation.config`
# and `evaluation.utils` to the local modules.
import types
import sys
pkg = types.ModuleType('evaluation')
# mark as package by providing __path__
pkg.__path__ = [str(Path(__file__).resolve().parent)]
pkg.config = sys.modules.get('config')
pkg.utils = sys.modules.get('utils')
sys.modules['evaluation'] = pkg
# also register submodules so `import evaluation.utils` works during dynamic exec
sys.modules['evaluation.utils'] = sys.modules.get('utils')
sys.modules['evaluation.config'] = sys.modules.get('config')


CATEGORY_DIRS = ['architecture', 'rag', 'agents', 'prompts', 'safety', 'observability']


@dataclass
class Finding:
    title: str
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    affected_files: List[str]
    recommendation: str


@dataclass
class ModuleResult:
    name: str
    path: str
    category: str
    success: bool
    runtime_ms: float
    result: Any
    error: Optional[str] = None


SEVERITY_WEIGHTS = {
    'CRITICAL': 4.0,
    'HIGH': 2.0,
    'MEDIUM': 1.0,
    'LOW': 0.5,
}


def infer_severity(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ['critical', 'unsafe', 'fatal', 'exploit']):
        return 'CRITICAL'
    if any(k in t for k in ['high', 'error', 'breach', 'privilege']):
        return 'HIGH'
    if any(k in t for k in ['medium', 'warn', 'warning', 'issue']):
        return 'MEDIUM'
    return 'LOW'


def discover_modules(root: Path, categories: List[str]) -> List[Path]:
    files: List[Path] = []
    for cat in categories:
        folder = root / cat
        if not folder.exists():
            continue
        for p in folder.rglob('*.py'):
            if p.name == '__init__.py':
                continue
            files.append(p)
    return files


def load_module_from_path(path: Path, name_hint: str) -> Any:
    spec = importlib.util.spec_from_file_location(name_hint, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name_hint] = module
    loader = spec.loader
    if loader is None:
        raise ImportError(f'No loader for {path}')
    loader.exec_module(module)
    return module


def find_eval_function(module) -> Optional[str]:
    for candidate in ('evaluate', 'analyze', 'run', 'audit'):
        if hasattr(module, candidate):
            return candidate
    return None


def normalize_result(raw: Any, module_path: Path) -> Dict[str, Any]:
    # Try to convert common result shapes into a dict with findings/score
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, '__dict__'):
        return dict(raw.__dict__)
    return {'result': str(raw)}


def execute_module(path: Path, category: str, verbose: bool=False) -> ModuleResult:
    start = time.time()
    name_hint = f'eval_mod_{path.stem}_{int(start*1000)}'
    try:
        module = load_module_from_path(path, name_hint)
        func_name = find_eval_function(module)
        if func_name is None:
            runtime = (time.time() - start) * 1000
            return ModuleResult(name=name_hint, path=str(path), category=category, success=False, runtime_ms=runtime, result=None, error='no-eval-func')
        func = getattr(module, func_name)
        t0 = time.time()
        raw = func()
        runtime = (time.time() - start) * 1000
        norm = normalize_result(raw, path)
        return ModuleResult(name=name_hint, path=str(path), category=category, success=True, runtime_ms=runtime, result=norm)
    except Exception as e:
        runtime = (time.time() - start) * 1000
        tb = traceback.format_exc()
        if verbose:
            print(f'Error executing {path}:', tb)
        return ModuleResult(name=name_hint, path=str(path), category=category, success=False, runtime_ms=runtime, result=None, error=str(tb))


def extract_findings(module_results: List[ModuleResult]) -> List[Finding]:
    findings: List[Finding] = []
    for m in module_results:
        if not m.success:
            findings.append(Finding(
                title=f'Module Failure: {Path(m.path).stem}',
                description=(m.error or 'module failed to run'),
                severity='HIGH',
                affected_files=[m.path],
                recommendation='Fix the evaluation module or ensure it exposes an evaluation function.'
            ))
            continue
        res = m.result or {}
        # Look for common keys
        for key in ('findings', 'issues', 'hits', 'details'):
            if key in res and isinstance(res[key], (list, tuple)):
                for item in res[key]:
                    if isinstance(item, dict):
                        title = item.get('title') or item.get('issue') or item.get('file') or 'Finding'
                        desc = item.get('description') or item.get('hint') or json.dumps(item)
                        sev = item.get('severity') or infer_severity(desc)
                        affected = item.get('file') or item.get('affected_files') or [m.path]
                        rec = item.get('recommendation') or 'Review and remediate.'
                    else:
                        title = str(item)
                        desc = str(item)
                        sev = infer_severity(desc)
                        affected = [m.path]
                        rec = 'Review and remediate.'
                    findings.append(Finding(title=title, description=desc, severity=sev, affected_files=affected if isinstance(affected, list) else [affected], recommendation=rec))
        # If result contains counts or score fields, optionally create a summary finding
        if 'score' in res:
            findings.append(Finding(title='Module Score', description=f"Score reported: {res['score']}", severity='LOW', affected_files=[m.path], recommendation='Score is informational.'))
    return findings


def compute_category_scores(findings: List[Finding]) -> Dict[str, float]:
    # Group by categories via affected_files path segments
    cat_scores: Dict[str, float] = {c: 10.0 for c in CATEGORY_DIRS}
    # Penalize based on severity counts in each category
    for f in findings:
        # infer category from first affected file
        aff = f.affected_files[0] if f.affected_files else ''
        cat = 'unknown'
        for c in CATEGORY_DIRS:
            if f'/{c}/' in aff.replace('\\', '/') or f'\\{c}\\' in aff:
                cat = c
                break
        weight = SEVERITY_WEIGHTS.get(f.severity.upper(), 0.5)
        if cat in cat_scores:
            cat_scores[cat] = max(0.0, cat_scores[cat] - weight)
    # Round and clamp
    for k in list(cat_scores.keys()):
        cat_scores[k] = round(max(0.0, min(10.0, cat_scores[k])), 2)
    return cat_scores


def generate_markdown(aggregated: Dict[str, Any], findings: List[Finding], category_scores: Dict[str, float], final_score: float) -> str:
    lines: List[str] = []
    lines.append('# Executive Summary')
    lines.append('')
    lines.append(f'Final Production Readiness Score: **{final_score:.2f}/10**')
    lines.append('')

    # Sections per required headings
    def add_section(title: str, content: List[str]):
        lines.append(f'## {title}')
        lines.extend(content)
        lines.append('')

    # Architecture Analysis
    arch = aggregated.get('architecture', {})
    add_section('Architecture Analysis', [f"- Score: {category_scores.get('architecture', 'N/A')}/10", f"- Findings: {len([f for f in findings if '/architecture/' in '/'.join(f.affected_files)])}"])  # noqa: E501

    # RAG Analysis
    add_section('RAG Analysis', [f"- Score: {category_scores.get('rag', 'N/A')}/10"])  # minimal

    # Agent Analysis
    add_section('Agent Analysis', [f"- Score: {category_scores.get('agents', 'N/A')}/10"])  # minimal

    # Prompt Safety
    add_section('Prompt Safety', [f"- Score: {category_scores.get('prompts', 'N/A')}/10"])  # minimal

    # Medical Safety
    add_section('Medical Safety', [f"- Score: {category_scores.get('safety', 'N/A')}/10"])  # minimal

    # Observability
    add_section('Observability', [f"- Score: {category_scores.get('observability', 'N/A')}/10"])  # minimal

    # Top Critical Risks
    criticals = [f for f in findings if f.severity == 'CRITICAL']
    lines.append('## Top Critical Risks')
    if criticals:
        for c in criticals:
            lines.append(f"- **{c.title}**: {c.description} (files: {', '.join(c.affected_files)})")
    else:
        lines.append('- None found')
    lines.append('')

    # Refactoring Priorities
    lines.append('## Refactoring Priorities')
    # Prioritize HIGH and CRITICAL
    prios = [f for f in findings if f.severity in ('CRITICAL', 'HIGH')]
    if prios:
        for p in prios:
            lines.append(f"- {p.title} ({p.severity}): {p.recommendation}")
    else:
        lines.append('- No high priority refactors identified')
    lines.append('')

    lines.append('## Final Notes')
    lines.append('This automated audit is heuristic. Review findings manually before making production changes.')
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-only', action='store_true')
    ap.add_argument('--markdown-only', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--category', type=str, help='Run only a specific category')
    args = ap.parse_args(argv)

    categories = CATEGORY_DIRS if not args.category else [args.category]
    modules = discover_modules(Path(__file__).resolve().parent, categories)
    results: List[ModuleResult] = []
    for p in modules:
        cat = next((c for c in categories if f'/{c}/' in str(p).replace('\\', '/')), 'unknown')
        if args.verbose:
            print('Executing', p)
        res = execute_module(p, cat, verbose=args.verbose)
        results.append(res)

    # Persist raw module results
    raw_out = OUTPUT_DIR / 'module_results.json'
    raw_out.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding='utf-8')

    findings = extract_findings(results)

    # Compute category scores
    category_scores = compute_category_scores(findings)

    # Compute final production readiness as mean of category scores
    vals = [v for v in category_scores.values()]
    final_score = float(sum(vals) / len(vals)) if vals else 0.0

    # Aggregated data
    aggregated = {
        'module_count': len(results),
        'categories': category_scores,
        'final_score': final_score,
    }

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'final_audit.json').write_text(json.dumps({'aggregated': aggregated, 'findings': [asdict(f) for f in findings]}, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUTPUT_DIR / 'risk_summary.json').write_text(json.dumps([asdict(f) for f in findings], indent=2, ensure_ascii=False), encoding='utf-8')
    (OUTPUT_DIR / 'category_scores.json').write_text(json.dumps(category_scores, indent=2, ensure_ascii=False), encoding='utf-8')

    if not args.json_only:
        md = generate_markdown(aggregated, findings, category_scores, final_score)
        (OUTPUT_DIR / 'final_audit.md').write_text(md, encoding='utf-8')

    if not args.markdown_only:
        print('Wrote outputs to', OUTPUT_DIR)


if __name__ == '__main__':
    main()
