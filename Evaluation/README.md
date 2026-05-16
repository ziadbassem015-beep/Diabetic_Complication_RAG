# Evaluation Framework

Automated evaluation framework for the Multi-Agent Medical Diagnostic System.

See individual folders for architecture, RAG, agent, prompt, safety, and observability analyses.

Usage:

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv_eval
.venv_eval\Scripts\activate
pip install -r requirements.txt
```

2. Run a single analysis, for example:

```bash
python -m evaluation.architecture.architecture_score
```

3. Generate the full audit report:

```bash
python -m evaluation.reports.generate_final_report
```

Outputs are saved in `outputs/`.
