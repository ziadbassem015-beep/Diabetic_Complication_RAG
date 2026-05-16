# Duplicate Analysis Report

Summary:
- Total files scanned: 75
- Duplicate file pairs detected: 34
- Root cause: A full project copy was committed under `Diabetic_Complication_RAG/Diabetic_Complication_RAG` resulting in many byte-identical modules.

Key exact duplicates (representative):
- d:/Diabetic_Complication_RAG/core/database/client.py  <->  d:/Diabetic_Complication_RAG/Diabetic_Complication_RAG/core/database/client.py
  - Evidence: identical `get_supabase()` implementation.
- d:/Diabetic_Complication_RAG/core/repositories/clinical_repo.py  <->  d:/Diabetic_Complication_RAG/Diabetic_Complication_RAG/core/repositories/clinical_repo.py
  - Evidence: `class ClinicalRepository` defined at line 7 in both files with identical methods.
- d:/Diabetic_Complication_RAG/core/rag_engine.py  <->  d:/Diabetic_Complication_RAG/Diabetic_Complication_RAG/core/rag_engine.py
  - Evidence: `calculate_fusion_score` at line 55 and `run_diagnostic_pipeline` duplicated.
- d:/Diabetic_Complication_RAG/core/services/diagnostic_service.py  <->  d:/Diabetic_Complication_RAG/Diabetic_Complication_RAG/core/services/diagnostic_service.py
  - Evidence: `compute_fusion` defined at line 88 in both files.
- d:/Diabetic_Complication_RAG/core/tools.py  <->  d:/Diabetic_Complication_RAG/Diabetic_Complication_RAG/core/tools.py
  - Evidence: tooling registry and `tool_calculate_fusion_score` duplicated (line 115).

AST-level / semantic duplication findings:
- Repository classes (ClinicalRepository, MLRepository, DecisionRepository, etc.) use near-identical method structures and direct Supabase calls. This is repeated across many files and both subtree copies.
- Fusion score logic (calculation + thresholds) appears in at least three modules (`core/rag_engine.py`, `core/services/diagnostic_service.py`, `core/tools.py`) creating a risk of inconsistent behavior if one copy changes.

Risk assessment (short):
- High: duplicated subtree causes confusion, accidental imports, and increases maintenance burden.
- Medium: duplicated business logic (fusion score) risks drift and inconsistent diagnoses.
- Low: utility duplication (client wrapper) — but still unnecessary.

Recommendations (next steps):
1. Remove/archive the nested duplicate project at `Diabetic_Complication_RAG/Diabetic_Complication_RAG` after verifying Git history and confirming nothing unique is in that subtree.
2. Consolidate shared logic:
   - Extract a single fusion-score utility and call it from all callers.
   - Create a repository base class (or helper) to factor repeated Supabase access patterns.
3. Add CI checks: detect identical file hashes and warn on subtree duplicates.
4. Add a short `MAINTENANCE.md` describing canonical project path and developer guidelines to avoid re-commits of nested copies.

Files produced:
- Evaluation/outputs/duplicate_report.json
- Evaluation/outputs/duplicate_report.md

If you want, I can now:
- (A) Remove the duplicate subtree (I will only delete after your confirmation). 
- (B) Generate a full CSV of all duplicate pairs with checksums and exact line ranges for the duplicated regions.
- (C) Run an AST-based fingerprinting pass that outputs per-function similarity scores (more detailed semantic clusters).

Which next step do you want?"