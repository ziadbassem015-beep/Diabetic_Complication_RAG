# Multi-Agent Medical Diagnostic System

## Project Overview

This repository implements a production-style medical diagnostic system for diabetic complications using:

- Streamlit
- Supabase
- Multi-Agent AI orchestration
- Retrieval-Augmented Generation (RAG)
- Heuristic ML inference

The platform supports:

- Patient onboarding
- Symptom collection
- Neuropathy and oral health assessment
- Long-term memory retrieval
- Model-assisted prediction
- Fusion decision scoring
- Structured diagnostic report generation

---

# System Architecture

The project follows a layered Clean Architecture design that separates:

- Presentation
- Business Logic
- Persistence
- AI Orchestration

This improves:

- Maintainability
- Scalability
- Testability
- Modularity

---

## Architecture Layers

### 1. UI Layer

`app.py`

Responsible for:

- Streamlit presentation
- Patient onboarding
- Questionnaire rendering
- Agent event streaming
- Final report visualization
- Session state management

The UI layer never directly accesses Supabase.

---

### 2. Service Layer

`core/services/diagnostic_service.py`

Acts as the business orchestration layer.

Responsibilities include:

- Loading patient context
- Coordinating repositories
- Running ML inference
- Managing fusion scoring
- Handling memory persistence/retrieval
- Encapsulating business rules

This layer serves as the boundary between the UI/Agents and the database layer.

---

### 3. Repository Layer

`core/repositories/*`

Implements isolated data access logic for:

- Patients
- Clinical assessments
- ML predictions
- Conversation memory
- Final diagnostic decisions

Repositories prevent direct database access from higher layers.

---

### 4. Database Layer

`core/database/client.py`

Centralized Supabase initialization and environment-based configuration.

Provides:

- Safe client initialization
- Singleton access pattern
- Environment validation
- Error-safe fallback handling

---

# Data Flow

The standard application flow is:

1. UI captures patient interaction
2. UI invokes `DiagnosticService`
3. Service layer calls repositories
4. Repositories communicate with Supabase
5. Agent graph updates shared state
6. Final report is generated and rendered

This ensures all persistence logic remains centralized and modular.

---

# Multi-Agent System

The multi-agent orchestration layer is implemented in:

- `multi_agent/graph.py`
- `multi_agent/agents.py`
- `multi_agent/state.py`
- `multi_agent/memory.py`

The system uses a shared state graph architecture inspired by LangGraph-style orchestration.

---

## Agent Roles

### PlannerAgent
Creates a diagnostic execution plan based on:

- Existing patient data
- Previous decisions
- Available memory context

---

### MemoryRAGAgent
Retrieves:

- Long-term semantic memory
- Episodic diagnostic history
- Relevant prior conversations

Uses vector similarity search via Supabase.

---

### ClinicalReasoningAgent
Core reasoning agent implementing a ReAct-style workflow.

Responsible for:

- Deciding next actions
- Asking questions
- Triggering tools
- Routing graph execution

---

### ToolNode
Executes registered tools from `core/tools.py`.

Examples:

- Retrieve clinical scores
- Save assessments
- Fetch ML predictions
- Query memory
- Ask patient questions

Observations are written back into shared graph state.

---

### MLInferenceAgent
Computes or retrieves neuropathy predictions using:

- Questionnaire-derived features
- NSS/NDS scoring
- Patient metadata

---

### FusionDecisionAgent
Combines:

- AI predictions
- Clinical scores
- ML outputs

Produces the final weighted diagnostic decision.

---

### ReflectionAgent
Performs self-validation checks on:

- Reasoning consistency
- Missing information
- Confidence alignment
- Diagnostic completeness

Can trigger replanning or additional reasoning loops.

---

### ReportGeneratorAgent
Generates the final structured diagnostic report.

Includes:

- Clinical findings
- ML interpretation
- Fusion decision
- Patient summary
- Recommendations

---

# Graph Execution Flow

The graph enforces a strict phase order:

```text
Planner
   ↓
Memory
   ↓
Questionnaire
   ↓
ML Inference
   ↓
Fusion Decision
   ↓
Reflection
   ↓
Report Generation
   ↓
END
The graph pauses whenever patient input is required and resumes after submission.
---

## Features
Patient Management
New patient registration
Existing patient retrieval
Session continuation
Diagnostic Workflow

Interactive multi-step questionnaire covering:

NSS assessment
NDS assessment
Gum health
Ulcer evaluation
Neuropathy risk indicators
ML Inference

Heuristic neuropathy prediction logic based on:

BMI
Age
HbA1c
Temperature sensitivity
NSS score
RAG Memory System

Supports:

Long-term memory persistence
Embedding storage
Semantic similarity retrieval
Historical patient context injection
Fusion Decision Engine

Combines:

AI reasoning
Clinical scores
ML predictions

Using weighted scoring logic.

Structured Report Generation

Produces patient-facing diagnostic summaries with:

Findings
Confidence
Recommendations
Clinical interpretation
CLI Alternative

main.py provides a command-line interface alternative to the Streamlit application.

Database Schema Overview

The Supabase schema includes:

patients
nss_assessments
nds_assessments
gum_assessments
ulcer_assessments
ml_neuropathy_predictions
final_diagnostic_decisions
conversation_memory
knowledge_base

The system also uses a match_memory stored procedure for vector similarity retrieval.

Setup Instructions
1. Install dependencies
pip install -r requirements.txt
2. Configure environment variables

Create a .env file:

SUPABASE_URL=<your_supabase_url>
SUPABASE_KEY=<your_supabase_key>

OPENAI_API_KEY=<your_openai_api_key>
OPENAI_BASE_URL=<optional_openai_base_url>
3. Run the Streamlit application
streamlit run app.py
4. Run CLI mode
python main.py
Project Structure
.
├── app.py
├── main.py
├── database.py
├── rag_engine.py
├── core
│   ├── database
│   │   ├── client.py
│   │   └── __init__.py
│   ├── repositories
│   │   ├── clinical_repo.py
│   │   ├── decision_repo.py
│   │   ├── memory_repo.py
│   │   ├── ml_repo.py
│   │   ├── patient_repo.py
│   │   └── __init__.py
│   ├── services
│   │   ├── diagnostic_service.py
│   │   └── __init__.py
│   ├── tools.py
│   ├── rag_engine.py
│   ├── questionnaire.py
│   └── __init__.py
├── multi_agent
│   ├── agents.py
│   ├── graph.py
│   ├── memory.py
│   ├── state.py
│   └── __init__.py
├── database
│   └── schema.sql
├── tests
│   └── test_db.py
├── requirements.txt
└── README.md
Architecture Rules
No direct database access outside repositories
No Supabase usage inside UI or agents
DiagnosticService is the business boundary
core/database/client.py is the only Supabase initializer
Agents communicate through tools and shared state only
Developer Guide
Safe Modification Areas
UI Layer
app.py
Streamlit rendering
Session presentation
Business Logic
core/services/diagnostic_service.py
Persistence
core/repositories/*
Agent Tooling
core/tools.py
Agent Logic
multi_agent/*
RAG + Prompting
core/rag_engine.py
Questionnaire + Scoring
core/questionnaire.py
Restricted Modification Areas

Avoid modifying unless necessary:

core/database/client.py
database.py
tests/test_db.py
Runtime/generated artifacts
__pycache__/
Design Principles
Why Multi-Agent Architecture?

The system separates responsibilities across specialized agents to improve:

Explainability
Modularity
Safety
Structured reasoning
Tool orchestration
Why Repository Pattern?

Repositories isolate persistence logic and:

Prevent database leakage
Simplify maintenance
Improve testing
Enable backend replacement
Why RAG + ML Hybrid Design?

The system combines:

Semantic memory retrieval (RAG)
Quantitative clinical scoring
ML-assisted inference
LLM reasoning

This hybrid approach improves contextual awareness and diagnostic robustness.

Limitations
LLM output quality depends on external APIs
Fusion scoring is heuristic and not medically validated
Scaling concurrent sessions may require async workers
Clinical deployment requires formal validation and compliance review
Future Improvements

Potential upgrades include:

Real ML model deployment
Async graph execution
Distributed memory storage
Clinical guideline retrieval
Advanced observability/logging
Agent evaluation pipelines
Authentication and RBAC
Multi-language support
Docker/Kubernetes deployment
---
