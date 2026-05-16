# Multi-Agent Medical Diagnostic System

## Project Overview

<<<<<<< Updated upstream
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
=======
This repository implements a production-style medical diagnostic system for diabetic complications, built using Streamlit, Supabase, a multi-agent orchestration graph, RAG memory retrieval, and a heuristic ML inference pipeline.

The application supports patient onboarding, symptom collection, neuropathy and oral health assessment, long-term memory retrieval, model-assisted prediction, fusion decision scoring, and structured report generation.

## Architecture Overview

The system follows a layered architecture that separates presentation, business logic, persistence, and AI orchestration.

### Layers

- **UI Layer**: pp.py provides the Streamlit presentation layer for patient selection, questionnaire flow, agent event streaming, and final report display.
- **Service Layer**: core/services/diagnostic_service.py orchestrates the main business logic, exposing a clean boundary between UI/agents and repository persistence.
- **Repository Layer**: core/repositories/* implements data access for patients, clinical assessments, ML predictions, RAG memory, and final decisions.
- **Database Layer**: core/database/client.py encapsulates Supabase client initialization and environment-based configuration.

### Data Flow

The normal data flow is:

1. UI captures patient input.
2. UI calls service methods in DiagnosticService.
3. Service methods call repository classes.
4. Repositories use core/database/client.py to access Supabase.

This ensures that database operations are centralized and there is no direct Supabase access from the UI or agents.

## Multi-Agent System

The multi-agent architecture is implemented in multi_agent/graph.py, multi_agent/agents.py, multi_agent/state.py, and multi_agent/memory.py.

### Agent Roles

- **PlannerAgent**: creates a diagnostic plan based on available patient data and memory.
- **MemoryRAGAgent**: retrieves long-term patient memory from the RAG store and loads episodic history.
- **ClinicalReasoningAgent**: decides the next action in a ReAct loop using tool descriptions and state.
- **ToolNode**: executes registered tool actions and writes observations back to shared state.
- **MLInferenceAgent**: loads stored ML predictions or computes a new neuropathy inference based on questionnaire answers.
- **FusionDecisionAgent**: computes the final diagnostic decision using weighted AI and clinical scores.
- **ReflectionAgent**: validates reasoning consistency and suggests whether to replan or continue.
- **ReportGeneratorAgent**: generates the final structured narrative report for the patient.

### Graph Execution Flow

The graph enforces a strict sequence of phases:

1. Planner → 2. Memory → 3. Questionnaire → 4. ML → 5. Fusion → 6. Reflection → 7. Report → END

The graph pauses whenever patient input is required and resumes once the answer is submitted.

## Features

- **Patient Management**: new patient registration and existing patient selection.
- **Diagnostic Workflow**: multi-step questionnaire spanning symptom scoring and clinical assessment.
- **ML Inference**: neuropathy prediction logic based on questionnaire-derived features.
- **RAG Memory**: long-term semantic retrieval of previous patient conversation memory.
- **Fusion Decision Engine**: weighted combination of AI output and clinical scores.
- **Final Report Generation**: structured patient-facing diagnostic summary.
- **CLI Alternative**: main.py provides a command-line interface for interaction.

## Database Schema Overview

The Supabase schema includes the following tables:

- patients
- 
ss_assessments
- 
ds_assessments
- gum_assessments
- ulcer_assessments
- ml_neuropathy_predictions
- inal_diagnostic_decisions
- conversation_memory
- knowledge_base

The system also defines a match_memory stored procedure for vector similarity retrieval.

## Setup Instructions

1. Install Python dependencies:

`ash
pip install -r requirements.txt
`

2. Create a .env file with the following values:

`ini
SUPABASE_URL=<your_supabase_url>
SUPABASE_KEY=<your_supabase_key>
OPENAI_API_KEY=<your_openai_api_key>
OPENAI_BASE_URL=<optional_openai_base_url>
`

3. Run the Streamlit application:

`ash
streamlit run app.py
`

4. For CLI interaction, run:

`ash
python main.py
`

## Project Structure

`	ext
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
`

## Architecture Rules

- No direct database access outside repository classes.
- UI code in pp.py and main.py is presentation and orchestration only.
- Agents invoke the service layer and tools, but do not interact with Supabase directly.
- core/database/client.py is the only centralized Supabase client initializer.
- DiagnosticService is the business boundary between UI/agents and persistence.

## Developer Guide

### Safe modification areas

- **pp.py**: update Streamlit UI and session presentation.
- **core/services/diagnostic_service.py**: update business orchestration and persistence coordination.
- **core/repositories/***: update Supabase data access logic and table mapping.
- **core/tools.py**: extend agent-accessible tools and observations.
- **multi_agent/***: modify agent behavior, graph routing, and state flow.
- **core/rag_engine.py**: update LLM prompts, embedding generation, and RAG retrieval logic.
- **core/questionnaire.py**: update questionnaire content and scoring logic.

### Where NOT to touch

- core/database/client.py: only change if Supabase connection configuration must be adjusted.
- database.py: legacy compatibility shim; avoid modifying it unless compatibility behavior changes.
- 	ests/test_db.py: repository connectivity validation only.
- generated or runtime artifacts, including __pycache__ directories.

## Notes

This README is generated from the current repository contents and reflects the existing system architecture, data flow, and agent orchestration.
>>>>>>> Stashed changes
