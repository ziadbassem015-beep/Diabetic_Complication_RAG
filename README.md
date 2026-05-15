# Multi-Agent Medical Diagnostic System

## Project Overview

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
