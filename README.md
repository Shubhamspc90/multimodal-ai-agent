# Multimodal AI Agent

An industry-oriented multimodal Generative AI agent designed to understand text, images, documents, and voice; plan tasks; select approved tools; execute multi-step workflows; and verify results.

## Project Overview

The goal is to build more than a multimodal chatbot. The system will combine multimodal understanding, LLM-based reasoning, controlled tool execution, task planning, verification, and failure recovery in a testable software architecture.

### Example Use Case

A user uploads a chart and asks:

> Explain this chart, calculate the percentage increase, save the summary, and create a report.

The agent should:

1. Interpret the user's request and analyze the chart.
2. Identify the required steps and available tools.
3. Calculate the percentage increase using a suitable calculation tool.
4. Generate and save a summary and report using approved operations.
5. Verify the calculations and generated output.
6. Return the result with an execution trace and clearly report any failures.

## Project Goals

- Accept text, image, document, and voice inputs.
- Integrate configurable LLM, vision, and speech providers.
- Orchestrate multi-step tasks through a controlled agent workflow.
- Provide approved tools with explicit permissions.
- Verify important tool outputs and recover from eligible failures.
- Persist conversations and execution records.
- Evaluate task completion, correctness, latency, and cost.
- Follow professional software engineering, security, testing, and deployment practices.

## Technology Stack

| Component | Planned Technology |
|---|---|
| Language | Python |
| Backend API | FastAPI |
| Agent orchestration | LangGraph or a custom state machine |
| Language models | Configurable provider with tool-calling support |
| Image understanding | Vision-language model |
| Speech processing | Whisper or a compatible speech API |
| Computer vision | OpenCV and pretrained models when useful |
| Database | PostgreSQL |
| Semantic retrieval | pgvector, if justified by experiments |
| Frontend | React and TypeScript |
| Testing | pytest, API tests, integration tests, task-level evaluation |
| Code quality | Ruff and automated validation |
| Deployment | Docker and GitHub Actions |

The technology choices will be finalized through the requirements and architecture documents. Provider-specific dependencies will be added only when needed.

## Architecture

The planned system consists of:

- **API layer:** Validates requests and exposes application endpoints.
- **Multimodal processing:** Handles images, documents, and audio.
- **Model provider layer:** Separates model integrations from application logic.
- **Agent orchestration:** Manages task understanding, planning, execution, verification, and recovery.
- **Tool layer:** Provides explicitly approved operations such as calculation, document search, and report generation.
- **Memory and persistence:** Manages conversations, relevant context, and execution records.
- **Observability:** Captures structured logs, execution traces, errors, and performance metrics.
- **Evaluation:** Measures reliability using repeatable datasets and scenarios.

## Research Direction

**Working research question:** Can explicit planning, evidence verification, and failure-aware recovery improve multimodal task completion while reducing unsupported answers and unnecessary tool calls?

Planned experimental configurations:

1. Baseline multimodal LLM with tool calling.
2. Baseline with explicit task planning.
3. Planning combined with verification and failure recovery.

Evaluation metrics will include task completion rate, factual correctness, tool-selection accuracy, recovery rate, latency, and model/API cost.

Research novelty will not be claimed until a literature review and reproducible baseline comparisons support it.

## Development Roadmap

- **Phase 1 — Foundation:** Requirements, architecture, repository setup, virtual environment, API foundation, configuration, logging, and testing.
- **Phase 2 — Multimodal input:** Text, images, documents, speech transcription, and optional speech generation.
- **Phase 3 — Agent intelligence:** Planning, tool selection, controlled execution, state management, and verification.
- **Phase 4 — Reliability and research:** Failure recovery, evaluation datasets, baselines, metrics, and experiments.
- **Phase 5 — Product and deployment:** React interface, PostgreSQL persistence, Docker, CI/CD, observability, and final documentation.

Implementation will be incremental. Each feature must meet its acceptance criteria before the next dependent feature begins.

## Engineering Workflow

- Keep `main` stable.
- Implement features on dedicated branches.
- Use descriptive commit messages.
- Open pull requests and review changes before merging.
- Run tests, formatting/lint checks, and relevant validation before merging.
- Document significant architectural decisions.
- Preserve execution traces and repeatable evaluation results.

## Security Principles

- Store credentials in environment variables.
- Never commit real `.env` files, API keys, passwords, or private certificates.
- Validate uploaded files and enforce size limits.
- Restrict tool access through explicit allowlists and permissions.
- Require confirmation for consequential actions.
- Treat user-provided content and model-generated plans as untrusted input.
- Record tool outcomes and distinguish successful operations from failures.

## Getting Started

### Prerequisites

- Python version compatible with the project's pinned dependencies.
- Git.
- A code editor such as Visual Studio Code.
- A GitHub repository.
- PostgreSQL and model-provider credentials when those integrations are introduced.

### Local Setup

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` when application configuration is introduced. Populate it with your own local values; never commit `.env`.

The application run command and complete environment configuration will be documented when the initial FastAPI feature is implemented.

## Testing and Validation

Run the available tests:

```powershell
python -m pytest
```

Run code quality checks when configured:

```powershell
python -m ruff check .
python -m compileall backend
git diff --check
```

Only checks relevant to implemented features should be expected to pass at each milestone. CI will expand as the application grows.

## Repository Structure

```text
backend/       FastAPI application and agent backend
frontend/      React and TypeScript interface
evaluation/    Datasets, metrics, baselines, and experiments
docs/          Requirements, architecture, security, and research
scripts/       Development and evaluation utilities
infra/         Deployment and environment configuration
tests/         Unit, integration, API, and end-to-end tests
.github/       Pull request templates and CI workflows
```

Directories will be added incrementally as their features are implemented.

## Project Status

**Current stage:** Phase 1 — Design and Foundation.

The initial milestones are requirements specification, architecture design, repository setup, and development workflow. Multimodal integrations and agent execution will follow after the foundation is validated.

## License

The project license will be selected and added before public distribution.
