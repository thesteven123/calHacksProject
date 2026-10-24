# Agent Workspace Overview & Role Allocation

This workspace is structured to support multi-agent collaboration across research and implementation:

## Roles

1. **Literature Review Agent (Antigravity)**:
   - Dedicated exclusively to research, paper analysis, state-of-the-art tracking, and adversarial critique.
   - Outputs maintained in `research/` (e.g. `research/papers.md`, `research/critiques/adversarial_review.md`).
   - Does **not** write implementation code.

2. **Architecture Design & Implementation Agent (Devin)**:
   - In charge of architectural design decisions and complete engineering implementation.
   - Owns implementation directories (such as `src/` for pipelines, model code, preprocessing, and training routines) and system architecture specifications in `docs/`.

## Folder Organization

- `research/`: Research papers, literature findings, benchmark comparisons, and adversarial red-teaming notes.
- `agent_roles/`: Reference profiles and guidelines for specialized research perspectives (Forensic Researcher, Adversarial Reviewer, Social Media Systems constraints).
- `docs/`: Reference plans, tooling suggestions, and future architectural design specifications (owned by Devin).
