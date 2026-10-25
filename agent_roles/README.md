# Agent Workspace Overview & Role Allocation

This workspace is structured to support multi-agent collaboration across research and implementation:

## Roles

1. **Git & GitHub Manager (`copilot-git-organizer`)**:
   - Exclusively owns and executes all git commands, branch creation, worktree setup, staging, commits, PRs, and merges.
   - Distinct from research and implementation: other agents never execute git commands directly.

2. **Web Search & Literature Review Agent (`Antigravity / Codex` in `calHacksProject` pane)**:
   - Dedicated on-demand web search, API documentation retrieval, and research agent for other agents in the workspace (primarily supporting `devin-impl`).
   - Retrieves package API references, Hugging Face dataset schemas/gating statuses, GitHub issue solutions, and academic paper details.
   - Outputs maintained in `research/` or returned directly via response briefs to Devin.
   - **Distinct boundary**: Does **not** perform git operations (owned by `copilot-git-organizer`) and does **not** write implementation code (owned by `devin-impl`).

3. **Architecture Design & Implementation Agent (`devin-impl`)**:
   - In charge of architectural design decisions and complete engineering implementation.
   - Owns implementation directories (`src/`, `scripts/`, `configs/`, `tests/`) and system architecture specifications in `docs/`.
   - Delegates web searches, library documentation lookups, and dataset verifications to the Web Search Agent.

## Folder Organization

- `research/`: Research papers, literature findings, benchmark comparisons, and adversarial red-teaming notes.
- `agent_roles/`: Reference profiles and guidelines for specialized research perspectives (Forensic Researcher, Adversarial Reviewer, Social Media Systems constraints).
- `docs/`: Reference plans, tooling suggestions, and future architectural design specifications (owned by Devin).
