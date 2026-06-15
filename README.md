# Truck Fleet Planning — LLM + Math Solver

A working example of what happens when you stop asking LLMs to do arithmetic and let them do what they are actually good at: understanding questions, structuring problems, and explaining results.

The math is handled by a MILP solver. The language is handled by agents. Neither does the other's job.

---

## The Core Idea

A fleet planner asks:

```
What is the best coverage within a $4.85M monthly budget?
```

The system does not guess. It:

1. **Classifies** the intent (what-if vs shock-response)
2. **Parses** the question into structured solver parameters via the OR Agent
3. **Solves** a Mixed Integer Linear Program with Pyomo + HiGHS
4. **Explains** the result in operational language via the Transportation Expert

The LLM never touches a number it did not receive from the solver. The solver never sees natural language.

```
User query
  → Intent Classifier (Haiku)
      │
      ├── what_if  → OR Agent → ScenarioParams
      │              → MILP Solver (Pyomo / HiGHS)
      │              → Lane-by-Lane + Weighted Cycle Time models
      │              → Transportation Expert → 2-sentence insight
      │
      └── shock_response → Shock Response Agent
                           → tests 3–5 mitigation strategies
                           → ranks by cost recovered
```

---

## Where Strands Comes In

Every agent in this project is built with [Strands Agents](https://strandsagents.com), an open-source framework from AWS for building tool-calling LLM agents in Python.

Strands handles the wiring between the LLM and the solver tools. Each agent is given a set of tools it is allowed to call, a system prompt that defines its role, and a structured output schema. The framework manages the tool-call loop and surfaces results via `result.structured_output`.

Three patterns from this project worth studying:

- **Stateless agent** — the OR Agent clears its message history before each call. Strands makes this trivial; the agent object exists but carries no session memory.
- **Autonomous agent** — the Shock Response Agent calls `run_milp_solver` multiple times in a single invocation, choosing strategies and comparing results without external orchestration. Strands manages the loop.
- **Tools as the contract** — `run_milp_solver`, `compare_coverage_costs`, and `load_network_data_tool` are plain Python functions decorated as Strands tools. The LLM calls them by name; Python executes them. No glue code needed.

---

## Why Three Models?

Every scenario produces a planning range, not a single number:

| Model | Role |
|---|---|
| Lane-by-Lane | Upper bound — dedicated trucks per route |
| Weighted Cycle Time | Middle estimate — shared fleet via demand-weighted average |
| MILP Solver | Lower bound — Pyomo/HiGHS optimized allocation |

This gives planners a defensible range to discuss, not a black-box answer to accept or reject.

---

## Example Questions

```text
What is the baseline fleet for the current operation?
What if payload drops to 28 tons and speed increases by 5%?
What is the new fleet if terminal TB is closed?
Minimize cost while serving at least 70% of collection points.
What is the best coverage within a $4.85M monthly budget?
What does it cost to increase coverage from 60% to 90%?
What is the best response if payload is reduced to 28 tons?
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Agent Framework | Strands Agents (AWS open-source) |
| LLM Provider | Anthropic (Claude) |
| Terminal UI | Rich |
| Optimization | Pyomo |
| Solver | HiGHS |
| Data | Pandas + Excel |
| Validation | Pydantic |

---

## Quick Start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="your-api-key"
python main.py
```

Inside the app:

```text
/language en
/onboarding
/baseline
/questions
```

Brazilian Portuguese is the default language. `/language en` switches to English.

---

## Project Layout

```text
.
├── main.py                  # Entry point
├── data/                    # Excel-backed network inputs
├── src/
│   ├── agents/              # OR Agent, Expert, Shock Response, Intent Classifier
│   ├── app/                 # CLI, pipeline, display, export, i18n
│   ├── domain/              # Data loading, dataclasses, capacity checks
│   └── models/              # Lane-by-Lane, Weighted Cycle Time, MILP solver

```

---

## Design Choices Worth Studying

**The OR Agent is stateless.** Each question is parsed independently. Parameters from previous scenarios never leak into the next query. This was a deliberate choice to keep the system predictable and the solver results trustworthy.

**The Expert has no tools.** The Transportation Expert receives only computed outputs and is not allowed to invent explanations not grounded in solver facts. Two sentences, 75-word maximum.

**Excel drives the network.** Changing the files in `data/` rewires the planning network — terminals, collection points, distances, demand, costs — without touching agent prompts or solver code.

---

## Testing

Deterministic regression checks (no API key required):

```powershell
python test_review_fixes.py
```

End-to-end integration (requires `ANTHROPIC_API_KEY`):

```powershell
python test_pipeline.py
```

---

## License

No license file is included. Add one before publishing if you want others to reuse or redistribute the code under clear terms.
