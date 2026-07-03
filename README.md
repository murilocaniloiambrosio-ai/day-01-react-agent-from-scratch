# Day 1 — Building a ReAct AI Agent From Scratch

![Language](https://img.shields.io/badge/language-Python-3776AB)
![Day](https://img.shields.io/badge/day-1%20%2F%2030-00CFFF)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

## Overview

A minimal implementation of a **ReAct-style AI agent** (Reasoning + Acting), written from scratch in plain Python with zero external dependencies and zero agent frameworks. It reproduces the same control-flow loop that powers real tool-using LLM agents (Claude Code, LangChain agents, custom function-calling assistants, etc.), so the mechanics are transparent instead of hidden behind a library.

## What it demonstrates

- **The ReAct loop**: `Thought → Action → Action Input → Observation`, repeated until the model produces a `Final Answer`.
- **Tool dispatch**: the "model" only *decides* which tool to call and with what input; the surrounding code is always the one that actually executes it. This separation is the core safety/architecture property of every real agent system.
- **A mocked LLM** (`fake_llm`): a small deterministic, keyword-based function stands in for a real LLM API call. It produces text in the exact same format a real model would (`Thought:` / `Action:` / `Action Input:` / `Final Answer:`), so the parsing and loop logic below it is identical to what a production agent uses — only the "brain" is swapped out.

## Tools implemented

| Tool | Description |
|---|---|
| `calculator` | Evaluates a basic arithmetic expression |
| `weather` | Returns a fake weather report from an in-memory lookup table |
| `search` | Keyword search over a small in-memory knowledge base |

## How to run

```bash
python3 agent.py
```

No dependencies beyond the Python standard library. It runs 3 example questions and prints the full `Thought → Action → Observation` trace for each. See [`demo_output.txt`](./demo_output.txt) for a captured run.

## Concepts learned

- **ReAct pattern**: interleaving reasoning and acting instead of answering in a single shot, which lets an agent gather information it doesn't already have.
- **Tool dispatch as a control-flow boundary**: the model proposes, the code disposes — execution never happens inside the model itself.
- **Debugging a real retrieval bug**: the first version of the `search` tool matched the *entire* query as a substring against the knowledge base and silently failed on real questions (e.g. "What is ReAct?" never appears verbatim in any knowledge base entry). Fixed by splitting the query into keywords and matching any of them — a small, realistic example of the kind of bug that shows up in production retrieval/RAG systems.
- **Separation of concerns**: the loop (`run_agent`), the response parser (`parse_response`), the mocked model (`fake_llm`), and the tools (`TOOLS`) only communicate through plain text and dictionaries — the same shape as a real agent's control flow.

## Next steps

Day 2 swaps `fake_llm` for a real call to the Anthropic API, comparing how a real model's tool-selection reasoning differs from this keyword-based mock.
