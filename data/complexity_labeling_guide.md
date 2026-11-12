
# LLM Cost Autopilot — Complexity Labelling Guide

## Purpose

Assign each prompt one of three labels:
- simple
- medium
- complex

The label represents the estimated difficulty of completing
the task successfully, not merely the length of the prompt.

## Simple

A task requiring straightforward knowledge, retrieval,
calculation, formatting, or a small transformation.

Examples:
- Define a Python variable.
- Calculate 18 multiplied by 7.
- Return three country names as JSON.

## Medium

A task requiring several manageable reasoning steps,
comparison, summarisation, or structured processing.

Examples:
- Compare INNER JOIN and LEFT JOIN with examples.
- Summarise meeting notes into decisions and actions.
- Analyse a small sales table and identify trends.

## Complex

A task requiring substantial multi-step reasoning,
system design, difficult troubleshooting, or several
interacting technical or business constraints.

Examples:
- Diagnose intermittent failures in a distributed system.
- Design a secure, scalable AI architecture with cost limits.
- Develop a migration strategy with rollback and compliance.

## Labelling Rules

1. Judge the work required, not the number of words.
2. Do not classify a task based only on its first verb.
3. Consider the reasoning, dependencies and constraints.
4. A long prompt can still describe a simple task.
5. A short prompt can describe a complex task.
6. Flag ambiguous examples for manual review.
7. Keep evaluation examples separate from training examples.

## Dataset Quality Rules

- Use varied wording and sentence structures.
- Include short and long prompts in every category.
- Include different technical and nontechnical domains.
- Avoid repeated templates with only a topic changed.
- Include realistic customer requests.
- Review labels before training.
- Never train on reserved evaluation questions.
```
