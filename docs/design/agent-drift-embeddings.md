# Agent Drift Detection via Embeddings

**Status:** Design / Brainstorm
**Date:** 2026-03-09
**Problem:** Agents executing multi-step plans can silently drift from their
original task goal — either tactically (distracted mid-step) or strategically
(plan itself diverges). There is no mechanism to detect or measure this drift.

---

## Problem Statement

When an agent works through a plan with multiple steps, two failure modes occur:

1. **Tactical drift** — The agent is mid-step ("Add auth middleware") but starts
   doing unrelated work (refactoring CSS). Each action looks reasonable in
   isolation but deviates from the current step's intent.

2. **Strategic drift** — Each step is completed diligently, but the aggregate
   trajectory of completed steps diverges from the original task. This is the
   "boiling frog" problem — no single step is obviously wrong, but the overall
   direction is off.

### Why naive approaches fail

- **Keyword matching** is too brittle — "reading database schema" doesn't share
  keywords with "add migration" but is clearly on-task for that step.
- **Global similarity** (comparing everything to the original task) produces
  false positives during legitimate sub-task exploration.
- **Fixed time windows** don't align with the agent's actual work structure.

---

## Design: Plan-Coupled Embedding Drift Detection

### Core Insight

Tie drift measurement to the **plan step lifecycle**. Each time a step is marked
`in_progress`, reset the drift tracker and use the step description as the
scoped goal. Measure drift only within that epoch. On step completion, freeze
metrics and optionally check inter-step strategic drift.

This eliminates the biggest false-positive source: legitimate exploration that
diverges from the *overall* goal but is on-task for the *current step*.

### Two-Layer Detection

| Layer | Compares | Catches |
|-------|----------|---------|
| **Intra-step** | Agent actions ↔ current step description | Tactical drift (distracted mid-step) |
| **Inter-step** | Completed steps aggregate ↔ overall goal | Strategic drift (plan itself is off-track) |

---

## Intra-Step Drift Detection

### Lifecycle

```
Plan Step: "Add authentication middleware to Express routes"
  │
  ├─ step marked in_progress → reset drift tracker
  │   step_goal_embedding = embed(step.description)
  │
  ├─ turn 1: agent reads existing routes        → similarity 0.82 ✓
  ├─ turn 2: agent reads auth library docs       → similarity 0.71 ✓
  ├─ turn 3: agent starts refactoring CSS        → similarity 0.35 ✗ DRIFT
  │   → intervention (re-grounding nudge, escalation, etc.)
  │
  └─ step marked completed → freeze metrics, move to next step
```

### What to Embed Per Turn

Raw tool calls + results are verbose and noisy. Options (in order of preference):

1. **Summarize-then-embed** — Use a cheap/fast model to produce a one-sentence
   turn summary, then embed the summary. Best signal, small LLM cost per turn.
2. **Agent text only** — Embed only the agent's reasoning/text output, skip raw
   tool results. Zero extra LLM cost, decent signal.
3. **Tool names + snippet** — Concatenate tool names and a truncated snippet of
   results. Lightweight, no LLM call, but noisier embeddings.

### Similarity Metric

Cosine similarity between the step goal embedding and the per-turn embedding.

```python
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))
```

### Threshold Strategy

Hard-coded thresholds are brittle. Preferred approach is **adaptive baseline**:

1. Compute similarity for the first 2–3 on-task turns after step start.
2. Establish a baseline mean and standard deviation.
3. Flag when similarity drops more than N standard deviations below the baseline.

Fallback: configurable static thresholds with sensible defaults.

```python
@dataclass
class DriftConfig:
    soft_threshold: float = 0.50    # inject re-grounding nudge
    hard_threshold: float = 0.30    # escalate / pause
    adaptive: bool = True           # use adaptive baseline
    baseline_turns: int = 3         # turns to establish baseline
    deviation_factor: float = 2.0   # std deviations for adaptive flag
    trajectory_window: int = 3      # consecutive declining scores to flag
```

### Trajectory-Based Detection

A single low score may be noise. Track the similarity curve over time:

- **Monotonic decline** over `trajectory_window` consecutive turns is a stronger
  drift signal than a single low score.
- This distinguishes "brief tangent that recovers" from "progressively losing
  the plot."

---

## Inter-Step (Strategic) Drift Detection

On each step completion, check whether the accumulated completed work is
converging toward the overall goal:

```python
overall_goal_embedding = embed(original_task)

def on_step_completed(completed_steps):
    summary = summarize(completed_steps)
    progress_embedding = embed(summary)
    strategic_score = cosine_similarity(overall_goal_embedding, progress_embedding)

    if strategic_score declining over last 3 completed steps:
        flag: "Plan may be diverging from original goal"
        suggest plan revision
```

This catches the case where an agent completes steps like:
1. "Set up database schema" → on track
2. "Configure logging framework" → slightly off
3. "Refactor CI pipeline" → clearly diverging

Each step was executed well, but the plan drifted.

---

## Architecture: Enrichment Plugin

The drift monitor is an **enrichment plugin** (`PLUGIN_KIND = "enrichment"`). It
observes the turn loop and plan lifecycle without providing tools to the model.

### Integration with Todo Plugin

```
todo plugin (existing)          drift_monitor plugin (new)
─────────────────               ──────────────────────────
step → in_progress  ─────────►  on_step_started(step_desc)
                                  step_goal_emb = embed(step_desc)
                                  drift_history = []
                                  baseline_scores = []

each turn output    ─────────►  on_turn_completed(actions, output)
                                  current_emb = embed(turn_summary)
                                  score = cosine(step_goal_emb, current_emb)
                                  drift_history.append(score)
                                  evaluate_drift(score)

step → completed    ─────────►  on_step_completed(step_desc, output)
                                  record_step_metrics()
                                  check_strategic_drift()
```

### Observation-Only Scope

This plugin is **observation and measurement only**. It:

- Computes and records drift scores per turn and per step.
- Emits drift events/metrics for external consumers.
- Exposes drift data for other plugins to act on.

It does **not** perform steering or intervention. Steering (re-grounding nudges,
plan revision, escalation) is handled by a separate steering plugin that
consumes drift signals. This separation keeps the drift monitor reusable and
testable independently.

### Plugin Skeleton

```python
"""Agent drift detection via plan-coupled embeddings."""

from dataclasses import dataclass, field
from shared.plugins.base import EnrichmentPlugin

PLUGIN_KIND = "enrichment"


@dataclass
class StepDriftState:
    """Drift tracking state for a single plan step epoch.

    Created when a step transitions to in_progress. Accumulates
    per-turn similarity scores and computes baseline statistics
    from the first N turns.
    """
    step_description: str
    step_goal_embedding: list[float] | None = None
    turn_scores: list[float] = field(default_factory=list)
    baseline_mean: float | None = None
    baseline_std: float | None = None


@dataclass
class DriftMetrics:
    """Aggregate drift metrics for a completed step.

    Frozen on step completion and appended to the inter-step
    strategic drift history.
    """
    step_description: str
    turn_count: int
    mean_similarity: float
    min_similarity: float
    drift_flags: int  # number of times drift was flagged


class AgentDriftMonitor(EnrichmentPlugin):
    """Enrichment plugin that measures agent drift relative to plan steps.

    Lifecycle:
    - on_step_started: resets tracker, embeds step description as goal
    - on_turn_completed: embeds turn summary, computes similarity, records
    - on_step_completed: freezes step metrics, checks strategic drift

    This plugin is observation-only. It computes and exposes drift scores
    but does not intervene. Steering is delegated to a separate plugin
    that consumes drift signals.
    """
    PLUGIN_NAME = "drift_monitor"

    def __init__(self, config: "DriftConfig"):
        self.config = config
        self.current_step: StepDriftState | None = None
        self.completed_step_metrics: list[DriftMetrics] = []
        self.overall_goal_embedding: list[float] | None = None
        self._embed_fn = None  # injected embedding callable

    # -- step lifecycle hooks --

    def on_step_started(self, step_description: str) -> None:
        """Reset drift tracker for new step epoch."""
        ...

    def on_turn_completed(self, turn_summary: str) -> float:
        """Embed turn, compute similarity, return score."""
        ...

    def on_step_completed(self) -> DriftMetrics:
        """Freeze metrics for completed step, check strategic drift."""
        ...

    # -- strategic drift --

    def check_strategic_drift(self) -> float | None:
        """Compare completed-steps aggregate to overall goal."""
        ...
```

### Embedding Source

The plugin needs an embedding function. Options:

| Source | Pros | Cons |
|--------|------|------|
| Configured model provider's embedding endpoint | No extra config | Not all providers expose embeddings |
| Ollama (local) | Zero cost, private | Requires local Ollama |
| Dedicated embedding API (e.g., OpenAI, Voyage) | High quality | Extra dependency + cost |
| Lightweight sentence-transformers (local) | Zero cost, fast | Adds `torch` dependency |

Recommended: make `_embed_fn` injectable so the embedding source is pluggable.
Provide a default that uses the configured provider if it supports embeddings,
falling back to a lightweight local option.

---

## Intervention Strategies (for steering plugin)

The drift monitor exposes signals. A separate **steering plugin** (in
jaato-premium) consumes them and decides on interventions:

| Severity | Trigger | Intervention |
|----------|---------|--------------|
| **Soft** | Score below soft threshold | Append reminder to system instructions: "Your current step is: X" |
| **Medium** | Consecutive declining scores | Inject synthetic re-grounding: "Let me refocus on: X" |
| **Hard** | Score below hard threshold | Interrupt tool loop, force plan re-evaluation |
| **Strategic** | Inter-step scores declining | Suggest plan revision to user or agent |

---

## Open Questions

1. **Embedding dimensionality vs cost** — Smaller embeddings are faster to
   compare but may lose semantic nuance. Is 256-dim sufficient or do we need
   1024+?

2. **Turn summary quality** — The summarize-then-embed approach adds an LLM
   call per turn. Is the quality improvement over raw-text embedding worth the
   latency and cost?

3. **Multi-modal drift** — If the agent is working with images or structured
   data, text embeddings may miss drift. Should we support multi-modal
   embeddings?

4. **Calibration** — How do we calibrate thresholds across different task types?
   A coding task and a research task likely have different "normal" similarity
   ranges.

5. **Metric exposure** — Should drift scores be exposed via:
   - OpenTelemetry spans/attributes (fits existing telemetry)?
   - Dedicated drift events in the event protocol?
   - Both?

---

## Next Steps

1. **Implement observation-only `drift_monitor` enrichment plugin** in jaato
   (open source) — embedding computation, score tracking, metrics.
2. **Integrate with steering plugin** in jaato-premium — consume drift signals,
   implement intervention strategies.
3. **Add telemetry** — Expose drift scores as OTel span attributes on
   `jaato.turn` spans.
4. **Calibration study** — Run on representative task types to establish baseline
   threshold ranges.
