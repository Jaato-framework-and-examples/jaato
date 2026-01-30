# JAATO Model Harness Architecture

## Executive Summary

A **model harness** is the complete runtime environment that wraps an AI model, transforming raw language model capabilities into a controlled, capable, and safe agentic system. JAATO's harness comprises three interconnected layers: **Instructions** (what the model knows), **Tools** (what the model can do), and **Permissions** (what the model is allowed to do). Together, these layers create a structured channel through which model intelligence flows to produce real-world effects.

---

## Part 1: What is a Model Harness?

### Definition

A **model harness** is the infrastructure that:
1. **Configures** the model with context, instructions, and capabilities
2. **Mediates** all interactions between the model and the external world
3. **Enforces** safety boundaries and operational constraints
4. **Tracks** resource usage, actions, and outcomes

### The Harness Metaphor

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│    Without Harness:              With Harness:                       │
│                                                                      │
│    ┌─────────┐                   ┌─────────────────────────────┐    │
│    │         │                   │        HARNESS               │    │
│    │  MODEL  │ → Raw text        │  ┌─────────────────────┐    │    │
│    │         │   completion      │  │       MODEL         │    │    │
│    └─────────┘                   │  └──────────┬──────────┘    │    │
│                                  │             │                │    │
│    - No tools                    │  ┌──────────▼──────────┐    │    │
│    - No persistence              │  │    INSTRUCTIONS     │    │    │
│    - No safety rails             │  │  (Context & Rules)  │    │    │
│    - No real-world effect        │  └──────────┬──────────┘    │    │
│                                  │             │                │    │
│                                  │  ┌──────────▼──────────┐    │    │
│                                  │  │       TOOLS         │    │    │
│                                  │  │   (Capabilities)    │    │    │
│                                  │  └──────────┬──────────┘    │    │
│                                  │             │                │    │
│                                  │  ┌──────────▼──────────┐    │    │
│                                  │  │    PERMISSIONS      │    │    │
│                                  │  │     (Safety)        │    │    │
│                                  │  └──────────┬──────────┘    │    │
│                                  │             │                │    │
│                                  │             ▼                │    │
│                                  │      Real-world effect      │    │
│                                  └─────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why a Harness is Necessary

| Challenge | Without Harness | With Harness |
|-----------|-----------------|--------------|
| **Capability** | Model can only generate text | Model can execute code, edit files, search web |
| **Context** | Model has no project knowledge | Model understands codebase, conventions, goals |
| **Safety** | No control over actions | Permissions gate sensitive operations |
| **Consistency** | Behavior varies unpredictably | Instructions enforce consistent behavior |
| **Accountability** | No audit trail | All actions logged with metadata |

---

## Part 2: The Three Harness Layers

JAATO's harness consists of three complementary layers, each serving a distinct purpose:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE THREE HARNESS LAYERS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   LAYER 1: INSTRUCTIONS                                      │    │
│  │   "What the model KNOWS"                                     │    │
│  │                                                              │    │
│  │   • Base system instructions (.jaato/system_instructions.md) │    │
│  │   • Session-specific instructions                            │    │
│  │   • Plugin instructions (tool usage guides)                  │    │
│  │   • Framework constants (task completion, parallel tools)    │    │
│  │   • Prompt enrichment (references, templates, memory)        │    │
│  │                                                              │    │
│  │   See: jaato_instruction_sources.md                          │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   LAYER 2: TOOLS                                             │    │
│  │   "What the model CAN DO"                                    │    │
│  │                                                              │    │
│  │   • Core tools (always available: introspection, read, run)  │    │
│  │   • Discoverable tools (on-demand: write, search, delegate)  │    │
│  │   • MCP server tools (external integrations)                 │    │
│  │   • Tool schemas (parameter definitions)                     │    │
│  │   • Execution pipeline (parallel, streaming, background)     │    │
│  │                                                              │    │
│  │   See: jaato_tool_system.md                                  │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   LAYER 3: PERMISSIONS                                       │    │
│  │   "What the model is ALLOWED to do"                          │    │
│  │                                                              │    │
│  │   • Auto-approved tools (safe, read-only operations)         │    │
│  │   • Policy evaluation (whitelist/blacklist rules)            │    │
│  │   • User approval channels (interactive, webhook, file)      │    │
│  │   • Suspension scopes (turn, idle, session)                  │    │
│  │   • Sanitization (security layer)                            │    │
│  │                                                              │    │
│  │   See: jaato_permission_system.md                            │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer Interactions

The layers are not independent—they form a coordinated system:

| Interaction | Example |
|-------------|---------|
| Instructions → Tools | Plugin instructions teach the model HOW to use each tool |
| Instructions → Permissions | Base instructions can mandate permission-seeking behavior |
| Tools → Permissions | Each tool call is gated by the permission system |
| Permissions → Instructions | Permission decisions inject metadata into tool results |
| Tools → Instructions | Tool schemas are part of the model's context budget |

---

## Part 3: Layer 1 - Instructions (The Mind)

Instructions shape the model's understanding, behavior, and decision-making.

### Instruction Assembly

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INSTRUCTION ASSEMBLY                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SESSION START                                                       │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │  1. BASE INSTRUCTIONS                        │  ~0-500 tokens    │
│  │     .jaato/system_instructions.md            │                    │
│  │     (behavioral rules, transparency)         │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │  2. SESSION INSTRUCTIONS                     │  ~0-1,000 tokens  │
│  │     Programmatic customization               │                    │
│  │     (task-specific guidance)                 │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │  3. PLUGIN INSTRUCTIONS                      │  ~200-3,000 tokens│
│  │     Tool usage guides from each plugin       │                    │
│  │     (how to use readFile, run, etc.)         │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │  4. FRAMEWORK CONSTANTS                      │  ~90-160 tokens   │
│  │     Task completion, parallel tools,         │                    │
│  │     sandbox guidance                         │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       ▼                                                              │
│  ASSEMBLED SYSTEM PROMPT (~2,000-4,500 tokens)                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Runtime Enrichment

Beyond static instructions, prompts are dynamically enriched:

```
USER PROMPT → references(20) → template(40) → multimodal(60) → memory(80) → session(90) → ENRICHED PROMPT
```

### Harness Contribution

| Aspect | Contribution |
|--------|--------------|
| **Behavioral Consistency** | Base instructions ensure model follows project conventions |
| **Tool Competence** | Plugin instructions teach correct tool usage |
| **Context Awareness** | Enrichment adds relevant project knowledge |
| **Token Economy** | Deferred tools reduce instruction overhead |

---

## Part 4: Layer 2 - Tools (The Hands)

Tools give the model the ability to affect the world beyond text generation.

### Tool Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOOL ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                    ┌─────────────────────────┐                      │
│                    │     PLUGIN REGISTRY     │                      │
│                    │   (Tool Management)     │                      │
│                    └───────────┬─────────────┘                      │
│                                │                                     │
│           ┌────────────────────┼────────────────────┐               │
│           │                    │                    │               │
│           ▼                    ▼                    ▼               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   CORE TOOLS    │  │  DISCOVERABLE   │  │   MCP TOOLS     │     │
│  │                 │  │     TOOLS       │  │                 │     │
│  │ Always loaded   │  │ On-demand via   │  │ External server │     │
│  │ ~14 tools       │  │ introspection   │  │ integrations    │     │
│  │ ~1,200 tokens   │  │ ~85+ tools      │  │ Dynamic count   │     │
│  │                 │  │ ~8,000+ tokens  │  │                 │     │
│  │ • list_tools    │  │ • updateFile    │  │ • GitHub        │     │
│  │ • get_schemas   │  │ • web_search    │  │ • Jira          │     │
│  │ • readFile      │  │ • delegate      │  │ • Confluence    │     │
│  │ • run           │  │ • grep_content  │  │ • Custom        │     │
│  │ • TODO system   │  │ • ...           │  │ • ...           │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│           │                    │                    │               │
│           └────────────────────┼────────────────────┘               │
│                                │                                     │
│                                ▼                                     │
│                    ┌─────────────────────────┐                      │
│                    │    TOOL EXECUTOR        │                      │
│                    │  (Execution Pipeline)   │                      │
│                    └─────────────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Discovery Workflow

The introspection mechanism bridges core and discoverable tools:

```
Model needs web_search
        │
        ▼
list_tools(category="search")
        │
        ▼
Returns: ["grep_content", "web_search", "ast_search"]
        │
        ▼
get_tool_schemas(names=["web_search"])
        │
        ▼
Tool ACTIVATED → Model can now call web_search()
```

### Harness Contribution

| Aspect | Contribution |
|--------|--------------|
| **Capability Extension** | Model can read/write files, run commands, search web |
| **Token Economy** | Deferred loading saves 70-85% of tool schema tokens |
| **Modularity** | Plugins can be added/removed without core changes |
| **Parallel Execution** | Multiple tools run concurrently for efficiency |

---

## Part 5: Layer 3 - Permissions (The Guardrails)

Permissions ensure the model's capabilities are exercised safely and with appropriate oversight.

### Permission Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERMISSION FLOW                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TOOL CALL: updateFile(path="src/main.py", content="...")           │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │  SUSPENSION CHECK                            │                    │
│  │  _idle_suspended? _turn_suspended? _all?     │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       ▼ (not suspended)                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │  POLICY EVALUATION                           │                    │
│  │  Sanitization → Blacklist → Whitelist → Default                  │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       ├─── ALLOW ───► Execute tool                                  │
│       │                                                              │
│       ├─── DENY ────► Return error with _permission metadata        │
│       │                                                              │
│       └─── ASK ─────► Channel prompt                                │
│                           │                                          │
│                           ▼                                          │
│                    ┌─────────────────┐                              │
│                    │   USER INPUT    │                              │
│                    │  y/n/a/t/i/all  │                              │
│                    └────────┬────────┘                              │
│                             │                                        │
│            ┌────────────────┼────────────────┐                      │
│            ▼                ▼                ▼                      │
│         ALLOW            DENY           REMEMBER                    │
│         (execute)        (block)        (whitelist/blacklist)       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Approval Scopes

```
NARROWEST ◄─────────────────────────────────────────────► WIDEST

  once  <  yes  <  turn  <  idle  <  always  <  all
   │        │       │        │         │         │
   │        │       │        │         │         └─ All tools, session
   │        │       │        │         └─ This tool, session
   │        │       │        └─ All tools, until idle
   │        │       └─ All tools, this turn
   │        └─ This call + learn
   └─ This call only
```

### Harness Contribution

| Aspect | Contribution |
|--------|--------------|
| **Safety** | Dangerous operations require explicit approval |
| **User Control** | User decides what the model can do |
| **Flexibility** | Scoped approvals balance safety and convenience |
| **Auditability** | All decisions logged with metadata |

---

## Part 6: How the Layers Work Together

The harness is more than the sum of its parts—the layers create emergent properties.

### Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE REQUEST LIFECYCLE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  USER: "Add logging to the authentication module"                   │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PROMPT ENRICHMENT (Instructions Layer)                      │    │
│  │  • Inject @references if mentioned                           │    │
│  │  • Add memory hints                                          │    │
│  │  • Apply templates                                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  MODEL GENERATION (with system instructions)                 │    │
│  │  • Base instructions guide behavior                          │    │
│  │  • Plugin instructions inform tool usage                     │    │
│  │  • Model decides: "I need to read auth files first"          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  TOOL CALL: readFile(path="src/auth/login.py")               │    │
│  │  (Tools Layer)                                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PERMISSION CHECK (Permissions Layer)                        │    │
│  │  • readFile is auto-approved                                 │    │
│  │  • No prompt needed                                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  TOOL EXECUTION                                              │    │
│  │  • File contents returned to model                           │    │
│  │  • _permission metadata: {method: "auto_approved"}           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  MODEL GENERATION (continued)                                │    │
│  │  • Model analyzes code, plans changes                        │    │
│  │  • Decides: "I'll add logging with updateFile"               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  TOOL CALL: updateFile(path="src/auth/login.py", ...)        │    │
│  │  (Tools Layer)                                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PERMISSION CHECK (Permissions Layer)                        │    │
│  │  • updateFile requires approval                              │    │
│  │  • Display diff to user                                      │    │
│  │  • Wait for response                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  USER: "y" (approve)                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  TOOL EXECUTION                                              │    │
│  │  • File updated                                              │    │
│  │  • _permission metadata: {method: "user_approved"}           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  MODEL: "I've added logging to the authentication module..."        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Cross-Layer Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROSS-LAYER DEPENDENCIES                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                      INSTRUCTIONS                                    │
│                           │                                          │
│         ┌─────────────────┼─────────────────┐                       │
│         │                 │                 │                       │
│         ▼                 │                 ▼                       │
│  ┌─────────────┐          │          ┌─────────────┐                │
│  │ Tool Usage  │          │          │ Behavioral  │                │
│  │   Guides    │          │          │   Rules     │                │
│  │             │          │          │             │                │
│  │ "Use run    │          │          │ "Ask before │                │
│  │  for shell  │          │          │  destructive│                │
│  │  commands"  │          │          │  actions"   │                │
│  └──────┬──────┘          │          └──────┬──────┘                │
│         │                 │                 │                       │
│         │     ┌───────────┴───────────┐     │                       │
│         │     │                       │     │                       │
│         ▼     ▼                       ▼     ▼                       │
│       TOOLS ◄─────────────────────────────► PERMISSIONS             │
│         │                                         │                  │
│         │  • Tools expose schemas                 │                  │
│         │  • Permissions gate execution           │                  │
│         │  • Results include permission metadata  │                  │
│         │                                         │                  │
│         └─────────────► EXECUTION ◄───────────────┘                  │
│                              │                                       │
│                              ▼                                       │
│                      REAL-WORLD EFFECT                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Harness Properties

The combination of layers creates emergent properties that define the harness character.

### Safety vs Capability Trade-off

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SAFETY-CAPABILITY SPECTRUM                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MAXIMUM SAFETY                              MAXIMUM CAPABILITY      │
│  (Minimal risk)                              (Maximum autonomy)      │
│       │                                              │               │
│       ▼                                              ▼               │
│  ┌─────────┐                                    ┌─────────┐         │
│  │ No tools│                                    │All tools│         │
│  │ No write│                                    │No perms │         │
│  │ Ask all │                                    │Auto-all │         │
│  └─────────┘                                    └─────────┘         │
│                                                                      │
│                     JAATO DEFAULT                                    │
│                          │                                           │
│                          ▼                                           │
│                    ┌───────────┐                                    │
│                    │ Core tools│                                    │
│                    │ Read auto │                                    │
│                    │ Write ask │                                    │
│                    │ Scoped    │                                    │
│                    │ approval  │                                    │
│                    └───────────┘                                    │
│                                                                      │
│  JAATO's default balances:                                          │
│  • Read operations: Auto-approved (low risk, high frequency)        │
│  • Write operations: Require approval (higher risk)                 │
│  • Shell commands: Require approval (highest risk)                  │
│  • Scoped approval: Turn/idle scopes reduce prompt fatigue          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Configuration Dimensions

| Dimension | Conservative | Balanced (Default) | Permissive |
|-----------|--------------|-------------------|------------|
| **Tools** | Core only | Core + discoverable | All loaded |
| **Permissions** | Ask all | Auto-read, ask-write | Auto-all |
| **Instructions** | Extensive guardrails | Standard guidance | Minimal |
| **Scope** | Single approval | Turn/idle scopes | Session-wide |

### Harness Profiles

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HARNESS PROFILES                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SUPERVISED MODE (High-touch human oversight)                       │
│  ├─ Instructions: Detailed behavioral constraints                   │
│  ├─ Tools: Core only, discoverable disabled                         │
│  ├─ Permissions: Ask for every write operation                      │
│  └─ Use case: Sensitive production systems, learning scenarios      │
│                                                                      │
│  COLLABORATIVE MODE (Default - Balanced)                            │
│  ├─ Instructions: Standard guidance                                 │
│  ├─ Tools: Core + discoverable on-demand                            │
│  ├─ Permissions: Auto-read, ask-write with turn/idle scopes         │
│  └─ Use case: General development, code review, refactoring         │
│                                                                      │
│  AUTONOMOUS MODE (Minimal oversight)                                │
│  ├─ Instructions: Minimal, goal-focused                             │
│  ├─ Tools: All tools loaded upfront                                 │
│  ├─ Permissions: Suspended for session (user trusts model)          │
│  └─ Use case: Trusted automation, batch processing, CI/CD           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 8: Token Budget Allocation

The harness consumes context tokens. Understanding the budget helps optimize configuration.

### Budget Breakdown

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HARNESS TOKEN BUDGET                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TYPICAL SESSION (Collaborative Mode)                               │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                                                             │     │
│  │  INSTRUCTIONS LAYER                          ~2,500 tokens │     │
│  │  ├─ Base instructions                           ~200      │     │
│  │  ├─ Plugin instructions (CLI, file_edit, etc.) ~1,800     │     │
│  │  ├─ Permission instructions                     ~150      │     │
│  │  ├─ Framework constants                         ~90       │     │
│  │  └─ Prompt enrichment (varies)                  ~260      │     │
│  │                                                             │     │
│  │  TOOLS LAYER (Deferred Loading)               ~1,200 tokens│     │
│  │  ├─ Core tool schemas                          ~1,200     │     │
│  │  └─ Discoverable schemas (loaded on demand)    +variable  │     │
│  │                                                             │     │
│  │  ─────────────────────────────────────────────────────────  │     │
│  │  TOTAL HARNESS OVERHEAD                       ~3,700 tokens│     │
│  │                                                             │     │
│  │  With 128K context window:                                 │     │
│  │  └─ Harness overhead: ~3% of context                       │     │
│  │                                                             │     │
│  │  With 32K context window:                                  │     │
│  │  └─ Harness overhead: ~12% of context                      │     │
│  │                                                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  OPTIMIZATION LEVERS                                                │
│  ├─ JAATO_DEFERRED_TOOLS=true  → Saves ~7,000 tokens               │
│  ├─ Minimal base instructions  → Saves ~300 tokens                 │
│  ├─ Fewer plugins              → Saves ~200-800 per plugin         │
│  └─ GC (garbage collection)    → Reclaims conversation tokens      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 9: Harness Customization

Each layer can be customized independently to create the desired harness profile.

### Instructions Customization

| Method | Effect |
|--------|--------|
| `.jaato/system_instructions.md` | Project-wide behavioral rules |
| `session.configure(system_instructions=...)` | Session-specific guidance |
| Plugin enable/disable | Control which plugin instructions are included |
| Prompt enrichment plugins | Add context automatically |

### Tools Customization

| Method | Effect |
|--------|--------|
| `JAATO_DEFERRED_TOOLS=false` | Load all tools upfront |
| `registry.expose_tool(name)` | Selectively expose plugins |
| `registry.disable_tool(name)` | Disable specific tools |
| MCP server configuration | Add external tool integrations |

### Permissions Customization

| Method | Effect |
|--------|--------|
| `permissions.json` | Static whitelist/blacklist rules |
| `permissions allow/deny <pattern>` | Session-level rule modifications |
| `permissions suspend` | Disable all permission checks |
| Channel configuration | Change approval mechanism (console, webhook, file) |
| Auto-approved tools | Plugins declare safe tools |

---

## Part 10: Visual Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JAATO MODEL HARNESS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         ┌─────────────────┐                         │
│                         │                 │                         │
│                         │   AI  MODEL     │                         │
│                         │                 │                         │
│                         └────────┬────────┘                         │
│                                  │                                   │
│  ════════════════════════════════╪══════════════════════════════════│
│                                  │                                   │
│                         ┌────────▼────────┐                         │
│                         │  INSTRUCTIONS   │                         │
│                         │    (Mind)       │                         │
│                         │                 │                         │
│                         │ • Base rules    │                         │
│                         │ • Tool guides   │                         │
│                         │ • Enrichment    │                         │
│                         └────────┬────────┘                         │
│                                  │                                   │
│                         ┌────────▼────────┐                         │
│                         │     TOOLS       │                         │
│                         │    (Hands)      │                         │
│                         │                 │                         │
│                         │ • Core (14)     │                         │
│                         │ • Discover (85+)│                         │
│                         │ • MCP (dynamic) │                         │
│                         └────────┬────────┘                         │
│                                  │                                   │
│                         ┌────────▼────────┐                         │
│                         │  PERMISSIONS    │                         │
│                         │  (Guardrails)   │                         │
│                         │                 │                         │
│                         │ • Auto-approve  │                         │
│                         │ • Policy eval   │                         │
│                         │ • User approval │                         │
│                         └────────┬────────┘                         │
│                                  │                                   │
│  ════════════════════════════════╪══════════════════════════════════│
│                                  │                                   │
│                                  ▼                                   │
│                         ┌─────────────────┐                         │
│                         │  REAL WORLD     │                         │
│                         │                 │                         │
│                         │ Files, Shell,   │                         │
│                         │ Web, APIs, ...  │                         │
│                         └─────────────────┘                         │
│                                                                      │
│  ══════════════════════════════════════════════════════════════════ │
│                                                                      │
│   THE HARNESS TRANSFORMS:                                            │
│                                                                      │
│   Raw LLM Capabilities    →    Controlled Agentic System            │
│   ────────────────────         ─────────────────────────            │
│   • Text generation            • Structured tool use                │
│   • Pattern matching           • Permission-gated actions           │
│   • No world model             • Project-aware context              │
│   • No persistence             • Auditable execution                │
│   • No safety rails            • Configurable boundaries            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 11: Related Documentation

| Document | Focus |
|----------|-------|
| [jaato_instruction_sources.md](jaato_instruction_sources.md) | Instruction assembly, token budgets, enrichment pipeline |
| [jaato_tool_system.md](jaato_tool_system.md) | Tool architecture, discoverability, execution flow |
| [jaato_permission_system.md](jaato_permission_system.md) | Permission evaluation, channels, suspension states |

---

## Part 12: Color Coding Suggestion for Infographic

- **Blue:** Instructions layer (knowledge, context, guidance)
- **Green:** Tools layer (capabilities, actions, execution)
- **Red/Orange:** Permissions layer (safety, control, approval)
- **Purple:** Model (the intelligence being harnessed)
- **Gray:** Real-world effects (files, shell, web, APIs)
- **Yellow:** Cross-layer interactions and data flow
