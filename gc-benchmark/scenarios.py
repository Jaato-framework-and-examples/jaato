"""Benchmark scenarios with embedded facts for quality testing."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from google.genai import types


@dataclass
class EmbeddedFact:
    """A fact embedded in conversation for retention testing."""

    fact_id: str
    """Unique identifier for this fact."""

    category: str
    """Fact category: 'entity', 'number', 'date', 'decision'."""

    turn_index: int
    """Turn index where the fact was introduced (0-based)."""

    fact_text: str
    """The actual fact content."""

    verification_question: str
    """Question to test if this fact is retained."""

    expected_answer: str
    """Expected answer for the verification question."""


@dataclass
class BenchmarkScenario:
    """A conversation scenario for benchmarking."""

    name: str
    """Scenario identifier."""

    description: str
    """Human-readable description."""

    history: List[types.Content]
    """Conversation history as Content objects."""

    embedded_facts: List[EmbeddedFact]
    """Facts embedded in the conversation for retention testing."""

    target_turns: int
    """Total number of turns in this scenario."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional scenario metadata."""


def _make_content(role: str, text: str) -> types.Content:
    """Create a Content object."""
    return types.Content(
        role=role,
        parts=[types.Part(text=text)]
    )


class ScenarioFactory:
    """Factory for creating benchmark scenarios."""

    @staticmethod
    def short_conversation() -> BenchmarkScenario:
        """10-turn conversation with 5 facts - basic test."""
        history = []
        facts = []

        # Turn 0
        history.append(_make_content("user",
            "I'm starting a new project called Phoenix. "
            "The kickoff date is April 10, 2024 and we have a budget of $75,000."
        ))
        history.append(_make_content("model",
            "Great! I'll help you with Project Phoenix. "
            "With a $75,000 budget and April 10th kickoff, let's plan accordingly."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="project_name",
                category="entity",
                turn_index=0,
                fact_text="Phoenix",
                verification_question="What is the name of the project?",
                expected_answer="Phoenix"
            ),
            EmbeddedFact(
                fact_id="kickoff_date",
                category="date",
                turn_index=0,
                fact_text="April 10, 2024",
                verification_question="What is the project kickoff date?",
                expected_answer="April 10, 2024"
            ),
            EmbeddedFact(
                fact_id="budget",
                category="number",
                turn_index=0,
                fact_text="$75,000",
                verification_question="What is the project budget?",
                expected_answer="$75,000"
            ),
        ])

        # Turn 1
        history.append(_make_content("user",
            "The project manager is Lisa Wong. She'll be overseeing the development team."
        ))
        history.append(_make_content("model",
            "Lisa Wong as PM is a great choice. I'll note her as the main point of contact."
        ))

        facts.append(EmbeddedFact(
            fact_id="pm_name",
            category="entity",
            turn_index=1,
            fact_text="Lisa Wong",
            verification_question="Who is the project manager?",
            expected_answer="Lisa Wong"
        ))

        # Turns 2-4: General conversation padding
        history.append(_make_content("user",
            "What frameworks do you recommend for the frontend?"
        ))
        history.append(_make_content("model",
            "For modern frontends, React or Vue are excellent choices. "
            "React has a larger ecosystem while Vue is more approachable."
        ))

        history.append(_make_content("user",
            "We decided to go with React for this project."
        ))
        history.append(_make_content("model",
            "React is a solid choice. I can help with component architecture and state management."
        ))

        facts.append(EmbeddedFact(
            fact_id="framework_decision",
            category="decision",
            turn_index=3,
            fact_text="React",
            verification_question="What frontend framework was chosen for the project?",
            expected_answer="React"
        ))

        history.append(_make_content("user",
            "Can you help me set up the project structure?"
        ))
        history.append(_make_content("model",
            "Of course! A typical React project structure includes src/, components/, hooks/, etc."
        ))

        return BenchmarkScenario(
            name="short_conversation",
            description="10-turn conversation with 5 facts - basic test",
            history=history,
            embedded_facts=facts,
            target_turns=5,
            metadata={"difficulty": "easy", "fact_density": "medium"}
        )

    @staticmethod
    def long_conversation() -> BenchmarkScenario:
        """50-turn conversation with 15 facts distributed throughout."""
        history = []
        facts = []

        # Early facts (turns 0-5) - most likely to be GC'd
        history.append(_make_content("user",
            "Let me brief you on our company. We're TechVenture Inc, founded in 2019. "
            "Our headquarters is in Austin, Texas."
        ))
        history.append(_make_content("model",
            "Thank you for the background on TechVenture Inc. "
            "Being based in Austin puts you in a great tech hub."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="company_name",
                category="entity",
                turn_index=0,
                fact_text="TechVenture Inc",
                verification_question="What is the company name?",
                expected_answer="TechVenture Inc"
            ),
            EmbeddedFact(
                fact_id="founded_year",
                category="date",
                turn_index=0,
                fact_text="2019",
                verification_question="When was the company founded?",
                expected_answer="2019"
            ),
            EmbeddedFact(
                fact_id="headquarters",
                category="entity",
                turn_index=0,
                fact_text="Austin, Texas",
                verification_question="Where is the company headquarters located?",
                expected_answer="Austin, Texas"
            ),
        ])

        history.append(_make_content("user",
            "Our CEO is Marcus Chen and we have 127 employees across 3 offices."
        ))
        history.append(_make_content("model",
            "127 employees across 3 offices shows good growth. "
            "Marcus Chen must be proud of what TechVenture has built."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="ceo_name",
                category="entity",
                turn_index=1,
                fact_text="Marcus Chen",
                verification_question="Who is the CEO?",
                expected_answer="Marcus Chen"
            ),
            EmbeddedFact(
                fact_id="employee_count",
                category="number",
                turn_index=1,
                fact_text="127",
                verification_question="How many employees does the company have?",
                expected_answer="127"
            ),
        ])

        # Add middle turns with padding
        for i in range(2, 15):
            history.append(_make_content("user",
                f"Let's discuss task {i}. What are your recommendations?"
            ))
            history.append(_make_content("model",
                f"For task {i}, I recommend a structured approach. "
                "Let me know if you need more specific guidance."
            ))

        # Middle facts (turns 15-20)
        history.append(_make_content("user",
            "Our main product is called DataSync Pro. It launched on September 15, 2023."
        ))
        history.append(_make_content("model",
            "DataSync Pro sounds like a data synchronization solution. "
            "How has adoption been since the September launch?"
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="product_name",
                category="entity",
                turn_index=15,
                fact_text="DataSync Pro",
                verification_question="What is the main product called?",
                expected_answer="DataSync Pro"
            ),
            EmbeddedFact(
                fact_id="launch_date",
                category="date",
                turn_index=15,
                fact_text="September 15, 2023",
                verification_question="When did the main product launch?",
                expected_answer="September 15, 2023"
            ),
        ])

        history.append(_make_content("user",
            "We have 2,500 active users and our monthly revenue is $185,000."
        ))
        history.append(_make_content("model",
            "2,500 users generating $185k monthly shows strong product-market fit."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="active_users",
                category="number",
                turn_index=16,
                fact_text="2,500",
                verification_question="How many active users does the product have?",
                expected_answer="2,500"
            ),
            EmbeddedFact(
                fact_id="monthly_revenue",
                category="number",
                turn_index=16,
                fact_text="$185,000",
                verification_question="What is the monthly revenue?",
                expected_answer="$185,000"
            ),
        ])

        # More padding
        for i in range(17, 22):
            history.append(_make_content("user",
                f"Moving to topic {i}. How should we proceed?"
            ))
            history.append(_make_content("model",
                f"For topic {i}, here's my analysis and recommendations."
            ))

        # Late facts (turns 22-25) - should be preserved
        history.append(_make_content("user",
            "We decided to expand to the European market next quarter. "
            "Our target is 500 new customers."
        ))
        history.append(_make_content("model",
            "European expansion is exciting! 500 customers is an ambitious but achievable target."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="expansion_market",
                category="decision",
                turn_index=22,
                fact_text="European market",
                verification_question="Which market is the company expanding to?",
                expected_answer="European market"
            ),
            EmbeddedFact(
                fact_id="customer_target",
                category="number",
                turn_index=22,
                fact_text="500",
                verification_question="What is the target number of new customers?",
                expected_answer="500"
            ),
        ])

        history.append(_make_content("user",
            "The expansion will be led by Sarah Martinez from our Berlin office."
        ))
        history.append(_make_content("model",
            "Sarah Martinez in Berlin is well-positioned to lead the European effort."
        ))

        facts.append(EmbeddedFact(
            fact_id="expansion_lead",
            category="entity",
            turn_index=23,
            fact_text="Sarah Martinez",
            verification_question="Who is leading the expansion?",
            expected_answer="Sarah Martinez"
        ))

        # Final turns
        history.append(_make_content("user",
            "The deadline for the first phase is December 31, 2024."
        ))
        history.append(_make_content("model",
            "December 31, 2024 gives you a clear timeline. Let's work backwards to plan milestones."
        ))

        facts.append(EmbeddedFact(
            fact_id="deadline",
            category="date",
            turn_index=24,
            fact_text="December 31, 2024",
            verification_question="What is the deadline for the first phase?",
            expected_answer="December 31, 2024"
        ))

        return BenchmarkScenario(
            name="long_conversation",
            description="50-turn conversation with 15 facts distributed throughout",
            history=history,
            embedded_facts=facts,
            target_turns=25,
            metadata={"difficulty": "hard", "fact_density": "low"}
        )

    @staticmethod
    def fact_dense() -> BenchmarkScenario:
        """25-turn conversation with 12 facts concentrated in early turns."""
        history = []
        facts = []

        # Turn 0: Dense facts
        history.append(_make_content("user",
            "Here's the full project brief: Project Titan, client Acme Corp, "
            "contact person John Smith, budget $250,000, deadline March 15, 2025, "
            "team size 8 developers."
        ))
        history.append(_make_content("model",
            "I've captured all the details for Project Titan with Acme Corp. "
            "John Smith is the contact, $250k budget, March 15 deadline, 8 developers."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="project",
                category="entity",
                turn_index=0,
                fact_text="Titan",
                verification_question="What is the project code name?",
                expected_answer="Titan"
            ),
            EmbeddedFact(
                fact_id="client",
                category="entity",
                turn_index=0,
                fact_text="Acme Corp",
                verification_question="Who is the client?",
                expected_answer="Acme Corp"
            ),
            EmbeddedFact(
                fact_id="contact",
                category="entity",
                turn_index=0,
                fact_text="John Smith",
                verification_question="Who is the client contact person?",
                expected_answer="John Smith"
            ),
            EmbeddedFact(
                fact_id="budget",
                category="number",
                turn_index=0,
                fact_text="$250,000",
                verification_question="What is the project budget?",
                expected_answer="$250,000"
            ),
            EmbeddedFact(
                fact_id="deadline",
                category="date",
                turn_index=0,
                fact_text="March 15, 2025",
                verification_question="What is the project deadline?",
                expected_answer="March 15, 2025"
            ),
            EmbeddedFact(
                fact_id="team_size",
                category="number",
                turn_index=0,
                fact_text="8",
                verification_question="How many developers are on the team?",
                expected_answer="8"
            ),
        ])

        # Turn 1: More facts
        history.append(_make_content("user",
            "The tech stack will be Python backend, React frontend, PostgreSQL database. "
            "Deployment to AWS, specifically the us-west-2 region."
        ))
        history.append(_make_content("model",
            "Solid stack choice. Python/React/PostgreSQL on AWS us-west-2 is well-supported."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="backend",
                category="decision",
                turn_index=1,
                fact_text="Python",
                verification_question="What language is used for the backend?",
                expected_answer="Python"
            ),
            EmbeddedFact(
                fact_id="database",
                category="decision",
                turn_index=1,
                fact_text="PostgreSQL",
                verification_question="What database is being used?",
                expected_answer="PostgreSQL"
            ),
            EmbeddedFact(
                fact_id="cloud_provider",
                category="decision",
                turn_index=1,
                fact_text="AWS",
                verification_question="Which cloud provider is being used?",
                expected_answer="AWS"
            ),
            EmbeddedFact(
                fact_id="aws_region",
                category="entity",
                turn_index=1,
                fact_text="us-west-2",
                verification_question="Which AWS region is being used?",
                expected_answer="us-west-2"
            ),
        ])

        # Turn 2: Last facts
        history.append(_make_content("user",
            "Sprint length is 2 weeks and we'll have daily standups at 9:30 AM."
        ))
        history.append(_make_content("model",
            "Two-week sprints with 9:30 AM standups - standard agile approach."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="sprint_length",
                category="number",
                turn_index=2,
                fact_text="2 weeks",
                verification_question="How long are the sprints?",
                expected_answer="2 weeks"
            ),
            EmbeddedFact(
                fact_id="standup_time",
                category="entity",
                turn_index=2,
                fact_text="9:30 AM",
                verification_question="What time are daily standups?",
                expected_answer="9:30 AM"
            ),
        ])

        # Padding turns (no new facts, just conversation)
        for i in range(3, 13):
            history.append(_make_content("user",
                f"Let's work on sprint task {i}. What's your approach?"
            ))
            history.append(_make_content("model",
                f"For task {i}, I suggest breaking it into smaller subtasks for the team."
            ))

        return BenchmarkScenario(
            name="fact_dense",
            description="25-turn conversation with 12 facts concentrated in early turns",
            history=history,
            embedded_facts=facts,
            target_turns=13,
            metadata={"difficulty": "hard", "fact_density": "high", "fact_concentration": "early"}
        )

    @staticmethod
    def tool_heavy() -> BenchmarkScenario:
        """30-turn conversation with function calls interspersed."""
        history = []
        facts = []

        # Turn 0: Context with facts
        history.append(_make_content("user",
            "I need help managing my task list for the Alpha project. "
            "The project lead is David Kim."
        ))
        history.append(_make_content("model",
            "I can help manage tasks for Project Alpha. Let me check your current tasks."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="project_name",
                category="entity",
                turn_index=0,
                fact_text="Alpha",
                verification_question="What is the project name?",
                expected_answer="Alpha"
            ),
            EmbeddedFact(
                fact_id="project_lead",
                category="entity",
                turn_index=0,
                fact_text="David Kim",
                verification_question="Who is the project lead?",
                expected_answer="David Kim"
            ),
        ])

        # Turn 1: Simulated function call
        history.append(types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(
                name="list_tasks",
                args={"project": "Alpha"}
            ))]
        ))
        history.append(types.Content(
            role="user",
            parts=[types.Part.from_function_response(
                name="list_tasks",
                response={"tasks": ["Setup CI/CD", "Write docs", "Review PR #42"]}
            )]
        ))
        history.append(_make_content("model",
            "You have 3 tasks: Setup CI/CD, Write docs, and Review PR #42."
        ))

        facts.append(EmbeddedFact(
            fact_id="task_count",
            category="number",
            turn_index=1,
            fact_text="3",
            verification_question="How many tasks were listed?",
            expected_answer="3"
        ))

        # Turn 2: More context
        history.append(_make_content("user",
            "The deadline for CI/CD setup is January 20, 2025. Priority is high."
        ))
        history.append(_make_content("model",
            "I'll note that CI/CD setup is high priority with a January 20th deadline."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="cicd_deadline",
                category="date",
                turn_index=2,
                fact_text="January 20, 2025",
                verification_question="What is the deadline for CI/CD setup?",
                expected_answer="January 20, 2025"
            ),
            EmbeddedFact(
                fact_id="cicd_priority",
                category="decision",
                turn_index=2,
                fact_text="high",
                verification_question="What is the priority of the CI/CD task?",
                expected_answer="high"
            ),
        ])

        # More function call turns
        for i in range(3, 8):
            history.append(_make_content("user",
                f"Update task {i} status to in progress."
            ))
            history.append(types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(
                    name="update_task",
                    args={"task_id": i, "status": "in_progress"}
                ))]
            ))
            history.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name="update_task",
                    response={"success": True, "task_id": i}
                )]
            ))
            history.append(_make_content("model",
                f"Task {i} has been updated to in progress."
            ))

        # Late facts
        history.append(_make_content("user",
            "The documentation owner is Emma Wilson. She needs 5 days to complete it."
        ))
        history.append(_make_content("model",
            "I'll assign documentation to Emma Wilson with a 5-day timeline."
        ))

        facts.extend([
            EmbeddedFact(
                fact_id="docs_owner",
                category="entity",
                turn_index=8,
                fact_text="Emma Wilson",
                verification_question="Who owns the documentation task?",
                expected_answer="Emma Wilson"
            ),
            EmbeddedFact(
                fact_id="docs_duration",
                category="number",
                turn_index=8,
                fact_text="5 days",
                verification_question="How long will documentation take?",
                expected_answer="5 days"
            ),
        ])

        return BenchmarkScenario(
            name="tool_heavy",
            description="30-turn conversation with function calls interspersed",
            history=history,
            embedded_facts=facts,
            target_turns=9,
            metadata={"difficulty": "medium", "has_function_calls": True}
        )

    @staticmethod
    def from_json(path: Path) -> BenchmarkScenario:
        """Load a scenario from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        history = []
        for turn in data["turns"]:
            history.append(_make_content(turn["role"], turn["text"]))

        facts = []
        for fact_data in data.get("facts", []):
            facts.append(EmbeddedFact(
                fact_id=fact_data["id"],
                category=fact_data["category"],
                turn_index=fact_data["turn_index"],
                fact_text=fact_data["text"],
                verification_question=fact_data["question"],
                expected_answer=fact_data["answer"]
            ))

        return BenchmarkScenario(
            name=data["name"],
            description=data.get("description", ""),
            history=history,
            embedded_facts=facts,
            target_turns=len(data["turns"]) // 2,
            metadata=data.get("metadata", {})
        )

    @classmethod
    def get_all_scenarios(cls) -> List[BenchmarkScenario]:
        """Get all built-in scenarios."""
        return [
            cls.short_conversation(),
            cls.long_conversation(),
            cls.fact_dense(),
            cls.tool_heavy(),
        ]

    @classmethod
    def get_scenario(cls, name: str) -> BenchmarkScenario:
        """Get a scenario by name."""
        scenarios = {
            "short_conversation": cls.short_conversation,
            "long_conversation": cls.long_conversation,
            "fact_dense": cls.fact_dense,
            "tool_heavy": cls.tool_heavy,
        }
        if name not in scenarios:
            available = list(scenarios.keys())
            raise ValueError(f"Unknown scenario '{name}'. Available: {available}")
        return scenarios[name]()
