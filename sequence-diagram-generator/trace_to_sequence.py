#!/usr/bin/env python3
"""
Sequence Diagram Generator for CLI vs MCP Traces

This script takes trace files from the cli_mcp_harness.py tool and generates
sequence diagrams showing the interactions between:
- Client Script
- Orchestration Framework
- LLM Model
- Plugin Tools (CLI/MCP)

Usage:
    python trace_to_sequence.py --trace path/to/trace.json --output diagram.pdf
    python trace_to_sequence.py --ledger path/to/ledger.jsonl --output diagram.pdf
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ParticipantType(Enum):
    CLIENT = "Client"
    ORCHESTRATOR = "Orchestrator"
    LLM = "LLM"
    TOOL = "Tool"


@dataclass
class Message:
    """Represents a message/interaction in the sequence diagram."""
    source: ParticipantType
    target: ParticipantType
    label: str
    note: Optional[str] = None
    timestamp: Optional[float] = None
    is_return: bool = False
    tool_name: Optional[str] = None


@dataclass
class SequenceDiagram:
    """Contains all data needed to render a sequence diagram."""
    title: str = "Trace Sequence Diagram"
    messages: list = field(default_factory=list)
    tools_used: set = field(default_factory=set)

    def add_message(self, msg: Message):
        self.messages.append(msg)
        if msg.tool_name:
            self.tools_used.add(msg.tool_name)


class TraceParser:
    """Parses trace and ledger files into sequence diagram data."""

    def __init__(self):
        self.diagram = SequenceDiagram()

    def parse_trace_json(self, trace_path: Path) -> SequenceDiagram:
        """Parse a .trace.json file."""
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)

        # Extract title from filename
        self.diagram.title = f"Trace: {trace_path.stem}"

        # Initial prompt from client to orchestrator
        prompt = trace_data.get('prompt', '')
        prompt_preview = self._truncate(prompt, 50)
        self.diagram.add_message(Message(
            source=ParticipantType.CLIENT,
            target=ParticipantType.ORCHESTRATOR,
            label=f"prompt: {prompt_preview}"
        ))

        # Parse events from summary
        summary = trace_data.get('summary', {})
        events = summary.get('events', [])
        self._parse_events(events)

        # Parse tool_result if present (contains trace list)
        tool_result = trace_data.get('tool_result', {})
        if isinstance(tool_result, dict):
            trace_list = tool_result.get('trace', [])
            self._parse_trace_list(trace_list)

        # Final response back to client
        final_text = trace_data.get('text', '')
        text_preview = self._truncate(final_text, 60)
        if text_preview:
            self.diagram.add_message(Message(
                source=ParticipantType.ORCHESTRATOR,
                target=ParticipantType.CLIENT,
                label=f"response: {text_preview}",
                is_return=True
            ))

        return self.diagram

    def parse_ledger_jsonl(self, ledger_path: Path) -> SequenceDiagram:
        """Parse a .jsonl ledger file."""
        self.diagram.title = f"Ledger: {ledger_path.stem}"

        events = []
        with open(ledger_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        # Add initial message
        self.diagram.add_message(Message(
            source=ParticipantType.CLIENT,
            target=ParticipantType.ORCHESTRATOR,
            label="start execution"
        ))

        self._parse_events(events)

        # Add final return
        self.diagram.add_message(Message(
            source=ParticipantType.ORCHESTRATOR,
            target=ParticipantType.CLIENT,
            label="execution complete",
            is_return=True
        ))

        return self.diagram

    def _parse_events(self, events: list):
        """Parse a list of events into messages."""
        for event in events:
            stage = event.get('stage', '')
            ts = event.get('ts')

            if stage == 'pre-count' or stage == 'prompt-count':
                tokens = event.get('total_tokens', 0)
                self.diagram.add_message(Message(
                    source=ParticipantType.ORCHESTRATOR,
                    target=ParticipantType.LLM,
                    label=f"count tokens ({tokens})",
                    timestamp=ts
                ))

            elif stage == 'response':
                prompt_tokens = event.get('prompt_tokens', 0)
                output_tokens = event.get('output_tokens', 0)
                self.diagram.add_message(Message(
                    source=ParticipantType.ORCHESTRATOR,
                    target=ParticipantType.LLM,
                    label=f"generate (in:{prompt_tokens}, out:{output_tokens})",
                    timestamp=ts
                ))
                self.diagram.add_message(Message(
                    source=ParticipantType.LLM,
                    target=ParticipantType.ORCHESTRATOR,
                    label="model response",
                    is_return=True,
                    timestamp=ts
                ))

            elif stage == 'tool-call':
                func_name = event.get('function', 'unknown')
                args = event.get('args', {})
                args_preview = self._truncate(str(args), 40)
                self.diagram.add_message(Message(
                    source=ParticipantType.ORCHESTRATOR,
                    target=ParticipantType.TOOL,
                    label=f"{func_name}({args_preview})",
                    tool_name=func_name,
                    timestamp=ts
                ))

            elif stage == 'tool-result':
                func_name = event.get('function', 'unknown')
                ok = event.get('ok', False)
                status = "success" if ok else "error"
                stdout = event.get('stdout', '')
                result_preview = self._truncate(stdout, 30) if stdout else status
                self.diagram.add_message(Message(
                    source=ParticipantType.TOOL,
                    target=ParticipantType.ORCHESTRATOR,
                    label=f"result: {result_preview}",
                    tool_name=func_name,
                    is_return=True,
                    timestamp=ts
                ))

            elif stage == 'api-error':
                attempt = event.get('attempt', 0)
                error = self._truncate(event.get('error', 'unknown'), 30)
                self.diagram.add_message(Message(
                    source=ParticipantType.LLM,
                    target=ParticipantType.ORCHESTRATOR,
                    label=f"error (attempt {attempt}): {error}",
                    is_return=True,
                    timestamp=ts,
                    note="Retry triggered"
                ))

    def _parse_trace_list(self, trace_list: list):
        """Parse the trace list from tool_result."""
        for turn_data in trace_list:
            turn = turn_data.get('turn', 0)
            text = turn_data.get('text', '')
            function_calls = turn_data.get('function_calls', [])

            # Model response with function calls
            if function_calls:
                for fc in function_calls:
                    name = fc.get('name', 'unknown')
                    args = fc.get('args', {})
                    self.diagram.add_message(Message(
                        source=ParticipantType.LLM,
                        target=ParticipantType.ORCHESTRATOR,
                        label=f"call: {name}",
                        note=f"Turn {turn}"
                    ))

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text and add ellipsis if needed."""
        text = str(text).replace('\n', ' ').strip()
        if len(text) > max_len:
            return text[:max_len-3] + "..."
        return text


class PlantUMLRenderer:
    """Renders sequence diagrams using PlantUML syntax."""

    def __init__(self, diagram: SequenceDiagram):
        self.diagram = diagram

    def to_plantuml(self) -> str:
        """Generate PlantUML syntax for the sequence diagram."""
        lines = [
            "@startuml",
            f"title {self.diagram.title}",
            "",
            "skinparam sequenceArrowThickness 2",
            "skinparam roundcorner 5",
            "skinparam maxmessagesize 200",
            "skinparam sequenceParticipant underline",
            "",
            "participant \"Client Script\" as Client",
            "participant \"Orchestrator\\n(ai_tool_runner)\" as Orchestrator",
            "participant \"LLM\\n(Gemini)\" as LLM",
        ]

        # Add tool participants
        for tool in sorted(self.diagram.tools_used):
            safe_name = tool.replace('-', '_').replace('.', '_')
            lines.append(f'participant "Tool:\\n{tool}" as {safe_name}')

        # Default tool participant if no specific tools
        if not self.diagram.tools_used:
            lines.append('participant "Plugin Tools" as Tool')

        lines.append("")

        # Add messages
        for msg in self.diagram.messages:
            source = self._participant_alias(msg.source, msg.tool_name)
            target = self._participant_alias(msg.target, msg.tool_name)

            arrow = "-->" if msg.is_return else "->"
            label = msg.label.replace('"', "'")

            lines.append(f'{source} {arrow} {target}: {label}')

            if msg.note:
                lines.append(f'note right: {msg.note}')

        lines.append("")
        lines.append("@enduml")

        return "\n".join(lines)

    def _participant_alias(self, ptype: ParticipantType, tool_name: Optional[str] = None) -> str:
        """Get the PlantUML participant alias."""
        if ptype == ParticipantType.TOOL and tool_name:
            return tool_name.replace('-', '_').replace('.', '_')
        elif ptype == ParticipantType.TOOL:
            return "Tool"
        return ptype.value


class MermaidRenderer:
    """Renders sequence diagrams using Mermaid syntax."""

    def __init__(self, diagram: SequenceDiagram):
        self.diagram = diagram

    def to_mermaid(self) -> str:
        """Generate Mermaid syntax for the sequence diagram."""
        lines = [
            "sequenceDiagram",
            f"    title {self.diagram.title}",
            "",
            "    participant Client as Client Script",
            "    participant Orchestrator as Orchestrator<br/>(ai_tool_runner)",
            "    participant LLM as LLM<br/>(Gemini)",
        ]

        # Add tool participants
        for tool in sorted(self.diagram.tools_used):
            safe_name = tool.replace('-', '_').replace('.', '_')
            lines.append(f'    participant {safe_name} as Tool: {tool}')

        if not self.diagram.tools_used:
            lines.append('    participant Tool as Plugin Tools')

        lines.append("")

        # Add messages
        for msg in self.diagram.messages:
            source = self._participant_alias(msg.source, msg.tool_name)
            target = self._participant_alias(msg.target, msg.tool_name)

            arrow = "-->>" if msg.is_return else "->>"
            label = msg.label.replace('"', "'")

            lines.append(f'    {source}{arrow}{target}: {label}')

            if msg.note:
                lines.append(f'    Note right of {target}: {msg.note}')

        return "\n".join(lines)

    def _participant_alias(self, ptype: ParticipantType, tool_name: Optional[str] = None) -> str:
        """Get the Mermaid participant alias."""
        if ptype == ParticipantType.TOOL and tool_name:
            return tool_name.replace('-', '_').replace('.', '_')
        elif ptype == ParticipantType.TOOL:
            return "Tool"
        return ptype.value


class PDFRenderer:
    """Renders sequence diagrams to PDF using various backends."""

    def __init__(self, diagram: SequenceDiagram):
        self.diagram = diagram

    def render_with_plantuml(self, output_path: Path, plantuml_jar: Optional[Path] = None) -> bool:
        """Render using PlantUML (requires Java and plantuml.jar)."""
        puml = PlantUMLRenderer(self.diagram).to_plantuml()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.puml', delete=False) as f:
            f.write(puml)
            puml_path = f.name

        try:
            jar_path = plantuml_jar or self._find_plantuml_jar()
            if not jar_path:
                return False

            cmd = ['java', '-jar', str(jar_path), '-tpdf', puml_path, '-o', str(output_path.parent)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # PlantUML outputs to same name with .pdf extension
                generated = Path(puml_path).with_suffix('.pdf')
                if generated.exists():
                    generated.rename(output_path)
                    return True

            print(f"PlantUML error: {result.stderr}", file=sys.stderr)
            return False
        finally:
            os.unlink(puml_path)

    def render_with_mermaid(self, output_path: Path) -> bool:
        """Render using Mermaid CLI (requires mmdc)."""
        mmd = MermaidRenderer(self.diagram).to_mermaid()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(mmd)
            mmd_path = f.name

        try:
            cmd = ['mmdc', '-i', mmd_path, '-o', str(output_path), '-b', 'white']
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return True

            print(f"Mermaid error: {result.stderr}", file=sys.stderr)
            return False
        finally:
            os.unlink(mmd_path)

    def render_with_reportlab(self, output_path: Path) -> bool:
        """Render using pure Python with ReportLab."""
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib.units import cm, mm
            from reportlab.lib.colors import (
                black, white, lightgrey, darkblue,
                darkgreen, darkorange, darkred, lightblue
            )
            from reportlab.pdfgen import canvas
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            print("ReportLab not installed. Install with: pip install reportlab", file=sys.stderr)
            return False

        # Page setup - landscape for better sequence diagram fit
        page_width, page_height = landscape(A4)
        c = canvas.Canvas(str(output_path), pagesize=landscape(A4))

        # Margins and spacing
        margin = 2 * cm
        participant_height = 1.5 * cm
        message_spacing = 1.2 * cm
        participant_width = 3.5 * cm

        # Calculate participants
        participants = [
            ("Client", "Client Script", darkblue),
            ("Orchestrator", "Orchestrator", darkgreen),
            ("LLM", "LLM (Gemini)", darkorange),
        ]

        for tool in sorted(self.diagram.tools_used):
            participants.append((tool, f"Tool: {tool[:15]}", darkred))

        if not self.diagram.tools_used:
            participants.append(("Tool", "Plugin Tools", darkred))

        # Calculate positions
        num_participants = len(participants)
        available_width = page_width - 2 * margin
        spacing = available_width / (num_participants - 1) if num_participants > 1 else available_width

        participant_x = {}
        for i, (alias, label, color) in enumerate(participants):
            x = margin + i * spacing
            participant_x[alias] = x

        # Draw title
        title_y = page_height - margin
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(black)
        c.drawCentredString(page_width / 2, title_y, self.diagram.title)

        # Draw participant boxes
        box_y = title_y - participant_height - 0.5 * cm
        c.setFont("Helvetica-Bold", 9)

        for alias, label, color in participants:
            x = participant_x[alias]
            # Box
            c.setFillColor(lightblue)
            c.setStrokeColor(color)
            c.setLineWidth(2)
            c.rect(x - participant_width/2, box_y, participant_width, participant_height, fill=1, stroke=1)
            # Label
            c.setFillColor(black)
            # Handle multi-line labels
            lines = label.split('\n') if '\n' in label else [label]
            for i, line in enumerate(lines):
                c.drawCentredString(x, box_y + participant_height/2 - i*10, line)

        # Draw lifelines
        lifeline_start = box_y
        lifeline_end = margin + 1 * cm
        c.setStrokeColor(lightgrey)
        c.setLineWidth(1)
        c.setDash(3, 3)

        for alias in participant_x:
            x = participant_x[alias]
            c.line(x, lifeline_start, x, lifeline_end)

        c.setDash()

        # Draw messages
        current_y = box_y - message_spacing
        c.setFont("Helvetica", 8)

        for msg in self.diagram.messages:
            if current_y < margin + 2 * cm:
                # New page
                c.showPage()
                current_y = page_height - margin - participant_height
                # Redraw participant boxes on new page
                c.setFont("Helvetica-Bold", 9)
                for alias, label, color in participants:
                    x = participant_x[alias]
                    c.setFillColor(lightblue)
                    c.setStrokeColor(color)
                    c.setLineWidth(2)
                    c.rect(x - participant_width/2, current_y, participant_width, participant_height, fill=1, stroke=1)
                    c.setFillColor(black)
                    c.drawCentredString(x, current_y + participant_height/2, label[:20])
                current_y -= message_spacing
                c.setFont("Helvetica", 8)

            source_alias = self._get_alias(msg.source, msg.tool_name, participant_x)
            target_alias = self._get_alias(msg.target, msg.tool_name, participant_x)

            x1 = participant_x.get(source_alias, margin)
            x2 = participant_x.get(target_alias, page_width - margin)

            # Arrow
            c.setStrokeColor(darkblue if not msg.is_return else darkgreen)
            c.setLineWidth(1.5)
            if msg.is_return:
                c.setDash(4, 2)

            c.line(x1, current_y, x2, current_y)

            # Arrowhead
            arrow_dir = 1 if x2 > x1 else -1
            arrow_size = 8
            c.line(x2, current_y, x2 - arrow_dir * arrow_size, current_y + 4)
            c.line(x2, current_y, x2 - arrow_dir * arrow_size, current_y - 4)

            c.setDash()

            # Label
            c.setFillColor(black)
            label = msg.label[:60] + "..." if len(msg.label) > 60 else msg.label
            label_x = (x1 + x2) / 2
            c.drawCentredString(label_x, current_y + 4, label)

            current_y -= message_spacing

        c.save()
        return True

    def _get_alias(self, ptype: ParticipantType, tool_name: Optional[str], participant_x: dict) -> str:
        """Get participant alias for rendering."""
        if ptype == ParticipantType.TOOL:
            if tool_name and tool_name in participant_x:
                return tool_name
            return "Tool"
        return ptype.value

    def _find_plantuml_jar(self) -> Optional[Path]:
        """Try to find plantuml.jar in common locations."""
        common_paths = [
            Path.home() / '.local' / 'share' / 'plantuml' / 'plantuml.jar',
            Path('/usr/share/plantuml/plantuml.jar'),
            Path('/usr/local/share/plantuml/plantuml.jar'),
            Path('plantuml.jar'),
        ]

        for p in common_paths:
            if p.exists():
                return p

        return None

    def render(self, output_path: Path, backend: str = 'auto') -> bool:
        """Render the diagram to PDF using the specified or best available backend."""
        output_path = Path(output_path)

        if backend == 'plantuml':
            return self.render_with_plantuml(output_path)
        elif backend == 'mermaid':
            return self.render_with_mermaid(output_path)
        elif backend == 'reportlab':
            return self.render_with_reportlab(output_path)
        elif backend == 'auto':
            # Try backends in order of preference
            # First try reportlab (pure Python, most reliable)
            if self.render_with_reportlab(output_path):
                return True
            # Then try mermaid
            if self._check_command('mmdc'):
                if self.render_with_mermaid(output_path):
                    return True
            # Finally try plantuml
            if self._check_command('java'):
                if self.render_with_plantuml(output_path):
                    return True

            print("No rendering backend available. Install reportlab: pip install reportlab", file=sys.stderr)
            return False
        else:
            print(f"Unknown backend: {backend}", file=sys.stderr)
            return False

    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([cmd, '--version'], capture_output=True)
            return True
        except FileNotFoundError:
            return False


def export_diagram_source(diagram: SequenceDiagram, output_path: Path, fmt: str):
    """Export diagram source in PlantUML or Mermaid format."""
    if fmt == 'plantuml':
        content = PlantUMLRenderer(diagram).to_plantuml()
    elif fmt == 'mermaid':
        content = MermaidRenderer(diagram).to_mermaid()
    else:
        raise ValueError(f"Unknown format: {fmt}")

    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Exported {fmt} source to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate sequence diagrams from CLI vs MCP traces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate PDF from trace file
    python trace_to_sequence.py --trace traces/cli_get_page_run1.trace.json -o diagram.pdf

    # Generate PDF from ledger file
    python trace_to_sequence.py --ledger cli_get_page_run1.jsonl -o diagram.pdf

    # Export PlantUML source
    python trace_to_sequence.py --trace traces/trace.json --export-plantuml diagram.puml

    # Export Mermaid source
    python trace_to_sequence.py --trace traces/trace.json --export-mermaid diagram.mmd

    # Use specific rendering backend
    python trace_to_sequence.py --trace traces/trace.json -o diagram.pdf --backend reportlab
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--trace', type=Path, help='Path to .trace.json file')
    input_group.add_argument('--ledger', type=Path, help='Path to .jsonl ledger file')

    parser.add_argument('-o', '--output', type=Path, help='Output PDF path')
    parser.add_argument('--backend', choices=['auto', 'plantuml', 'mermaid', 'reportlab'],
                       default='auto', help='Rendering backend (default: auto)')
    parser.add_argument('--export-plantuml', type=Path, help='Export PlantUML source file')
    parser.add_argument('--export-mermaid', type=Path, help='Export Mermaid source file')
    parser.add_argument('--title', type=str, help='Custom diagram title')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Parse input file
    parser_obj = TraceParser()

    if args.trace:
        if not args.trace.exists():
            print(f"Error: Trace file not found: {args.trace}", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(f"Parsing trace file: {args.trace}")
        diagram = parser_obj.parse_trace_json(args.trace)
    else:
        if not args.ledger.exists():
            print(f"Error: Ledger file not found: {args.ledger}", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(f"Parsing ledger file: {args.ledger}")
        diagram = parser_obj.parse_ledger_jsonl(args.ledger)

    # Override title if provided
    if args.title:
        diagram.title = args.title

    if args.verbose:
        print(f"Found {len(diagram.messages)} messages, {len(diagram.tools_used)} unique tools")

    # Export source files if requested
    if args.export_plantuml:
        export_diagram_source(diagram, args.export_plantuml, 'plantuml')

    if args.export_mermaid:
        export_diagram_source(diagram, args.export_mermaid, 'mermaid')

    # Render PDF if output specified
    if args.output:
        if args.verbose:
            print(f"Rendering PDF with backend: {args.backend}")

        renderer = PDFRenderer(diagram)
        success = renderer.render(args.output, args.backend)

        if success:
            print(f"Generated: {args.output}")
        else:
            print("Failed to generate PDF", file=sys.stderr)
            sys.exit(1)
    elif not args.export_plantuml and not args.export_mermaid:
        # No output specified, print PlantUML to stdout
        print(PlantUMLRenderer(diagram).to_plantuml())


if __name__ == '__main__':
    main()
